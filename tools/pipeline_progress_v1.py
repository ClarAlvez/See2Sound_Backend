"""Progresso de etapas e contadores reais da pipeline See2Sound.

Nao depende de Torch, OpenCV, psutil ou outros pacotes. Sem um monitor
registrado, emit_progress() e uma operacao sem efeito (fluxo API preservado).

Percentual global = percentual de ETAPAS alcancadas, nao estimativa de
tempo/CPU. Percentual de subetapa e exibido apenas quando existe total real.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STAGE_NAMES = {
    'metadata': 'Metadados',
    'audio_extraction': 'Extraindo áudio',
    'speech_analysis': 'Whisper / fala',
    'frames': 'Extraindo frames',
    'scenes': 'Detectando cenas',
    'spectra': 'Spectra / modelos visuais',
    'correlation': 'Correlation / identidades',
    'narrative': 'Narrative / Qwen',
    'voice_engine': 'Voice Engine / TTS',
    'saving_artifacts': 'Salvando artefatos',
    'extended_playback': 'FFmpeg / vídeo com pausas',
    'saving_result': 'Salvando resultado final',
}

_MONITOR_LOCK = threading.Lock()
_ACTIVE_MONITOR: Optional['ProgressMonitor'] = None


def emit_progress(stage: str, current=None, total=None, unit='') -> None:
    """Envia progresso de uma rotina interna, sem afetar APIs antigas."""
    with _MONITOR_LOCK:
        monitor = _ACTIVE_MONITOR
    if monitor is not None:
        monitor.tick(stage, current, total, unit)


def _clock(seconds: float) -> str:
    secs = max(0, int(seconds))
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class ProgressMonitor:
    def __init__(self, output_dir, *, interval_seconds=10.0, run_tts=True, stream=None):
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / 'pipeline_progress.json'
        if not (isinstance(interval_seconds, (int, float)) and math.isfinite(interval_seconds) and interval_seconds >= 1.0):
            raise ValueError('interval_seconds deve ser numero >= 1')
        self.interval_seconds = float(interval_seconds)
        self.stages = list(STAGE_NAMES)
        if not run_tts:
            self.stages = [s for s in self.stages if s not in ('voice_engine','extended_playback')]
        self.stream = sys.stderr if stream is None else stream
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.start_monotonic = time.monotonic()
        self.stage_start_monotonic = self.start_monotonic
        self.last_tick_monotonic = self.start_monotonic
        self.stage = 'inicializando'
        self.stage_index = -1
        self.current = None
        self.total = None
        self.unit = ''
        self.status = 'running'
        self.error = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_message = None

    def __enter__(self):
        global _ACTIVE_MONITOR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with _MONITOR_LOCK:
            if _ACTIVE_MONITOR is not None:
                raise RuntimeError('Ja existe monitor ativo nesta execucao')
            _ACTIVE_MONITOR = self
        self._write_snapshot()
        self._thread = threading.Thread(target=self._heartbeat, name='see2sound-progress', daemon=True)
        self._thread.start()
        self.render()
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_MONITOR
        with self.lock:
            self.status = 'failed' if exc_type is not None else 'completed'
            self.error = (f'{exc_type.__name__}: {exc}' if exc_type is not None else None)
            if exc_type is None:
                self.stage = 'completed'
                self.stage_index = len(self.stages)
                self.current = self.total = None
            self.last_tick_monotonic = time.monotonic()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(5.0, self.interval_seconds + 0.5))
        self.render()
        with _MONITOR_LOCK:
            if _ACTIVE_MONITOR is self:
                _ACTIVE_MONITOR = None
        return False

    def on_stage(self, stage):
        """Callback original de process_video: aceita strings, sem alterar contrato."""
        if not isinstance(stage, str):
            return
        with self.lock:
            now = time.monotonic()
            if self.stage != stage:
                self.stage = stage
                self.stage_start_monotonic = now
                self.current = self.total = None
                self.unit = ''
                self.last_tick_monotonic = now
            if stage in self.stages:
                self.stage_index = max(self.stage_index, self.stages.index(stage))
        self.render()

    def tick(self, stage, current=None, total=None, unit=''):
        with self.lock:
            # Evita eventos antigos de tarefas secundarias sobrescreverem a fase.
            if stage != self.stage or self.status != 'running':
                return
            if isinstance(total, (int,float)) and math.isfinite(total) and total > 0:
                self.total = int(total)
            else:
                self.total = None
            if isinstance(current, (int,float)) and math.isfinite(current) and current >= 0:
                self.current = int(current)
            else:
                self.current = None
            self.unit = str(unit or '')
            self.last_tick_monotonic = time.monotonic()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            total_stages = len(self.stages)
            completed_stages = total_stages if self.status == 'completed' else max(0, self.stage_index)
            sub = None
            if self.current is not None and self.total is not None:
                sub = round(100 * min(self.current, self.total) / self.total, 1)
            return {
                'status': self.status,
                'started_at_utc': self.started_at,
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                'stage': self.stage,
                'stage_label': STAGE_NAMES.get(self.stage, self.stage),
                'completed_stages': completed_stages,
                'total_stages': total_stages,
                'stages_percentage_not_runtime': round(100 * completed_stages / total_stages, 1),
                'stage_current': self.current,
                'stage_total': self.total,
                'stage_percentage': sub,
                'stage_unit': self.unit,
                'elapsed_seconds': round(now - self.start_monotonic, 1),
                'stage_elapsed_seconds': round(now - self.stage_start_monotonic, 1),
                'seconds_since_measured_progress': round(now - self.last_tick_monotonic, 1),
                'error': self.error,
                'note': 'Porcentagem geral = etapas alcancadas, NAO porcentagem de tempo restante. '
                        'Sem contador de subetapa nao e possivel estimar conclusao.',
            }

    def _write_snapshot(self):
        data = self.snapshot()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(self.path.name + '.tmp')
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            os.replace(temp, self.path)
        except OSError as exc:
            print(f'[PROGRESSO] Falha ao salvar JSON de status: {exc}', file=self.stream, flush=True)
        return data

    def render(self):
        data = self._write_snapshot()
        done, total = data['completed_stages'], data['total_stages']
        width = 18
        filled = round(width * done / total)
        bar = '#' * filled + '-' * (width - filled)
        stage = data['stage_label']
        if data['stage_current'] is not None:
            detail = f" | {data['stage_current']}"
            if data['stage_total'] is not None:
                detail += f"/{data['stage_total']} ({data['stage_percentage']:.1f}%)"
            if data['stage_unit']:
                detail += f" {data['stage_unit']}"
        elif data['stage_unit']:
            detail = ' | ' + data['stage_unit']
        else:
            detail = ''
        if data['status'] == 'completed':
            stage = 'Concluído'
        elif data['status'] == 'failed':
            stage = 'Falhou em ' + stage
        inactive = data['seconds_since_measured_progress']
        silent = f' | sem medicao ha {_clock(inactive)}' if inactive >= 180 else ''
        message = (f'\n[PROGRESSO] [{bar}] {done}/{total} etapas | '
                   f"{_clock(data['elapsed_seconds'])} decorrido | {stage}{detail}{silent}")
        try:
            print(message, file=self.stream, flush=True)
            if data['error']:
                print(f"[PROGRESSO] Erro: {data['error']}", file=self.stream, flush=True)
        except (OSError, ValueError):
            pass
        return data

    def _heartbeat(self):
        while not self._stop.wait(self.interval_seconds):
            self.render()
