"""Gera vídeo estendido e WAV sincronizado quando uma descrição não cabe nas pausas.

Este módulo é um pós-processamento: não modifica a Spectra, o Narrative, o
Whisper ou o algoritmo de mixagem usado quando as descrições cabem no vídeo.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

# A mesma janela de silêncio antes/depois da descrição, em uma pausa estendida.
PAD_SECONDS = 0.10


@dataclass
class _FrozenCue:
    data: dict[str, Any]
    pcm: bytes
    sample_count: int
    start_sample: int = 0


@dataclass
class _Freeze:
    frame: int
    cues: list[_FrozenCue] = field(default_factory=list)
    frame_count: int = 0
    sample_count: int = 0


def _safe_path(value: str, root: Path) -> Path:
    path = Path(value).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Arquivo intermediário não encontrado: {path.name}")
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Arquivo intermediário fora do diretório da geração.")
    return path


def _cue_pcm(path: Path, target_rate: int, target_channels: int) -> tuple[bytes, int]:
    rate, source = wavfile.read(str(path))
    if rate <= 0 or not len(source):
        raise ValueError("Áudio TTS vazio ou com taxa inválida.")

    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        data = source.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        data = source.astype(np.float32)

    if rate != target_rate:
        factor = math.gcd(rate, target_rate)
        data = resample_poly(data, target_rate // factor, rate // factor, axis=0)

    if target_channels == 1:
        if data.ndim == 2:
            data = data.mean(axis=1)
    else:
        if data.ndim == 1:
            data = np.repeat(data[:, np.newaxis], target_channels, axis=1)
        elif data.shape[1] != target_channels:
            data = np.repeat(data.mean(axis=1)[:, np.newaxis], target_channels, axis=1)

    sample_count = len(data)
    if sample_count <= 0:
        raise ValueError("Áudio TTS não contém amostras válidas.")
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype('<i2').tobytes()
    return pcm, sample_count


def _select_frame(
    cue: dict[str, Any],
    fps: float,
    last_frame: int,
    normal_intervals: list[tuple[float, float]],
) -> int:
    # Congela no fim da cena: a imagem descrita já apareceu ao espectador.
    scene_end = cue.get('scene_end_time')
    scene_start = cue.get('scene_start_time')
    preferred = scene_end if scene_end is not None else scene_start
    try:
        seconds = float(preferred)
    except (TypeError, ValueError):
        seconds = 0.0
    if not math.isfinite(seconds):
        seconds = 0.0
    frame = min(max(1, round(seconds * fps)), last_frame)

    # Não interrompe uma descrição que já tinha sido mixada naturalmente.
    for _ in range(len(normal_intervals) + 1):
        t = frame / fps
        hit = next((end for start, end in normal_intervals if start - 0.02 <= t < end + 0.02), None)
        if hit is None:
            break
        following = math.ceil((hit + 0.05) * fps)
        if following <= last_frame:
            frame = max(frame + 1, following)
        else:
            # Na cauda do vídeo, antecede a fala em vez de cortá-la.
            first_start = min(start for start, end in normal_intervals if start - 0.02 <= t < end + 0.02)
            frame = max(1, min(last_frame, math.floor((first_start - 0.05) * fps)))
            break
    return frame


def _build_freezes(
    cues: list[dict[str, Any]],
    root: Path,
    sample_rate: int,
    channels: int,
    fps: float,
    total_frames: int,
) -> list[_Freeze]:
    normal_intervals = sorted(
        (float(c['playback_start']), float(c['playback_end']))
        for c in cues
        if c.get('inserted') and c.get('playback_start') is not None
        and c.get('playback_end') is not None
    )
    grouped: dict[int, _Freeze] = {}
    for cue in cues:
        if cue.get('inserted') or cue.get('skip_reason') is None:
            continue
        # Somente a ausência de espaço é recuperável. Falhas técnicas não são.
        if cue.get('skip_reason') != 'Nenhum espaço de áudio suficientemente longo foi encontrado.':
            continue
        if not cue.get('audio_path') or not cue.get('tts_duration'):
            continue
        tts_path = _safe_path(str(cue['audio_path']), root)
        pcm, count = _cue_pcm(tts_path, sample_rate, channels)
        frame = _select_frame(cue, fps, max(1, total_frames - 1), normal_intervals)
        group = grouped.setdefault(frame, _Freeze(frame=frame))
        group.cues.append(_FrozenCue(data=cue, pcm=pcm, sample_count=count))

    result = []
    for frame, group in sorted(grouped.items()):
        # Arredonda a duração da pausa para uma quantidade inteira de quadros.
        raw_samples = sum(c.sample_count for c in group.cues)
        padding_samples = round(PAD_SECONDS * sample_rate) * 2 * len(group.cues)
        group.frame_count = max(1, math.ceil((raw_samples + padding_samples) * fps / sample_rate))
        group.sample_count = round(group.frame_count * sample_rate / fps)
        result.append(group)
    return result


def _write_extended_wave(source: Path, target: Path, groups: list[_Freeze],
                         fps: float, total_frames: int) -> None:
    with wave.open(str(source), 'rb') as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != 'NONE':
            raise ValueError('WAV original precisa ser PCM de 16 bits.')
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        silence_frame = b'\x00' * (2 * channels)
        source_frame_cursor = 0
        total_source_samples = round(total_frames * sample_rate / fps)

        with wave.open(str(target), 'wb') as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)

            def write_silence(count: int) -> None:
                while count > 0:
                    batch = min(count, 32768)
                    writer.writeframesraw(silence_frame * batch)
                    count -= batch

            def copy_to(sample_target: int) -> None:
                nonlocal source_frame_cursor
                sample_target = max(source_frame_cursor, sample_target)
                while source_frame_cursor < sample_target:
                    requested = min(32768, sample_target - source_frame_cursor)
                    chunk = reader.readframes(requested)
                    written = len(chunk) // len(silence_frame)
                    if chunk:
                        writer.writeframesraw(chunk)
                    if written < requested:
                        write_silence(requested - written)
                    source_frame_cursor += requested

            for group in groups:
                copy_to(round(group.frame * sample_rate / fps))
                placed = 0
                for cue in group.cues:
                    pre = round(PAD_SECONDS * sample_rate)
                    write_silence(pre)
                    placed += pre
                    cue.start_sample = placed
                    writer.writeframesraw(cue.pcm)
                    placed += cue.sample_count
                    post = round(PAD_SECONDS * sample_rate)
                    write_silence(post)
                    placed += post
                if placed > group.sample_count:
                    # A pausa foi dimensionada usando o mesmo arredondamento;
                    # o excesso de <= 1 amostra só ocorre por erro numérico.
                    raise ValueError('Duração da pausa insuficiente para TTS.')
                write_silence(group.sample_count - placed)
            copy_to(total_source_samples)


def _build_filter(groups: list[_Freeze], total_frames: int, fps_text: str) -> str:
    # Uma só decodificação do arquivo e N fatias no mesmo filtro FFmpeg.
    segments = len(groups) + 1
    names = ''.join(f'[src{i}]' for i in range(segments))
    parts = [f'[0:v:0]fps={fps_text},format=yuv420p,split={segments}{names}']
    previous = 0
    for i, pause in enumerate(groups):
        segment_length = pause.frame - previous
        if segment_length < 1:
            raise ValueError('Não há quadros suficientes entre as pausas.')
        # tpad não propaga corretamente EOF depois de trim em várias builds
        # do FFmpeg; loop repete exatamente a última imagem N vezes.
        parts.append(
            f'[src{i}]trim=start_frame={previous}:end_frame={pause.frame},'
            f'loop=loop={pause.frame_count}:size=1:start={segment_length - 1},'
            f'setpts=N/(({fps_text})*TB)[part{i}]'
        )
        previous = pause.frame
    parts.append(
        f'[src{segments-1}]trim=start_frame={previous}:end_frame={total_frames},'
        f'setpts=N/(({fps_text})*TB)[part{segments-1}]'
    )
    inputs = ''.join(f'[part{i}]' for i in range(segments))
    parts.append(f'{inputs}concat=n={segments}:v=1:a=0[outv]')
    return ';'.join(parts)


def _write_video(source: Path, audio: Path, target: Path, groups: list[_Freeze],
                 total_frames: int, fps: Fraction, ffmpeg: str) -> None:
    fps_text = f'{fps.numerator}/{fps.denominator}'
    graph = _build_filter(groups, total_frames, fps_text)
    command = [
        ffmpeg, '-hide_banner', '-nostdin', '-loglevel', 'error', '-y',
        '-i', str(source), '-i', str(audio),
        '-filter_complex', graph,
        '-map', '[outv]', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
        '-pix_fmt', 'yuv420p', '-fps_mode', 'cfr', '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart', '-shortest', str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0 or not target.is_file() or not target.stat().st_size:
        log.error('ffmpeg erro: %s', completed.stderr[-4000:])
        raise RuntimeError('Não foi possível gerar o vídeo com pausas. Verifique FFmpeg/libx264.')


def extend_video_when_needed(video_path: str, result: dict[str, Any], output_dir: str | Path) -> bool:
    """Atualiza ``result`` e os artefatos; retorna False se não há pausas extras.

    Deve ser chamado antes de persistir result.json e marcar o job como completo.
    """
    voice = result.get('voice_engine')
    if not isinstance(voice, dict):
        return False
    cues = voice.get('cues') or []
    if not any(not c.get('inserted') and c.get('audio_path') and c.get('tts_duration')
               and c.get('skip_reason') == 'Nenhum espaço de áudio suficientemente longo foi encontrado.'
               for c in cues):
        return False

    root = Path(output_dir).resolve()
    video = Path(video_path).resolve()
    mixed = _safe_path(str(voice['modified_audio_path']), root)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('FFmpeg não está disponível no PATH do servidor.')
    metadata = result.get('metadata') or {}
    source_fps = float(metadata.get('fps') or 30.0)
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    fps = Fraction(str(min(source_fps, 60.0))).limit_denominator(1001)
    fps_float = float(fps)
    original_duration = float(metadata.get('duration_seconds') or 0.0)
    if not math.isfinite(original_duration) or original_duration <= 0:
        raise ValueError('Duração original inválida para estender o vídeo.')
    total_frames = max(2, round(original_duration * fps_float))

    with wave.open(str(mixed), 'rb') as reader:
        sample_rate, channels = reader.getframerate(), reader.getnchannels()
        if sample_rate <= 0 or channels <= 0:
            raise ValueError('Formato WAV inválido.')

    groups = _build_freezes(cues, root, sample_rate, channels, fps_float, total_frames)
    if not groups:
        return False
    audio_tmp = root / 'voice_engine' / 'audio_with_description_extended.tmp.wav'
    video_tmp = root / 'voice_engine' / 'video_with_pauses.tmp.mp4'
    audio_final = root / 'voice_engine' / 'audio_with_description_extended.wav'
    video_final = root / 'voice_engine' / 'video_with_pauses.mp4'
    try:
        _write_extended_wave(mixed, audio_tmp, groups, fps_float, total_frames)
        _write_video(video, audio_tmp, video_tmp, groups, total_frames, fps, ffmpeg)
        os.replace(audio_tmp, audio_final)
        os.replace(video_tmp, video_final)
    finally:
        audio_tmp.unlink(missing_ok=True)
        video_tmp.unlink(missing_ok=True)

    # Recalcula todos os timestamps em função das novas pausas.
    extended_ids = {id(c.data) for g in groups for c in g.cues}
    for cue in cues:
        if id(cue) in extended_ids:
            continue
        if cue.get('inserted') and cue.get('playback_start') is not None:
            t0 = float(cue['playback_start'])
            t1 = float(cue['playback_end'])
            shift0 = sum(g.frame_count / fps_float for g in groups if g.frame / fps_float <= t0)
            shift1 = sum(g.frame_count / fps_float for g in groups if g.frame / fps_float <= t1)
            cue['playback_start'] = round(t0 + shift0, 3)
            cue['playback_end'] = round(t1 + shift1, 3)

    shift = 0.0
    for group in groups:
        anchor = group.frame / fps_float
        playback_anchor = anchor + shift
        for frozen in group.cues:
            cue = frozen.data
            cue['inserted'] = True
            cue['skip_reason'] = None
            cue['mode'] = 'extended_pause'
            cue['pause_video_at'] = round(anchor, 3)
            cue['freeze_duration'] = round(group.frame_count / fps_float, 3)
            cue['playback_start'] = round(playback_anchor + frozen.start_sample / sample_rate, 3)
            cue['playback_end'] = round(playback_anchor + (frozen.start_sample + frozen.sample_count) / sample_rate, 3)
        shift += group.frame_count / fps_float

    inserted = sum(bool(c.get('inserted')) for c in cues)
    voice['inserted_descriptions'] = inserted
    voice['skipped_descriptions'] = len(cues) - inserted
    voice['modified_audio_path'] = str(audio_final)
    voice['video_with_pauses_path'] = str(video_final)
    diagnostics = voice.setdefault('diagnostics', {})
    failures = int(diagnostics.get('technical_failures', 0))
    diagnostics['extended_descriptions'] = len(extended_ids)
    diagnostics['no_pause_descriptions'] = 0
    diagnostics['outcome'] = 'success' if inserted == len(cues) else ('partial_success' if inserted else 'technical_failure')
    old = {v['index']: v['outcome'] for v in diagnostics.get('cue_outcomes', []) if isinstance(v, dict)}
    diagnostics['cue_outcomes'] = [
        {'index': c['index'], 'outcome': 'inserted_extended' if c.get('mode') == 'extended_pause'
         else old.get(c['index'], 'inserted' if c.get('inserted') else 'unknown')}
        for c in cues
    ]
    artifacts = result.setdefault('artifacts', {})
    artifacts['audio_description'] = str(audio_final)
    artifacts['video_with_pauses'] = str(video_final)
    result['playback'] = {
        'has_extended_pauses': True,
        'extra_duration_seconds': round(shift, 3),
        'duration_seconds': round(total_frames / fps_float + shift, 3),
        'extended_descriptions': len(extended_ids),
    }
    manifest = voice.get('manifest_path')
    if manifest:
        with _safe_path(str(manifest), root).open('w', encoding='utf-8') as handle:
            json.dump(voice, handle, indent=2, ensure_ascii=False)
    log.info('extended_pauses=%s extra_duration=%.3fs failures=%s', len(extended_ids), shift, failures)
    return True
