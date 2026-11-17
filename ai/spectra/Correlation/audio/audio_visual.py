"""Calls Light-ASD in an isolated Python process to avoid module-name clashes.

We reuse the upstream ASD_Model+classifier weights, but not its CUDA-only demo
or its full-video data preparation and deletion behavior.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def plan_asd_tasks(turns, scene_intervals=None, speech_segments=None, *,
                   max_turn_seconds=8.0, max_total_seconds=180.0,
                   min_turn_seconds=0.60, max_chunks_per_turn=2):
    """Amostragem balanceada e ancorada em nomes, sem repetir inferencia Spectra.

    Prioriza apresentacoes, depois distribui amostras entre falantes e ao longo
    do video. Chunks do mesmo turn_id continuam sendo UMA evidencia de fala.
    """
    from collections import defaultdict
    from ai.spectra.Correlation.speech_identity import detect_mentions, anchored_names

    boundaries = sorted(set(float(scene['end_time']) for scene in (scene_intervals or [])
                            if isinstance(scene, dict) and scene.get('end_time') is not None))
    speech = list(speech_segments or [])
    known = anchored_names(speech)
    by_speaker = defaultdict(list)
    lengths = defaultdict(float)
    for idx, turn in enumerate(turns or []):
        try:
            a, b = float(turn['start']), float(turn['end'])
            sid = str(turn['speaker_id'])
        except (KeyError, ValueError, TypeError):
            continue
        if b-a < min_turn_seconds:
            continue
        lengths[sid] += b-a
        matching = [seg for seg in speech
                    if (seg.get('speaker_id') in (sid, None) and
                        max(0., min(b, float(seg.get('end', 0))) - max(a, float(seg.get('start', 0)))) > .20)]
        names = [m for seg in matching for m in detect_mentions(seg.get('text', ''), known_names=known)]
        priority = (3 if any(m['kind']=='self_introduction' for m in names) else
                    2 if any(m['kind']=='introduction' for m in names) else
                    1 if names else 0)
        cuts = [v for v in boundaries if a+min_turn_seconds < v < b-min_turn_seconds]
        segments = list(zip([a] + cuts, cuts + [b]))
        for start, end in segments:
            if end-start < min_turn_seconds:
                continue
            # Um trecho longo oferece maximo dois chunks, em momentos distintos.
            # Nao gasta todo o orcamento com uma fala extensa.
            max_ch = min(max_chunks_per_turn, max(1, int((end-start) // max_turn_seconds)))
            if end-start >= max_turn_seconds * 1.5:
                max_ch = max(1, min(max_chunks_per_turn, 2))
            for ci in range(max_ch):
                length = min(max_turn_seconds, (end-start) / max_ch)
                middle = start + (ci+.5) * (end-start) / max_ch
                ca = max(start, middle-length/2)
                cb = min(end, ca+length)
                if cb-ca < min_turn_seconds:
                    continue
                by_speaker[sid].append({
                    'turn_id': f'{idx:04d}', 'speaker_id': sid,
                    'start': round(ca,4), 'end': round(cb,4),
                    'priority': priority,
                })
    if not by_speaker:
        return [], {'policy':'name_anchored_speaker_balanced','selected_speakers':[],
                    'name_anchor_chunks':0, 'budget_seconds':max_total_seconds}
    # Primeiro os trechos de apresentacao (ao menos um por falante).
    picked = []
    used = set()
    budget = float(max_total_seconds)
    def take(row):
        nonlocal budget
        key = (row['turn_id'], row['speaker_id'], row['start'])
        duration = row['end']-row['start']
        if key in used or duration > budget + 1e-6:
            return False
        picked.append(row)
        used.add(key)
        budget -= duration
        return True
    anchors = sorted((r for group in by_speaker.values() for r in group if r['priority'] >= 2),
                     key=lambda r:(-r['priority'],r['start']))
    anchor_per_speaker = set()
    for row in anchors:
        if row['speaker_id'] not in anchor_per_speaker and take(row):
            anchor_per_speaker.add(row['speaker_id'])
    # Distribuicao longitudinal: inicio / meio / fim por falante, e rodadas
    # adicionais apenas se houver orcamento. Prioriza vozes recorrentes.
    speakers = sorted(by_speaker, key=lambda sid:(-lengths[sid],sid))
    global_end = max(r['end'] for group in by_speaker.values() for r in group)
    targets = [.15, .55, .85, .32, .72, .08, .95]
    for target in targets:
        if budget < min_turn_seconds:
            break
        for sid in speakers:
            rows = [r for r in by_speaker[sid]
                    if (r['turn_id'],r['speaker_id'],r['start']) not in used]
            if not rows:
                continue
            already_turns = {x['turn_id'] for x in picked if x['speaker_id']==sid}
            rows.sort(key=lambda r:(r['turn_id'] in already_turns,
                                    -r['priority'],
                                    abs((r['start']+r['end'])/(2*max(global_end,1))-target)))
            for row in rows:
                if take(row):
                    break
    # Resultados em ordem de tempo para leitura de video eficiente.
    picked.sort(key=lambda r:(r['start'],r['turn_id']))
    tasks = [{'turn_id':row['turn_id'], 'chunk_id':f"{row['turn_id']}_{idx:03d}",
              'speaker_id':row['speaker_id'], 'start':row['start'], 'end':row['end']}
             for idx,row in enumerate(picked)]
    return tasks, {'policy':'name_anchored_speaker_balanced',
                   'selected_speakers': sorted(set(t['speaker_id'] for t in tasks)),
                   'name_anchor_chunks': sum(r['priority'] >= 2 for r in picked),
                   'budget_seconds':max_total_seconds,
                   'used_seconds':round(max_total_seconds-budget,3),
                   'available_speakers':len(by_speaker),
                   'selected_turn_ids':len({t['turn_id'] for t in tasks})}

class LightASDRunner:
    def __init__(self, light_asd_root='third_party/Light-ASD', *, weights=None,
                 device='auto', timeout_seconds=900, max_turn_seconds=8.0,
                 max_total_seconds=180.0, min_turn_seconds=0.60):
        self.root = Path(light_asd_root).resolve()
        self.weights = Path(weights).resolve() if weights else self.root/'weight'/'pretrain_AVA_CVPR.model'
        self.device = device
        self.timeout_seconds = int(timeout_seconds)
        self.max_turn_seconds = float(max_turn_seconds)
        self.max_total_seconds = float(max_total_seconds)
        self.min_turn_seconds = float(min_turn_seconds)

    def run(self, video_path, audio_path, turns, processed_frames, *, scene_intervals=None, speech_segments=None):
        if not (self.root/'model'/'Model.py').exists():
            raise FileNotFoundError(f'Light-ASD nao encontrado: {self.root}')
        if not self.weights.exists():
            raise FileNotFoundError(f'Pesos Light-ASD nao encontrados: {self.weights}')
        if not Path(video_path).is_file() or not Path(audio_path).is_file():
            raise FileNotFoundError('Light-ASD precisa dos caminhos validos do video e audio extraido.')
        tasks, sampling = plan_asd_tasks(
            turns, scene_intervals=scene_intervals, speech_segments=speech_segments,
            max_turn_seconds=self.max_turn_seconds, max_total_seconds=self.max_total_seconds,
            min_turn_seconds=self.min_turn_seconds)
        if not tasks:
            return {'available': False, 'source': 'light_asd', 'turn_scores': [],
                    'selection': sampling,
                    'warnings': ['Nenhum trecho de fala longo o suficiente para ASD.']}
        data = {'light_asd_root': str(self.root), 'weights': str(self.weights),
                'video_path': str(Path(video_path).resolve()), 'audio_path': str(Path(audio_path).resolve()),
                'device': self.device, 'turns': tasks,
                'frames': [{'timestamp': float(f['timestamp']), 'scene_id': f.get('scene_id'),
                            'people': [{'entity_id':p.get('entity_id'),'bbox':p.get('bbox')}
                                       for p in f.get('people',[]) if p.get('entity_id') and p.get('bbox')]}
                           for f in processed_frames],
                'scene_intervals': scene_intervals or []}
        worker = Path(__file__).with_name('light_asd_worker.py')
        with tempfile.TemporaryDirectory(prefix='see2sound_asd_') as directory:
            inp, outp = Path(directory)/'input.json',Path(directory)/'output.json'
            inp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
            command = [sys.executable, str(worker), '--input', str(inp), '--output', str(outp)]
            completed = subprocess.run(command, cwd=str(self.root), capture_output=True,
                                       text=True, timeout=self.timeout_seconds, check=False)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or '')[-1800:]
                raise RuntimeError(f'Light-ASD falhou (exit {completed.returncode}): {detail}')
            if not outp.exists():
                raise RuntimeError('Light-ASD nao gerou arquivo JSON de resposta.')
            result = json.loads(outp.read_text(encoding='utf8'))
            result['selection'] = sampling
            return result
