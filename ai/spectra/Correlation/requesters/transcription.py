"""Whisper -> timeline normalizada; nao executa ASR/diarizacao novamente.

Contrato oficial: pipeline_result['audio']['speech_analysis']['speech_segments'].
Os campos speaker_id e word timestamps sao opcionais e NAO sao gerados
pelo WhisperPauseDetector atual.
"""
from __future__ import annotations
from typing import Any


def extract_speech_analysis(payload: Any):
    """Aceita resultado integral da pipeline ou resultado isolado do Whisper."""
    if not isinstance(payload, dict):
        return {}
    audio = payload.get('audio') or {}
    if isinstance(audio, dict) and isinstance(audio.get('speech_analysis'), dict):
        return audio['speech_analysis']
    for key in ('speech_analysis', 'whisper_result', 'transcription'):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    if any(k in payload for k in ('speech_segments', 'segments', 'full_text')):
        return payload
    return {}


class TranscriptionRequester:
    def request(self, payload: Any):
        analysis = extract_speech_analysis(payload)
        raw = analysis.get('speech_segments') or analysis.get('segments') or []
        results = []
        for index, seg in enumerate(raw):
            if not isinstance(seg, dict):
                continue
            try:
                start, end = float(seg['start']), float(seg['end'])
            except (ValueError, TypeError, KeyError):
                continue
            if start < 0 or end < start:
                continue
            text = str(seg.get('text') or '').strip()
            if not text:
                continue
            speaker = next((seg[key] for key in ('speaker_id', 'speaker', 'speaker_label')
                            if key in seg and seg[key] is not None), None)
            results.append({
                'segment_id': str(seg.get('segment_id') or f'speech_{index:04d}'),
                'start': round(start, 3), 'end': round(end, 3),
                'text': text,
                'speaker_id': str(speaker) if speaker is not None else None,
                'words': seg.get('words') or [],
            })
        results.sort(key=lambda row: (row['start'], row['end']))
        return {
            'language': analysis.get('language'),
            'language_probability': analysis.get('language_probability'),
            'segments': results,
            'available': bool(results),
            'diarization_available': any(row['speaker_id'] for row in results),
            'source': 'whisper_speech_segments',
        }
