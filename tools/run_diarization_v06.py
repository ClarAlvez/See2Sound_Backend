"""Standalone pyannote Community-1 smoke test reusing Whisper JSON if available."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.spectra.Correlation.audio.diarization import SpeakerDiarizer, align_transcription
from ai.spectra.Correlation.requesters.transcription import extract_speech_analysis


def main():
    p=argparse.ArgumentParser(description='pyannote local - smoke test See2Sound')
    p.add_argument('--audio-file', required=True)
    p.add_argument('--output', default='data/output/diarization_v06.json')
    p.add_argument('--speech-json', default=None, help='JSON Whisper ou resultado completo da pipeline')
    p.add_argument('--device',choices=['cpu','cuda'],default=None)
    p.add_argument('--num-speakers',type=int,default=None)
    args=p.parse_args()
    result=SpeakerDiarizer(device=args.device).diarize(args.audio_file,num_speakers=args.num_speakers)
    if args.speech_json:
        value=json.loads(Path(args.speech_json).read_text(encoding='utf8'))
        result['aligned_speech_analysis']=align_transcription(extract_speech_analysis(value),result)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('Vozes:',result['speakers'])
    print('Intervalos:',len(result['turns']))
    print('Salvo em:',path)
    for x in result['exclusive_turns'][:12]:
        print(f" {x['start']:7.2f}s -> {x['end']:7.2f}s {x['speaker_id']}")

if __name__=='__main__':
    main()
