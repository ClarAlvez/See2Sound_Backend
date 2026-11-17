"""Light-ASD smoke test with existing Correlation JSON: NO PersonNet rerun."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.spectra.Correlation.audio.audio_visual import LightASDRunner
from ai.spectra.Correlation.audio.speaker_linking import SpeakerLinkMemory


def main():
    p=argparse.ArgumentParser(description='Light-ASD no See2Sound (video + Correlation v0.3+)')
    p.add_argument('--video-file',required=True)
    p.add_argument('--audio-file',required=True)
    p.add_argument('--diarization-json',required=True)
    p.add_argument('--correlation-json',required=True)
    p.add_argument('--output',default='data/output/asd_v06.json')
    p.add_argument('--light-asd-root',default='third_party/Light-ASD')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--max-seconds',type=float,default=90.)
    args=p.parse_args()
    diag=json.loads(Path(args.diarization_json).read_text(encoding='utf8'))
    frames=json.loads(Path(args.correlation_json).read_text(encoding='utf8'))
    if not isinstance(frames,dict) or not isinstance(frames.get('frames'),list):
        raise ValueError('O --correlation-json deve conter frames[] com people[] e entity_id.')
    runner=LightASDRunner(args.light_asd_root,device=args.device,max_total_seconds=args.max_seconds)
    result=runner.run(args.video_file,args.audio_file,
                      diag.get('exclusive_turns') or diag.get('turns') or [],frames['frames'])
    ids=frames.get('entities',{})
    links=SpeakerLinkMemory().resolve(result.get('turn_scores',[]),valid_entity_ids=ids)
    output={'active_speaker':result,'speaker_links':links}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf8')
    print('Trechos pontuados:',result.get('turns_scored',0))
    print('Vinculos:',[(sid,link['entity_id'],link['status']) for sid,link in links['links'].items()])
    print('Salvo em:',path)

if __name__=='__main__':
    main()
