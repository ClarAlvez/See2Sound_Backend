"""Preview rápido do Narrative com outputs JSON existentes, sem executar Spectra/Whisper/ASD.

python tools/preview_narrative_v07.py --spectra-json data/output/spectra_outputs.json \
    --correlation-json data/output/correlation_v06.json --output data/output/narrative_preview_v07.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spectra-json', required=True)
    parser.add_argument('--correlation-json', required=True)
    parser.add_argument('--output', default='data/output/narrative_preview_v07.json')
    parser.add_argument('--with-llm', action='store_true', help='Ativa Qwen; padrão é sem LLM para preview rápido')
    parser.add_argument('--narrative-model', default='data/models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf')
    args = parser.parse_args()
    spectra = json.loads(Path(args.spectra_json).read_text(encoding='utf-8'))
    if isinstance(spectra, dict):
        spectra = (spectra.get('spectra') or {}).get('outputs', spectra.get('outputs'))
    if not isinstance(spectra, list):
        parser.error('Esperado um array de cenas ou JSON da pipeline com spectra.outputs.')
    correlation = json.loads(Path(args.correlation_json).read_text(encoding='utf-8'))
    if not isinstance(correlation, dict) or 'frames' not in correlation:
        parser.error('Esperado JSON da Correlation 0.6, com frames[].')

    from ai.narrative.correlation_narrative import CorrelationNarrativeGenerator
    generator = CorrelationNarrativeGenerator(
        model_path=args.narrative_model, use_llm=args.with_llm,
    )
    timeline = generator.generate_timeline_from_dicts(spectra, correlation_result=correlation)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Descrições geradas: {len(timeline)} / cenas: {len(spectra)}')
    for item in timeline:
        print(f"[{item.get('start_time')}–{item.get('end_time')}] {item.get('description')}")
    print('Arquivo:', dest)


if __name__ == '__main__':
    main()
