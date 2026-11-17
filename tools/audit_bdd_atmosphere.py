"""Audita anotações versus arquivos BDD100K sem alterar datasets.

Execute: python -m tools.audit_bdd_atmosphere --external-root data/external
"""
import argparse
from collections import Counter
from pathlib import Path

from tools import builder_atmosphere as legacy


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--external-root', default='data/external')
    args = parser.parse_args()
    root = Path(args.external_root)
    image_root = legacy.find_bdd_images_root(root)
    label_files = legacy.find_bdd_label_files(root)
    if image_root is None:
        raise RuntimeError('Pasta images/100k não encontrada')
    print('Imagens BDD:', image_root, flush=True)
    names_by_split = {}
    for split in ('train', 'val', 'test'):
        sub = image_root / split
        direct_count = 0
        if sub.exists():
            direct_count = sum(1 for p in sub.iterdir()
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            image_files = [p for p in sub.rglob('*')
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            names_by_split[split] = {p.name.lower() for p in image_files}
            print(f'Pasta {split}: {direct_count} diretamente, '
                  f'{len(image_files)} recursivamente, {len(names_by_split[split])} nomes únicos', flush=True)
        else:
            names_by_split[split] = set()
            print(f'Pasta {split}: NÃO EXISTE', flush=True)

    print('\nArquivos JSON selecionados pelo builder legado:', len(label_files), flush=True)
    for path in label_files:
        print('  ', path, flush=True)
    if not label_files:
        raise RuntimeError('Nenhum JSON com bdd100k+labels+images no nome foi selecionado')

    for path in label_files:
        source_split = legacy.infer_bdd_split(path)
        expected = {'train':'train', 'validation':'val', 'test':'test'}.get(source_split)
        data = legacy.load_json(path)
        if isinstance(data, dict):
            data = data.get('frames') or data.get('images') or data.get('data') or []
        if not isinstance(data, list):
            print('ERRO: payload não é lista:', type(data).__name__, path, flush=True)
            continue
        stats = Counter()
        times, weather = Counter(), Counter()
        sample_missing, sample_attrs = [], []
        for item in data:
            stats['json_entries'] += 1
            name = item.get('name') or item.get('file_name') or item.get('image')
            attrs = item.get('attributes') or {}
            if name is None or not isinstance(attrs, dict):
                stats['bad_record'] += 1
                continue
            if len(sample_attrs) < 3:
                sample_attrs.append({'name': str(name), 'attributes': attrs})
            basename = Path(str(name).replace('\\', '/')).name.lower()
            when = legacy.normalize_text(attrs.get('timeofday'))
            climate = legacy.normalize_text(attrs.get('weather'))
            times[when] += 1
            weather[climate] += 1
            has_time = legacy.BDD_TIME_MAP.get(when) is not None
            has_weather = legacy.BDD_WEATHER_MAP.get(climate) is not None
            if not (has_time or has_weather):
                stats['unmapped_attributes'] += 1
            else:
                stats['mapped_attributes'] += 1
            matches = [split for split, names in names_by_split.items() if basename in names]
            if not matches:
                stats['image_missing'] += 1
                if len(sample_missing) < 8:
                    sample_missing.append(str(name))
            else:
                stats['image_found_any_split'] += 1
                if expected in matches:
                    stats['image_found_expected_split'] += 1
                else:
                    stats['image_found_wrong_split'] += 1
            if matches and (has_time or has_weather):
                stats['eligible_rows_any_split'] += 1
                if expected in matches:
                    stats['eligible_rows_expected_split'] += 1
        print('\n' + '=' * 75)
        print('JSON:', path)
        print('Identificação source_split:', source_split, '| pasta esperada:', expected)
        print('ESTATÍSTICAS:', dict(stats))
        print('timeofday:', times.most_common(12))
        print('weather:', weather.most_common(12))
        print('Amostras do JSON:', sample_attrs)
        print('Imagens não encontradas (exemplos):', sample_missing)
        print('=' * 75, flush=True)

    print('\nDIAGNÓSTICO: compare json_entries e eligible_rows_expected_split por JSON.')
    print('A contagem é recursiva, inclusive dentro de subpastas dos splits.')
    print('Se train tem poucos registros, o JSON de anotações está incompleto ou o builder seleciona arquivo errado.')
    print('Se muitos registros são unmapped_attributes, investigue timeofday/weather.', flush=True)


if __name__ == '__main__':
    main()
