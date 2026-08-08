"""
       (material_statistics.py)

   tmp/finaljson/     .json         parts[k]['material']    
                     cot/dim5_surface.py     
            5        

    :
    python material_statistics.py
    python material_statistics.py --json_dir ./tmp/finaljson --top 50
"""
from __future__ import annotations

import os
import json
import argparse
from collections import Counter


def parse_materials(filepath: str) -> list[str]:
    """    finaljson                """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [p['material'] for p in data.get('parts', []) if 'material' in p]


def main() -> None:
    parser = argparse.ArgumentParser(description='Material statistics for finaljson')
    parser.add_argument('--json_dir', default='./tmp/finaljson',
                        help='Path to finaljson directory')
    parser.add_argument('--top', type=int, default=0,
                        help='Show only top-N materials (0 = all)')
    args = parser.parse_args()

    if not os.path.isdir(args.json_dir):
        raise FileNotFoundError(f'Directory not found: {args.json_dir}')

    counter: Counter[str] = Counter()
    file_count = 0
    for fname in os.listdir(args.json_dir):
        if not fname.endswith('.json'):
            continue
        materials = parse_materials(os.path.join(args.json_dir, fname))
        counter.update(materials)
        file_count += 1

    total_parts = sum(counter.values())
    total_types = len(counter)
    print(f'\nScanned {file_count} files   {total_parts} part entries, {total_types} unique materials.\n')

    ranked = counter.most_common(args.top if args.top > 0 else None)
    col_w  = max(len(m) for m, _ in ranked) + 2

    print(f'{"Rank":<6} {"Material":<{col_w}} {"Count":>8}  {"Percent":>8}')
    print('-' * (6 + col_w + 20))
    for rank, (mat, cnt) in enumerate(ranked, 1):
        pct = cnt / total_parts * 100
        print(f'{rank:<6} {mat:<{col_w}} {cnt:>8}  {pct:>7.2f}%')

    #      dim5_surface.py                                                
    try:
        from cot.dim5_surface import _MATERIAL_TABLE
        known_keywords = set(_MATERIAL_TABLE.keys())

        unmapped = [
            (mat, cnt) for mat, cnt in counter.most_common()
            if not any(kw in mat.lower() for kw in known_keywords)
        ]

        print(f'\n{"=" * 60}')
        if unmapped:
            pct_unmapped = sum(c for _, c in unmapped) / total_parts * 100
            print(
                f'          : {len(unmapped)}   '
                f'({sum(c for _, c in unmapped)}  , {pct_unmapped:.1f}%)\n'
            )
            for mat, cnt in unmapped:
                print(f'  {mat:<{col_w}} {cnt:>8}  {cnt/total_parts*100:>7.2f}%')
        else:
            print('        dim5_surface.py       ')
    except ImportError:
        print('\n     cot/dim5_surface.py         ')


if __name__ == '__main__':
    main()
