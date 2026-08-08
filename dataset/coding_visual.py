"""
coding_visual.py          vs           

           
  1.         (x<<10)|(y<<5)|z     RLE       
  2.         x'*dy*dz+y'*dz+z'    RLE       
  3.                      
  4.      3D        |      |     
  5.    token    ID          

   
  python coding_visual.py --obj 10049              #        
  python coding_visual.py --obj 10049 --parts 0,1  #      
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

VOXEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'partseg')

_PALETTE = [
    '#FF4444', '#4499FF', '#44FF88', '#FF9944',
    '#CC44FF', '#44EEEE', '#FFFF44', '#FF8844',
    '#FF77AA', '#77FF77',
]

def _color(k: int) -> str:
    return _PALETTE[k % len(_PALETTE)]


#       /                                                   

def global_encode(indices: np.ndarray) -> list[int]:
    v = np.asarray(indices, dtype=np.int64)
    ids = (v[:, 0] << 10) | (v[:, 1] << 5) | v[:, 2]
    return sorted(set(ids.tolist()))


def global_decode(ids: list[int]) -> np.ndarray:
    a = np.asarray(ids, dtype=np.int64)
    z = a & 0x1F
    y = (a >> 5) & 0x1F
    x = (a >> 10) & 0x1F
    return np.stack([x, y, z], axis=1)


def local_encode(indices: np.ndarray) -> tuple[list[int], list[int]]:
    v = np.asarray(indices, dtype=np.int64)
    x_min, y_min, z_min = int(v[:, 0].min()), int(v[:, 1].min()), int(v[:, 2].min())
    x_max, y_max, z_max = int(v[:, 0].max()), int(v[:, 1].max()), int(v[:, 2].max())
    dy = y_max - y_min + 1
    dz = z_max - z_min + 1
    local_ids = (v[:, 0] - x_min) * dy * dz + (v[:, 1] - y_min) * dz + (v[:, 2] - z_min)
    bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
    return sorted(set(local_ids.tolist())), bbox


def local_decode(ids: list[int], bbox: list[int]) -> np.ndarray:
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    dy = y_max - y_min + 1
    dz = z_max - z_min + 1
    a = np.asarray(ids, dtype=np.int64)
    x_prime = a // (dy * dz)
    y_prime = (a % (dy * dz)) // dz
    z_prime = a % dz
    return np.stack([x_prime + x_min, y_prime + y_min, z_prime + z_min], axis=1)


#    RLE                                                        

def rle_encode(ids: list[int]) -> str:
    result: list[str] = []
    start = prev = ids[0]
    for n in ids[1:]:
        if n == prev + 1:
            prev = n
        else:
            result.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    result.append(f"{start}-{prev}" if start != prev else str(start))
    return " ".join(result)


def rle_decode(s: str) -> list[int]:
    ids: list[int] = []
    for token in s.split():
        if '-' in token:
            a, b = map(int, token.split('-', 1))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(token))
    return ids


#          +                                             

def _setup_ax(ax, title: str) -> None:
    ax.set_facecolor('#1a1a1a')
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 31)
    ax.set_zlim(0, 31)
    ax.set_xlabel('X', fontsize=6, color='#aaaaaa', labelpad=1)
    ax.set_ylabel('Y', fontsize=6, color='#aaaaaa', labelpad=1)
    ax.set_zlabel('Z', fontsize=6, color='#aaaaaa', labelpad=1)
    ax.tick_params(labelsize=5, colors='#666666')
    ax.grid(True, color='#333333', linewidth=0.3)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a2a')
    ax.set_title(title, fontsize=8, color='#dddddd', pad=3)
    ax.view_init(elev=20, azim=-50)


def process_part(
    obj_id: str,
    part_k: int,
    ax_orig: 'Axes3D',
    ax_global: 'Axes3D',
    ax_local: 'Axes3D',
) -> dict | None:

    npy_path = os.path.join(VOXEL_ROOT, obj_id, '32', f'ind_{part_k}.npy')
    if not os.path.exists(npy_path):
        print(f'[WARN]          {npy_path}')
        return None

    indices = np.load(npy_path).astype(np.int64)
    color   = _color(part_k)

    #                                                      
    g_ids     = global_encode(indices)
    g_rle     = rle_encode(g_ids)
    g_dec     = global_decode(rle_decode(g_rle))

    #                                                      
    l_ids, bbox = local_encode(indices)
    l_rle       = rle_encode(l_ids)
    l_dec       = local_decode(rle_decode(l_rle), bbox)

    #                                                      
    orig_set = {tuple(r) for r in indices.tolist()}
    g_ok     = {tuple(r) for r in g_dec.tolist()} == orig_set
    l_ok     = {tuple(r) for r in l_dec.tolist()} == orig_set

    g_tokens = len(g_rle.split())
    l_tokens = len(l_rle.split())
    dx = bbox[1] - bbox[0] + 1
    dy = bbox[3] - bbox[2] + 1
    dz = bbox[5] - bbox[4] + 1

    #                                                       
    for ax, vox, title in [
        (ax_orig,   indices, f'l_{part_k}  Original\n{len(indices)} voxels'),
        (ax_global, g_dec,   f'l_{part_k}  Global decode\nID [0,{max(g_ids)}]  '
                              f'{"  match" if g_ok else "  MISMATCH"}'),
        (ax_local,  l_dec,   f'l_{part_k}  Local decode\nbbox {dx} {dy} {dz}  '
                              f'ID [0,{max(l_ids)}]  {"  match" if l_ok else "  MISMATCH"}'),
    ]:
        _setup_ax(ax, title)
        ax.scatter(
            vox[:, 0], vox[:, 1], vox[:, 2],
            c=color, s=6, alpha=0.75, depthshade=True,
        )

    return {
        'part':        part_k,
        'voxels':      len(indices),
        'bbox':        bbox,
        'bbox_vol':    dx * dy * dz,
        'fill_rate':   f'{len(indices)/(dx*dy*dz)*100:.1f}%',
        'g_id_max':    max(g_ids),
        'l_id_max':    max(l_ids),
        'g_tokens':    g_tokens,
        'l_tokens':    l_tokens,
        'token_ratio': l_tokens / g_tokens,
        'g_ok':        g_ok,
        'l_ok':        l_ok,
    }


#                                                            

def run(obj_id: str, parts: list[int]) -> None:
    n = len(parts)
    fig = plt.figure(figsize=(15, 5 * n + 0.5), facecolor='#111111')
    fig.suptitle(
        f'Voxel Coding Comparison     Object {obj_id}\n'
        'Column 1: Original  |  Column 2: Global decode  |  Column 3: Local decode',
        color='#eeeeee', fontsize=11, y=1.002,
    )

    stats: list[dict] = []
    for row, part_k in enumerate(parts):
        ax_o = fig.add_subplot(n, 3, row * 3 + 1, projection='3d')
        ax_g = fig.add_subplot(n, 3, row * 3 + 2, projection='3d')
        ax_l = fig.add_subplot(n, 3, row * 3 + 3, projection='3d')
        s = process_part(obj_id, part_k, ax_o, ax_g, ax_l)
        if s:
            stats.append(s)

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.99])

    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coding_visual_out')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{obj_id}_parts{"_".join(str(p) for p in parts)}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor='#111111', pad_inches=0.1)
    plt.close(fig)
    print(f'\n[  ]   {out_path}')

    #                                                       
    W = 92
    print(f'\n{" "*W}')
    print(f'  Object {obj_id}  |       vs        ')
    print(f'{" "*W}')
    hdr = (f'  {"Part":<6} {"Voxels":<8} {"bbox(dx dy dz)":<18} {"Fill%":<8} '
           f'{"G-IDmax":<10} {"L-IDmax":<10} {"G-tok":<8} {"L-tok":<8} '
           f'{"L/G":<8} {"G-OK":<6} {"L-OK"}')
    print(hdr)
    print(f'{" "*W}')
    for s in stats:
        bx = s['bbox']
        dx, dy, dz = bx[1]-bx[0]+1, bx[3]-bx[2]+1, bx[5]-bx[4]+1
        print(
            f'  l_{s["part"]:<4} {s["voxels"]:<8} {f"{dx} {dy} {dz}":<18} '
            f'{s["fill_rate"]:<8} {s["g_id_max"]:<10} {s["l_id_max"]:<10} '
            f'{s["g_tokens"]:<8} {s["l_tokens"]:<8} '
            f'{s["token_ratio"]:.2%}  '
            f'{" " if s["g_ok"] else " "}     {" " if s["l_ok"] else " "}'
        )
    print(f'{" "*W}')
    if stats:
        avg_ratio = sum(s['token_ratio'] for s in stats) / len(stats)
        all_ok = all(s['g_ok'] and s['l_ok'] for s in stats)
        print(f'     token       /    {avg_ratio:.2%}')
        print(f'            {"    " if all_ok else "       "}')
    print(f'{" "*W}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='     vs             ')
    parser.add_argument('--obj',   type=str, default='10049',
                        help='   ID tmp/partseg/{id}/32/     ')
    parser.add_argument('--parts', type=str, default='all',
                        help='           "0,1"    all')
    args = parser.parse_args()

    obj_dir = os.path.join(VOXEL_ROOT, args.obj, '32')
    if not os.path.isdir(obj_dir):
        print(f'[ERROR]       {obj_dir}')
        return

    if args.parts == 'all':
        k, parts = 0, []
        while os.path.exists(os.path.join(obj_dir, f'ind_{k}.npy')):
            parts.append(k)
            k += 1
    else:
        parts = [int(x.strip()) for x in args.parts.split(',')]

    if not parts:
        print('[ERROR]       ind_k.npy   ')
        return

    print(f'[INFO] Object={args.obj}  Parts={parts}')
    run(args.obj, parts)


if __name__ == '__main__':
    main()
