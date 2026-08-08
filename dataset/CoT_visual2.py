"""
CoT V3       (CoT_visual2.py)

  V2       CoT3           
       :      +    2D BBox        +    
       : 3D      32     
              -     bbox_3d     
              -             relative_position_3d    
       :     CoT       
              - shape_label | major_axis | aspect_ratio
              - relative_position_3d
              - hardness | roughness | reflectivity | transparency

    :
    python CoT_visual2.py --cot cot_tmp_v3/149_000.txt
    python CoT_visual2.py --all
    python CoT_visual2.py --all --limit 20 --cot_dir cot_tmp_v3
"""
from __future__ import annotations

import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


#                                                                            

_PALETTE = [
    (1.00, 0.27, 0.27),   #  
    (0.27, 0.67, 1.00),   #  
    (0.27, 1.00, 0.53),   #  
    (1.00, 0.67, 0.27),   #  
    (0.87, 0.27, 1.00),   #  
    (0.27, 0.93, 0.93),   #  
    (1.00, 1.00, 0.27),   #  
    (1.00, 0.53, 0.27),   #   
    (0.53, 0.27, 1.00),   #   
    (0.27, 1.00, 0.27),   #   
    (1.00, 0.47, 0.71),   #   
    (0.47, 0.87, 0.47),   #   
]


def _color(k: int) -> tuple[float, float, float]:
    return _PALETTE[k % len(_PALETTE)]


#    CoT V3                                                                   

def parse_cot_v3(cot_path: str) -> dict:
    """
       cot_tmp_v3/*.txt         

    Returns:
        {
            'part_count' : int,
            'bbox_2d'    : {k: [cx, cy, w, h]},
            'bbox_3d'    : {k: [x0,x1,y0,y1,z0,z1]},
            'inter_pos'  : {k: {j_str: [dir, ...], ...}},  #        
            'primitive'  : {k: {'shape':str, 'major_axis':str, 'aspect_ratio':str}},
            'surface'    : {k: {'hardness':str, 'roughness':str,
                                'reflectivity':str, 'transparency':str}},
        }
    """
    with open(cot_path, 'r', encoding='utf-8') as f:
        text = f.read()

    m    = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    body = m.group(1) if m else text

    #    part_count                                                             
    m = re.search(r'core parts:\s*(\d+)', body)
    part_count = int(m.group(1)) if m else 0

    #    Step 2 bbox_2d & bbox_3d                                   
    # bbox_2d : [x_min, x_max, y_min, y_max]     [0, 1]
    # bbox_3d : [x_min, x_max, y_min, y_max, z_min, z_max]     
    bbox_2d: dict[int, list[float]] = {}
    bbox_3d: dict[int, list[int]]   = {}
    for m in re.finditer(
        r'Part `l_(\d+)`.*?`bbox_2d`\s*=\s*\[([^\]]+)\]'
        r'.*?`bbox_3d`\s*=\s*\[([^\]]+)\]',
        body,
    ):
        k = int(m.group(1))
        bbox_2d[k] = [float(x.strip()) for x in m.group(2).split(',')]
        bbox_3d[k] = [int(x.strip())   for x in m.group(3).split(',')]

    #    Step 3 inter-part relative position                                   
    #    Part `l_k`: `l_j`   ['dir', ...], `l_j2`   [...].
    #       Part `l_k`: no adjacent parts.
    inter_pos: dict[int, dict[str, list[str]]] = {}
    step3_m = re.search(r'Step 3:.*?(?=Step 4:|$)', body, re.DOTALL)
    if step3_m:
        step3_body = step3_m.group(0)
        for m in re.finditer(
            r'Part `l_(\d+)`:\s*(.*?)\.?\s*$', step3_body, re.MULTILINE
        ):
            k       = int(m.group(1))
            content = m.group(2).strip()
            if 'no adjacent parts' in content:
                inter_pos[k] = {}
                continue
            neighbors: dict[str, list[str]] = {}
            for nm in re.finditer(r'`l_(\d+)`\s+is\s+at\s+(\[[^\]]*\])', content):
                j    = nm.group(1)
                dirs = re.findall(r"'(\w+)'", nm.group(2))
                neighbors[j] = dirs
            inter_pos[k] = neighbors

    #    Step 4 primitive shape + major_axis + aspect_ratio                   
    primitive: dict[int, dict[str, str]] = {}
    for m in re.finditer(
        r'Part `l_(\d+)`.*?`shape_label`\s*=\s*(\w+),\s*'
        r'`major_axis`\s*=\s*(\w+),\s*'
        r'`aspect_ratio`\s*=\s*(\w+)',
        body,
    ):
        primitive[int(m.group(1))] = {
            'shape':        m.group(2),
            'major_axis':   m.group(3),
            'aspect_ratio': m.group(4),
        }

    #    Step 5 surface_features                                               
    surface: dict[int, dict[str, str]] = {}
    for m in re.finditer(
        r'Part `l_(\d+)`.*?`hardness`\s*=\s*(\S+?),\s*'
        r'`roughness`\s*=\s*(\S+?),\s*'
        r'`reflectivity`\s*=\s*(\S+?),\s*'
        r'`transparency`\s*=\s*(\w+)',
        body,
    ):
        surface[int(m.group(1))] = {
            'hardness':     m.group(2).rstrip(','),
            'roughness':    m.group(3).rstrip(','),
            'reflectivity': m.group(4).rstrip(','),
            'transparency': m.group(5).rstrip('.'),
        }

    return {
        'part_count': part_count,
        'bbox_2d':    bbox_2d,
        'bbox_3d':    bbox_3d,
        'inter_pos':  inter_pos,
        'primitive':  primitive,
        'surface':    surface,
    }


#                                                                         

def _infer_image_path(cot_path: str, render_dir: str) -> str | None:
    stem = os.path.splitext(os.path.basename(cot_path))[0]
    m    = re.match(r'^(.+)_(\d{3})$', stem)
    if not m:
        return None
    return os.path.join(render_dir, f'{m.group(1)}_', f'{m.group(2)}.png')


#    3D                                                                    

def _draw_bbox3d_wireframe(
    ax3d,
    bbox: list[int],
    color: tuple,
    label: str,
    lw: float = 1.8,
    alpha: float = 0.90,
) -> None:
    """  Axes3D     bbox_3d        part    """
    x0, x1, y0, y1, z0, z1 = bbox

    edges = [
        #   
        [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
        #   
        [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
        #   
        [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)],
    ]
    for seg in edges:
        xs = [seg[0][0], seg[1][0]]
        ys = [seg[0][1], seg[1][1]]
        zs = [seg[0][2], seg[1][2]]
        ax3d.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)

    #       
    faces = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
    ]
    poly = Poly3DCollection(faces, alpha=0.08)
    poly.set_facecolor(color)
    poly.set_edgecolor('none')
    ax3d.add_collection3d(poly)

    #   
    cx, cy, cz = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
    ax3d.text(
        cx, cy, z1 + 0.6, label,
        color=color, fontsize=7.5, fontweight='bold',
        ha='center', va='bottom',
    )


def _bbox3d_center(bbox: list[int]) -> tuple[float, float, float]:
    return (bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2


#                                                                         

def visualize(
    cot_path:   str,
    render_dir: str,
    output_dir: str,
) -> str | None:
    """
         V3                   None 
        -   +2D BBox |   -3D bbox  |   -CoT    
    """
    img_path = _infer_image_path(cot_path, render_dir)
    if img_path is None or not os.path.exists(img_path):
        print(f'[WARN] image not found: {img_path}')
        return None

    cot = parse_cot_v3(cot_path)
    img = np.array(Image.open(img_path).convert('RGB'))
    H, W = img.shape[:2]

    #                                                                       
    fig  = plt.figure(figsize=(18, 9), facecolor='#111111')
    gs   = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[3, 2],
        height_ratios=[3, 2],
        hspace=0.04,
        wspace=0.05,
        left=0.01, right=0.99,
        top=0.95,  bottom=0.02,
    )
    ax_img  = fig.add_subplot(gs[:, 0])
    ax_3d   = fig.add_subplot(gs[0, 1], projection='3d')
    ax_info = fig.add_subplot(gs[1, 1])

    #                                                                            
    #       + 2D BBox
    #                                                                            
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_facecolor('#111111')

    for k, bbox in cot['bbox_2d'].items():
        # bbox_2d      [x_min, x_max, y_min, y_max]     [0, 1]
        x_min, x_max, y_min, y_max = bbox
        bw = x_max - x_min
        bh = y_max - y_min
        if bw < 1e-4 or bh < 1e-4:
            continue
        color = _color(k)
        x0 = x_min * W
        y0 = y_min * H

        rect = mpatches.Rectangle(
            (x0, y0), bw * W, bh * H,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--',
        )
        ax_img.add_patch(rect)
        ax_img.text(
            x0 + bw * W, y0,
            f' l_{k}',
            color='white', fontsize=9, fontweight='bold',
            va='bottom', ha='left',
            bbox=dict(boxstyle='square,pad=0.1',
                      facecolor=color, alpha=0.85, linewidth=0),
        )

    stem = os.path.splitext(os.path.basename(cot_path))[0]
    ax_img.set_title(
        f'{stem}   |   parts: {cot["part_count"]}',
        fontsize=11, color='white', pad=5,
        backgroundcolor='#222222',
    )

    #                                                                            
    #    3D bbox    
    #                                                                            
    ax_3d.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#111111')

    GRID = 32
    ax_3d.set_xlim(0, GRID)
    ax_3d.set_ylim(0, GRID)
    ax_3d.set_zlim(0, GRID)
    ax_3d.set_xlabel('X', color='#aaaaaa', labelpad=2, fontsize=8)
    ax_3d.set_ylabel('Y', color='#aaaaaa', labelpad=2, fontsize=8)
    ax_3d.set_zlabel('Z (up)', color='#aaaaaa', labelpad=2, fontsize=8)
    ax_3d.tick_params(colors='#666666', labelsize=6)
    for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#333333')
    ax_3d.grid(True, color='#333333', linewidth=0.5)
    ax_3d.set_title('3D BBox  (voxel space 32 )',
                    color='#cccccc', fontsize=9, pad=4)

    #       bbox_3d   
    for k, bbox in cot['bbox_3d'].items():
        color = _color(k)
        _draw_bbox3d_wireframe(ax_3d, bbox, color, f'l_{k}')

    #                        i   j 
    drawn_pairs: set[tuple[int, int]] = set()
    for k, neighbors in cot['inter_pos'].items():
        if k not in cot['bbox_3d']:
            continue
        kx, ky, kz = _bbox3d_center(cot['bbox_3d'][k])
        for j_str in neighbors:
            j = int(j_str)
            pair = (min(k, j), max(k, j))
            if pair in drawn_pairs or j not in cot['bbox_3d']:
                continue
            drawn_pairs.add(pair)
            jx, jy, jz = _bbox3d_center(cot['bbox_3d'][j])
            dx, dy, dz = jx - kx, jy - ky, jz - kz
            length = np.sqrt(dx**2 + dy**2 + dz**2)
            if length < 0.5:
                continue
            scale = max(0.0, 1.0 - 1.5 / max(length, 1))
            ax_3d.quiver(
                kx, ky, kz, dx, dy, dz,
                length=scale,
                normalize=False,
                color=_color(k),
                arrow_length_ratio=0.25,
                linewidth=1.2,
                alpha=0.70,
            )

    ax_3d.view_init(elev=22, azim=-55)

    #                                                                            
    #    CoT     
    #                                                                            
    ax_info.set_facecolor('#1a1a1a')
    ax_info.axis('off')

    #         [(text, color, bold)]
    lines: list[tuple[str, str, bool]] = []

    _W   = '#ffffff'   #      
    _Y   = '#d4c97a'   #       
    _G   = '#888888'   #     /   

    #                 
    _DIR_ABBR = {
        'top': ' ', 'bottom': ' ', 'left': ' ', 'right': ' ',
        'front': 'Fr', 'back': 'Bk', 'center': ' ',
    }

    def _fmt_dirs(dirs: list[str]) -> str:
        return ''.join(_DIR_ABBR.get(d, d) for d in dirs)

    #   
    lines.append(('  shape_label | ax | ratio      '
                  'adjacent (  dir)                '
                  'hard | rough | reflect | transp', _G, False))
    lines.append((' ' * 95, _G, False))

    for k in range(cot['part_count']):
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(_color(k)[0]*255),
            int(_color(k)[1]*255),
            int(_color(k)[2]*255),
        )
        prim  = cot['primitive'].get(k, {})
        shape = prim.get('shape', '?')
        axis  = prim.get('major_axis', '?')
        ratio = prim.get('aspect_ratio', '?')

        neighbors = cot['inter_pos'].get(k, {})
        if neighbors:
            adj_str = '  '.join(
                f"l_{j}:{_fmt_dirs(dirs)}"
                for j, dirs in sorted(neighbors.items(), key=lambda x: int(x[0]))
            )
        else:
            adj_str = 'none'

        surf   = cot['surface'].get(k, {})
        hard   = surf.get('hardness', '?')
        rough  = surf.get('roughness', '?')
        refl   = surf.get('reflectivity', '?')
        transp = surf.get('transparency', '?')

        row = (
            f'  l_{k:<2}  {shape:<8}  {axis:<1}  {ratio:<12}  '
            f'{adj_str:<28}  '
            f'{hard:<10} {rough:<10} {refl:<18} {transp}'
        )
        lines.append((row, color_hex, True))

    #              
    n_lines  = max(len(lines), 1)
    font_size = min(9.0, max(5.5, 80.0 / n_lines))

    y_start = 0.97
    dy      = 1.0 / (n_lines + 1)

    for i, (text, color, bold) in enumerate(lines):
        ax_info.text(
            0.01, y_start - i * dy,
            text,
            transform=ax_info.transAxes,
            color=color,
            fontsize=font_size,
            fontfamily='monospace',
            fontweight='bold' if bold else 'normal',
            va='top', ha='left',
        )

    #                                                                        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{stem}.png')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='#111111', dpi=120)
    plt.close(fig)
    return out_path


#                                                                            

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize CoT V3 annotations on rendered images'
    )
    parser.add_argument('--cot',        type=str, default=None,
                        help='Path to a single cot_tmp_v3/*.txt file')
    parser.add_argument('--all',        action='store_true',
                        help='Batch process all files in --cot_dir')
    parser.add_argument('--cot_dir',    type=str, default='./cot_tmp_v3',
                        help='Directory containing cot_tmp_v3 .txt files')
    parser.add_argument('--render_dir', type=str, default='./renders_cond',
                        help='Directory containing renders_cond/{obj_id}_/')
    parser.add_argument('--output_dir', type=str, default='./cot_visual_v3',
                        help='Output directory for visualized images')
    parser.add_argument('--limit',      type=int, default=0,
                        help='Max files in --all mode (0 = no limit)')
    args = parser.parse_args()

    if args.cot:
        out = visualize(args.cot, args.render_dir, args.output_dir)
        if out:
            print(f'Saved   {out}')
        return

    if args.all:
        files = sorted(
            os.path.join(args.cot_dir, fn)
            for fn in os.listdir(args.cot_dir)
            if fn.endswith('.txt')
        )
        if args.limit > 0:
            files = files[:args.limit]
        print(f'Processing {len(files)} files...')
        ok = 0
        for p in files:
            out = visualize(p, args.render_dir, args.output_dir)
            if out:
                ok += 1
                print(f'  [{ok}/{len(files)}] {out}')
            else:
                print(f'  [SKIP] {p}')
        print(f'Done. {ok}/{len(files)} saved to {args.output_dir}/')
        return

    parser.print_help()


if __name__ == '__main__':
    main()
