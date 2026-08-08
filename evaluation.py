"""
Master evaluation entry point for PhysX-CoT.

Coverage
--------
- Geometry             : PSNR (normal), CD, F-score          via evaluation_geo.py
- Physical attributes  : Absolute scale, Material, Affordance, Description
                                                              via evaluation_phy.py
- Kinematic parameters : VLM-based geometry/motion ranking    via evaluation_kine.py
                          (aggregated from per-sample GPT outputs)

Pipeline
--------
1. (Optional, --run_render)  Generate per-sample videos from URDFs       (render_urdf.py)
2. (Optional, --run_kine)    Run VLM ranking on pre-composed 2x3 grids   (evaluation_kine.py)
3. Run geometry evaluation                                                (evaluation_geo.py)
4. Run physical-attribute evaluation                                      (evaluation_phy.py)
5. Aggregate kinematic VLM ranks from `./evaluation_video/results/`
6. Save the full summary JSON and print a final report.

The geometry and physical-field evaluators accept the same dataset/output paths.
Incomplete samples are reported and skipped rather than terminating a full batch.

Usage
-----
    # Full computational evaluation (geometry + physical attributes)
    python evaluation.py

    # Including URDF rendering and VLM kinematic eval
    python evaluation.py --run_render --run_kine --kine_method_position A
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

from evaluation_geo import evaluate_geometry


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Subprocess wrappers around existing evaluation scripts
# ---------------------------------------------------------------------------

def _run_script(
    script_path: str,
    *,
    capture: bool,
    cwd: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script_path, *(args or [])],
        capture_output=capture,
        text=True,
        cwd=cwd or os.path.dirname(os.path.abspath(script_path)) or '.',
    )


def run_render_urdf(script_path: str) -> None:
    print(f"\n>>> Running URDF rendering: {script_path}")
    proc = _run_script(script_path, capture=False)
    if proc.returncode != 0:
        print('[warn] render_urdf.py returned non-zero exit; continuing.')


def run_kine_eval(script_path: str) -> None:
    print(f"\n>>> Running kinematic (VLM) evaluation: {script_path}")
    proc = _run_script(script_path, capture=False)
    if proc.returncode != 0:
        print('[warn] evaluation_kine.py returned non-zero exit; continuing.')


def run_physical_eval(
    script_path: str,
    resultpath: str,
    datasetpath: str,
    namelist_path: Optional[str],
    num_frames: int,
    save_json: Optional[str],
) -> Dict[str, float]:
    """Run evaluation_phy.py and parse its trailing summary lines."""
    print(f"\n>>> Running physical attribute evaluation: {script_path}")

    if not namelist_path or not os.path.exists(namelist_path):
        print(f"[error] namelist_path={namelist_path!r} is required for physical evaluation.")
        return {}
    script_args = [
        '--resultpath', resultpath,
        '--datasetpath', datasetpath,
        '--namelist', namelist_path,
        '--num_frames', str(num_frames),
    ]
    if save_json:
        script_args.extend(['--out', save_json])
    proc = _run_script(script_path, capture=True, args=script_args)

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print('[error] evaluation_phy.py failed.')
        return {}

    metrics: Dict[str, float] = {}
    for key in ['scale', 'affordance', 'material', 'description', 'n_samples_phy', 'n_skip']:
        m = re.search(rf"^{key}:\s+([-\d\.eE]+)", proc.stdout, re.MULTILINE)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass
    return metrics


# ---------------------------------------------------------------------------
# Kinematic VLM result aggregation
# ---------------------------------------------------------------------------

def aggregate_kine_results(
    results_dir: str,
    method_position: str = 'A',
) -> Dict[str, float]:
    """Aggregate per-sample GPT rank JSONs into mean-rank / top-1 statistics.

    The VLM emits per sample::

        {"A": {"geometry_rank": x, "motion_rank": x},
         "B": {...}, "C": {...}, "D": {...}, "E": {...}}

    `method_position` picks which slot corresponds to the model under test.
    """
    if not os.path.isdir(results_dir):
        print(f"[info] kine results not found at {results_dir}; skipping aggregation.")
        return {}

    geometry_ranks, motion_ranks = [], []
    n_parse_fail = 0
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path, 'r') as fp:
                txt = fp.read().strip()
            try:
                data = json.loads(txt)
            except json.JSONDecodeError:
                m = re.search(r"\{[\s\S]*\}", txt)
                data = json.loads(m.group(0)) if m else None
            if not data or method_position not in data:
                n_parse_fail += 1
                continue
            slot = data[method_position]
            g = slot.get('geometry_rank')
            mo = slot.get('motion_rank')
            if g is not None:
                geometry_ranks.append(float(g))
            if mo is not None:
                motion_ranks.append(float(mo))
        except Exception:
            n_parse_fail += 1

    n_g = len(geometry_ranks)
    n_m = len(motion_ranks)
    return {
        'kine_geometry_mean_rank': (sum(geometry_ranks) / n_g) if n_g else float('nan'),
        'kine_motion_mean_rank': (sum(motion_ranks) / n_m) if n_m else float('nan'),
        'kine_top1_geometry': (sum(1 for r in geometry_ranks if r == 1) / n_g) if n_g else float('nan'),
        'kine_top1_motion': (sum(1 for r in motion_ranks if r == 1) / n_m) if n_m else float('nan'),
        'n_samples_kine': n_g,
        'n_parse_fail': n_parse_fail,
        'method_position': method_position,
    }


# ---------------------------------------------------------------------------
# CLI & final report
# ---------------------------------------------------------------------------

_METRIC_DIRECTION = {
    # higher-is-better metrics are marked True
    'psnr_mean': True,
    'cd_mean': False,
    'fscore_mean': True,
    'scale': False,
    'material': True,
    'affordance': True,
    'description': True,
    'kine_geometry_mean_rank': False,
    'kine_motion_mean_rank': False,
    'kine_top1_geometry': True,
    'kine_top1_motion': True,
}


def _print_section(title: str, data: Optional[Dict[str, Any]], fields):
    print(f"\n[{title}]")
    if not data:
        print('  (no data)')
        return
    for label, key, fmt in fields:
        v = data.get(key)
        if v is None or (isinstance(v, float) and v != v):
            print(f"  {label:46s} = N/A")
            continue
        better = _METRIC_DIRECTION.get(key)
        direction = 'higher is better' if better is True else (
            'lower is better' if better is False else ''
        )
        print(f"  {label:46s} = {fmt.format(v)}  ({direction})")


def main():
    parser = argparse.ArgumentParser(description='PhysX-CoT master evaluation')
    parser.add_argument('--resultpath', default='./test_demo',
                        help='Path to model outputs (basic_info.json + objs/{part}/{part}.obj)')
    parser.add_argument('--datasetpath', default='./PhysX_mobility', help='GT dataset root')
    parser.add_argument(
        '--namelist', default='./val_test_list.npy',
        help='Test sample list (.npy). Pass an empty string or a non-existent path to '
             'auto-discover names from resultpath intersect datasetpath/finaljson.'
    )
    parser.add_argument('--out', default='./eval_results/summary.json',
                        help='Aggregated summary output JSON path')

    parser.add_argument('--skip_geo', action='store_true', help='Skip geometry evaluation')
    parser.add_argument('--skip_phy', action='store_true', help='Skip physical-attribute evaluation')
    parser.add_argument('--run_render', action='store_true', help='Also run render_urdf.py')
    parser.add_argument('--run_kine', action='store_true',
                        help='Also run evaluation_kine.py (requires composed grid videos + OpenAI key)')
    parser.add_argument('--skip_kine_aggregate', action='store_true',
                        help='Do not aggregate kinematic rank JSONs in this run')
    parser.add_argument('--kine_method_position', default='A', choices=['A', 'B', 'C', 'D', 'E'],
                        help='Slot of the model under test in the 2x3 grid videos used by kine eval')
    parser.add_argument('--kine_results_dir', default='./evaluation_video/results',
                        help='Directory holding per-sample VLM rank JSONs')

    parser.add_argument('--n_points', type=int, default=10000,
                        help='Number of surface samples for CD / F-score')
    parser.add_argument('--fscore_tau', type=float, default=0.01,
                        help='F-score distance threshold in normalized space')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Number of rendered views for PSNR computation')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    summary: Dict[str, Any] = {'config': vars(args)}

    if args.run_render:
        print('\n========== Stage 1: URDF Rendering ==========')
        run_render_urdf(os.path.join(here, 'render_urdf.py'))

    if args.run_kine:
        print('\n========== Stage 2: Kinematic VLM Ranking ==========')
        run_kine_eval(os.path.join(here, 'evaluation_kine.py'))

    if not args.skip_geo:
        print('\n========== Stage 3: Geometry Evaluation ==========')
        geo = evaluate_geometry(
            resultpath=args.resultpath,
            datasetpath=args.datasetpath,
            namelist_path=args.namelist,
            n_points=args.n_points,
            fscore_tau=args.fscore_tau,
            num_frames=args.num_frames,
            seed=args.seed,
            save_json=os.path.join(os.path.dirname(args.out) or '.', 'geometry.json'),
        )
        summary['geometry'] = geo

    if not args.skip_phy:
        print('\n========== Stage 4: Physical Attribute Evaluation ==========')
        phy = run_physical_eval(
            script_path=os.path.join(here, 'evaluation_phy.py'),
            resultpath=args.resultpath,
            datasetpath=args.datasetpath,
            namelist_path=args.namelist,
            num_frames=args.num_frames,
            save_json=os.path.join(os.path.dirname(args.out) or '.', 'physical.json'),
        )
        summary['physical'] = phy

    if not args.skip_kine_aggregate:
        print('\n========== Stage 5: Kinematic Rank Aggregation ==========')
        kine_dir = args.kine_results_dir if os.path.isabs(args.kine_results_dir) \
            else os.path.join(here, args.kine_results_dir)
        kine = aggregate_kine_results(kine_dir, method_position=args.kine_method_position)
        if kine:
            summary['kinematic'] = kine

    print('\n\n=================== Final Summary ===================')
    _print_section('Geometry', summary.get('geometry'), [
        ('PSNR (normal map)',         'psnr_mean',    '{:.4f}'),
        ('Chamfer Distance',          'cd_mean',      '{:.6f}'),
        (f'F-score @ {args.fscore_tau}', 'fscore_mean', '{:.4f}'),
    ])
    _print_section('Physical Attributes', summary.get('physical'), [
        ('Absolute scale (|err|)',    'scale',        '{:.4f}'),
        ('Material (PSNR)',           'material',     '{:.4f}'),
        ('Affordance (PSNR)',         'affordance',   '{:.4f}'),
        ('Description (PSNR)',        'description',  '{:.4f}'),
    ])
    _print_section('Kinematic (VLM)', summary.get('kinematic'), [
        ('Geometry mean rank',        'kine_geometry_mean_rank', '{:.4f}'),
        ('Motion mean rank',          'kine_motion_mean_rank',   '{:.4f}'),
        ('Geometry top-1 rate',       'kine_top1_geometry',      '{:.4f}'),
        ('Motion top-1 rate',         'kine_top1_motion',        '{:.4f}'),
    ])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {args.out}")


if __name__ == '__main__':
    main()
