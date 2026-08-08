"""Batch inference runner for the PhysX-CoT asset pipeline.

Stages:
    1_vlm_cot.py       image -> structured physical CoT and voxel coordinates
    2_decoder.py       image + coordinates -> sample.glb
    3_split.py         sample.glb + coordinates -> per-part OBJ meshes
    4_simready_gen.py  physical text + meshes -> JSON metadata and URDF

The runner keeps the stage output contract stable and supports resumable runs.
"""

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Dict, Iterable, List, Optional



# Project root = directory containing this script (so the runner can be invoked
# from anywhere). The four inference scripts live here as well.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_repo_path(path: Optional[str]) -> Optional[str]:
    """Resolve a repository-relative checkpoint path independently of CWD."""
    if path is None or os.path.isabs(path):
        return path
    if path.startswith(("./pretrain", "pretrain", "./configs", "configs")):
        return os.path.join(PROJECT_ROOT, path.lstrip("./"))
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Offline dinov2 cache: 2_decoder.py -> trellis_image_to_3d -> torch.hub.load
# tries to reach github.com + dl.fbaipublicfiles.com. We pre-populate a
# project-local torch hub cache with symlinks to user-provided files and pass
# it via TORCH_HOME so the subprocess never hits the network.
#
# Expected layout produced by the user:
#   {PROJECT_ROOT}/dinov2-main/                          # cloned source
#   {PROJECT_ROOT}/pretrain/dinov2/<*reg4_pretrain.pth>  # weights
# ---------------------------------------------------------------------------

DINOV2_SRC_DIR  = os.path.join(PROJECT_ROOT, 'dinov2-main')
DINOV2_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'pretrain', 'dinov2')
TORCH_HUB_CACHE = os.path.join(PROJECT_ROOT, '.torch_hub_cache')


def _ensure_symlink(target: str, link: str) -> None:
    if os.path.islink(link):
        if os.readlink(link) == target:
            return
        os.unlink(link)
    elif os.path.exists(link):
        # Real file/dir already present; leave it alone.
        return
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(target, link)


def setup_dinov2_cache() -> Optional[str]:
    """Wire {PROJECT_ROOT}/dinov2-main and pretrain/dinov2/*.pth into a torch
    hub cache layout. Returns the TORCH_HOME path or None if pre-reqs missing.
    """
    if not os.path.isdir(DINOV2_SRC_DIR):
        print(f"[warn] dinov2 source not found at {DINOV2_SRC_DIR}; "
              f"stage 2 will fall back to torch.hub online download.")
        return None
    if not os.path.isdir(DINOV2_WEIGHTS_DIR):
        print(f"[warn] dinov2 weights dir not found at {DINOV2_WEIGHTS_DIR}; "
              f"stage 2 will fall back to torch.hub online download.")
        return None

    weights = [
        f for f in os.listdir(DINOV2_WEIGHTS_DIR)
        if f.endswith('.pth')
    ]
    if not weights:
        print(f"[warn] no .pth weights in {DINOV2_WEIGHTS_DIR}; "
              f"stage 2 will fall back to torch.hub online download.")
        return None

    hub_dir = os.path.join(TORCH_HUB_CACHE, 'hub')
    ckpt_dir = os.path.join(hub_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Source code: torch.hub names the dir as `{owner}_{repo}_{branch}`,
    # default branch is `main` -> `facebookresearch_dinov2_main`.
    _ensure_symlink(
        os.path.abspath(DINOV2_SRC_DIR),
        os.path.join(hub_dir, 'facebookresearch_dinov2_main'),
    )

    # Weights: torch.hub.load_state_dict_from_url looks up
    # `{TORCH_HOME}/hub/checkpoints/<basename(url)>`.
    for w in weights:
        _ensure_symlink(
            os.path.abspath(os.path.join(DINOV2_WEIGHTS_DIR, w)),
            os.path.join(ckpt_dir, w),
        )

    print(f"[info] dinov2 cache ready at {TORCH_HUB_CACHE}")
    print(f"       source : {os.path.join(hub_dir, 'facebookresearch_dinov2_main')} -> {DINOV2_SRC_DIR}")
    for w in weights:
        print(f"       weight : {os.path.join(ckpt_dir, w)} -> {os.path.join(DINOV2_WEIGHTS_DIR, w)}")
    return TORCH_HUB_CACHE


# ---------------------------------------------------------------------------
# Resume support: which on-disk artefacts indicate each stage is complete?
# ---------------------------------------------------------------------------

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _list_sample_names(images_dir: str) -> List[str]:
    """Sample name = image filename without extension."""
    out = []
    for f in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS:
            out.append(stem)
    return out


def _stage_done_for_sample(method_output: str, name: str, stage: int) -> bool:
    """A sample is considered 'done' for a stage if its output exists OR if
    the stage permanently gave up on it (e.g. 2_decoder.py wrote oom_skip.txt
    after catching CUDA OOM). This prevents `--resume` from looping forever
    on a sample that will OOM every time.
    """
    d = os.path.join(method_output, name)
    if stage == 1:
        return os.path.isfile(os.path.join(d, 'allind.npy'))
    if stage == 2:
        return (os.path.isfile(os.path.join(d, 'sample.glb'))
                or os.path.isfile(os.path.join(d, 'oom_skip.txt')))
    if stage == 3:
        objs = os.path.join(d, 'objs')
        if not os.path.isdir(objs):
            # If stage 2 permanently skipped this sample, there's no glb to
            # split, so stage 3 is implicitly "done" (with zero output) too.
            return os.path.isfile(os.path.join(d, 'oom_skip.txt'))
        return any(os.path.isdir(os.path.join(objs, sub)) for sub in os.listdir(objs))
    if stage == 4:
        if (os.path.isfile(os.path.join(d, 'basic.urdf'))
                and os.path.isfile(os.path.join(d, 'basic_info.json'))):
            return True
        # Same upstream-skip propagation as stage 3.
        return os.path.isfile(os.path.join(d, 'oom_skip.txt'))
    raise ValueError(f"unknown stage: {stage}")


def _stage_completion(
    method_output: str, sample_names: List[str], stages: List[int]
) -> Dict[int, Dict[str, int]]:
    """For each stage, returns {'done': n_done, 'total': n_total}."""
    n_total = len(sample_names)
    status: Dict[int, Dict[str, int]] = {}
    for st in stages:
        n_done = sum(1 for n in sample_names if _stage_done_for_sample(method_output, n, st))
        status[st] = {'done': n_done, 'total': n_total}
    return status


def _print_completion(status: Dict[int, Dict[str, int]]) -> None:
    print("\n  stage  done/total  status")
    for st in sorted(status.keys()):
        d, t = status[st]['done'], status[st]['total']
        flag = 'OK' if d == t else ('partial' if d > 0 else 'todo')
        print(f"    {st}    {d:>4}/{t:<4}  {flag}")


# ---------------------------------------------------------------------------
# Per-sample resume utilities
# ---------------------------------------------------------------------------

_SKIP_SUFFIX_RE = re.compile(r'^(.*)\.skip_resume_[0-9a-f]{8}$')


def _find_image_for_sample(images_dir: str, name: str) -> Optional[str]:
    for ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
        p = os.path.join(images_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


@contextlib.contextmanager
def _filtered_images_dir(images_dir: str, pending_names: Iterable[str]):
    """Create a temp dir under PROJECT_ROOT containing only the symlinks of
    `pending_names` images. Yields the temp dir path; cleans up on exit.
    """
    tmp = tempfile.mkdtemp(prefix='runinf_imgs_', dir=PROJECT_ROOT)
    try:
        for name in pending_names:
            src = _find_image_for_sample(images_dir, name)
            if src is None:
                continue
            os.symlink(os.path.abspath(src), os.path.join(tmp, os.path.basename(src)))
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def _hide_paths(paths: List[str]):
    """Rename each existing path to `<path>.skip_resume_<uuid>`. On exit,
    restore them. The unique suffix lets a startup recovery pass distinguish
    these from any user files."""
    suffix = '.skip_resume_' + uuid.uuid4().hex[:8]
    backups = []
    try:
        for p in paths:
            if os.path.lexists(p):
                bak = p + suffix
                os.rename(p, bak)
                backups.append((p, bak))
        yield
    finally:
        for orig, bak in reversed(backups):
            if not os.path.lexists(bak):
                continue
            if os.path.lexists(orig):
                print(f"[warn] cannot restore {bak} -> {orig}: target already exists; "
                      f"leaving the backup in place.")
                continue
            os.rename(bak, orig)


def _recover_skip_resume_files(root: str) -> int:
    """Restore any files/dirs left as `*.skip_resume_<hex>` from a previous
    crash. Returns the number restored."""
    if not os.path.isdir(root):
        return 0
    restored = 0
    for cur, dirs, files in os.walk(root):
        for entry in list(dirs) + list(files):
            m = _SKIP_SUFFIX_RE.match(entry)
            if not m:
                continue
            src = os.path.join(cur, entry)
            dst = os.path.join(cur, m.group(1))
            if os.path.lexists(dst):
                print(f"[warn] leftover {src!r} cannot be restored; target {dst!r} already exists.")
                continue
            os.rename(src, dst)
            restored += 1
    if restored:
        print(f"[recover] restored {restored} skip_resume entries from a previous crash")
    return restored


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd, cwd=None, env_overrides=None, check=True):
    print(f"\n[run_inference] $ {' '.join(cmd)}")
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit={proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _normalize_vlm_outputs(method_output: str) -> int:
    """After 1_vlm_cot.py finishes, copy ``<stem>/<stem>_overall.txt`` to
    ``<stem>/basic_info.txt`` so downstream stages (which look for
    ``basic_info.txt``) keep working without touching the legacy scripts.

    Idempotent: skips samples that already have a ``basic_info.txt``.
    Returns the number of samples normalized.
    """
    if not os.path.isdir(method_output):
        return 0
    n = 0
    for stem in os.listdir(method_output):
        d = os.path.join(method_output, stem)
        if not os.path.isdir(d):
            continue
        dst = os.path.join(d, 'basic_info.txt')
        if os.path.exists(dst):
            continue
        src = os.path.join(d, f'{stem}_overall.txt')
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            n += 1
    if n:
        print(f"[run_inference] VLM output normalized: {n} samples got basic_info.txt")
    return n


def stage_1_vlm(
    images_dir: str,
    method_output: str,
    ckpt_vlm: str,
    *,
    vlm_script: str = 'vlm_cot',
    base_model: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
):
    """Run the Structured Physical CoT frontend and normalize its outputs.

    ``ckpt_vlm`` is the LoRA adapter directory and ``base_model`` is the
    Qwen3-VL base model directory. Extra arguments are forwarded verbatim.
    """
    extra = list(extra_args or [])

    if vlm_script == 'vlm_cot':
        if not base_model:
            raise ValueError("--vlm_script vlm_cot requires --vlm_base_model (the "
                             "Qwen3-VL base model dir).")
        base_model_ref = (
            os.path.abspath(base_model)
            if os.path.exists(base_model) or os.path.isabs(base_model)
            else base_model
        )
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, '1_vlm_cot.py'),
            '--image_dir',    os.path.abspath(images_dir),
            '--output_dir',   os.path.abspath(method_output),
            '--adapter_path', os.path.abspath(ckpt_vlm),
            '--base_model',   base_model_ref,
            *extra,
        ]
        rc = _run(cmd)
        _normalize_vlm_outputs(method_output)
        return rc

    raise ValueError(f"unknown vlm_script: {vlm_script!r} "
                     f"(expected 'vlm_cot')")


def stage_2_decoder(images_dir, method_output, decoder_path):
    """Decode predicted local geometry into textured GLB assets."""
    torch_home = setup_dinov2_cache()
    env_overrides = {'TORCH_HOME': torch_home} if torch_home else None
    return _run(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, '2_decoder.py'),
            '--images_dir', os.path.abspath(images_dir),
            '--output_dir', os.path.abspath(method_output),
            '--decoder_path', os.path.abspath(decoder_path),
        ],
        env_overrides=env_overrides,
    )


def stage_3_split(method_output):
    """Split each decoded mesh into part-level OBJ files."""
    return _run([
        sys.executable,
        os.path.join(PROJECT_ROOT, '3_split.py'),
        '--basepath', os.path.abspath(method_output),
    ])


@contextlib.contextmanager
def _quarantine_dirs(dirs: List[str]):
    """Physically MOVE each dir out of its parent for the duration of the
    block, then move it back. Different from ``_hide_paths`` which only
    renames in-place: legacy stage-4 does ``os.listdir(method_output)`` so
    a renamed sibling is still visible. We move to a sibling staging dir
    (created next to method_output) to keep things on the same filesystem
    -> ``os.rename`` is atomic and cheap.
    """
    if not dirs:
        yield
        return
    parents = {os.path.dirname(os.path.abspath(d)) for d in dirs}
    if len(parents) != 1:
        raise ValueError(f"_quarantine_dirs: dirs must share a parent, got {parents}")
    parent = parents.pop()
    staging = os.path.join(parent, f'.quarantine_{uuid.uuid4().hex[:8]}')
    os.makedirs(staging, exist_ok=True)
    moved: List[tuple] = []
    try:
        for d in dirs:
            d = os.path.abspath(d)
            if not os.path.exists(d):
                continue
            tgt = os.path.join(staging, os.path.basename(d))
            os.rename(d, tgt)
            moved.append((d, tgt))
        yield
    finally:
        for orig, tgt in reversed(moved):
            if not os.path.lexists(tgt):
                continue
            if os.path.lexists(orig):
                print(f"[warn] cannot restore {tgt} -> {orig}: target already exists; "
                      f"leaving the quarantined copy in place.")
                continue
            os.rename(tgt, orig)
        try:
            os.rmdir(staging)
        except OSError:
            # Non-empty (something failed to restore) -- leave it for inspection.
            pass


# 4_simready_gen.py uses this exact pattern at line ~1044 to split each l_*
# line into 8 fields. We mirror it here to detect malformed lines BEFORE the
# legacy script reaches them.
_L_LINE_RE = re.compile(
    r'l_(\d+):\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*(.*)'
)
_PAREN_RE = re.compile(r'\([^()]*\)')


def _sanitize_basic_info_text(text: str) -> str:
    """Replace ',' inside parenthesized substrings with ';' on ``l_*`` lines.

    Some VLM outputs put commas inside the part name, e.g.
        l_4: Other (e.g., Door Latch or Button), 4, Plastic, ...
    which breaks the comma-delimited regex used by 4_simready_gen.py and
    silently mis-parses material_sim in evaluation_vlm.py. Replacing the
    inner comma with ``;`` keeps the text human-readable and makes both
    parsers see the correct 8 fields.

    No-op for inputs without parenthesized commas (idempotent).
    """
    out: List[str] = []
    for line in text.splitlines():
        if line.startswith('l_'):
            line = _PAREN_RE.sub(lambda m: m.group(0).replace(',', ';'), line)
        out.append(line)
    tail = '\n' if text.endswith('\n') else ''
    return '\n'.join(out) + tail


def _basic_info_well_formed(path: str) -> bool:
    """Mirror every assumption 4_simready_gen.py makes about basic_info.txt
    (lines 1036-1056) so we can pre-filter samples that would crash it:

    * ``lines[0]`` starts with ``Name:``      (line 1036 indexes lines[0])
    * ``lines[1]`` starts with ``Category:``  (line 1037)
    * ``lines[2]`` starts with ``Dimension:`` (line 1038)
    * every ``l_*:`` line matches the 8-field comma pattern (line 1043)
    * at least one ``l_*:`` line exists (otherwise downstream produces an
      empty URDF and crashes elsewhere)

    Truncated VLM outputs commonly drop the first three header lines and
    start straight at ``Parts:`` -- those samples must be quarantined.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return False
    if len(lines) < 3:
        return False
    if not lines[0].startswith('Name:'):
        return False
    if not lines[1].startswith('Category:'):
        return False
    if not lines[2].startswith('Dimension:'):
        return False
    has_part = False
    for line in lines:
        if line.startswith('l_'):
            has_part = True
            if not _L_LINE_RE.match(line):
                return False
    return has_part


def _sanitize_basic_info_files(method_output: str) -> int:
    """Apply ``_sanitize_basic_info_text`` to every basic_info.txt under
    method_output; return the count of files actually modified.
    """
    if not os.path.isdir(method_output):
        return 0
    n = 0
    for stem in os.listdir(method_output):
        f = os.path.join(method_output, stem, 'basic_info.txt')
        if not os.path.isfile(f):
            continue
        with open(f, 'r', encoding='utf-8') as fp:
            txt = fp.read()
        new_txt = _sanitize_basic_info_text(txt)
        if new_txt != txt:
            with open(f, 'w', encoding='utf-8') as fp:
                fp.write(new_txt)
            n += 1
    return n


def stage_4_simready(method_output):
    """Run the SimReady export stage with an explicit output directory.

    Two robustness wrappers around the legacy script (which uses
    ``os.listdir`` + strict comma regex and crashes on the first bad
    sample):

    1. **Sanitize** every ``basic_info.txt`` to normalize commas inside
       parenthesized part names (e.g. ``Other (e.g., Door Latch or Button)``).
       Idempotent; runs every time.

    2. **Quarantine** sample dirs that either (a) have no
       ``basic_info.txt`` (Pass-2 truncation in 1_vlm_cot.py drops the
       ``<overall>`` block) or (b) still contain malformed ``l_*:`` lines
       after sanitize. Quarantined dirs are physically moved to a sibling
       staging folder so legacy ``os.listdir`` does not see them, and
       restored after the script returns.
    """
    fixed = _sanitize_basic_info_files(method_output)
    if fixed:
        print(f"[run_inference] stage 4: sanitized {fixed} basic_info.txt "
              f"(commas inside parens -> semicolons)")

    quarantine: List[str] = []
    reasons: Dict[str, str] = {}
    if os.path.isdir(method_output):
        for d in sorted(os.listdir(method_output)):
            full = os.path.join(method_output, d)
            if not os.path.isdir(full):
                continue
            bi = os.path.join(full, 'basic_info.txt')
            if not os.path.isfile(bi):
                quarantine.append(full)
                reasons[d] = 'missing'
            elif not _basic_info_well_formed(bi):
                quarantine.append(full)
                reasons[d] = 'malformed'

    if quarantine:
        breakdown = {r: [n for n, rs in reasons.items() if rs == r]
                     for r in ('missing', 'malformed')}
        print(f"[run_inference] stage 4: quarantining {len(quarantine)} sample dir(s)")
        for r, names in breakdown.items():
            if names:
                print(f"  - {r:9s} ({len(names)}): {names}")

    with _quarantine_dirs(quarantine):
        return _run([
            sys.executable, os.path.join(PROJECT_ROOT, '4_simready_gen.py'),
            '--basepath', os.path.abspath(method_output),
        ])


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_inference(
    images_dir: str,
    method_name: str,
    output_root: str = './outputs',
    ckpt_vlm: str = './pretrain/vlm',
    ckpt_decoder: str = './pretrain/decoder',
    stages: Optional[List[int]] = None,
    resume: bool = False,
    vlm_script: str = 'vlm_cot',
    vlm_base_model: Optional[str] = None,
    vlm_extra_args: Optional[List[str]] = None,
) -> str:
    if stages is None:
        stages = [1, 2, 3, 4]

    ckpt_vlm = _resolve_repo_path(ckpt_vlm)
    ckpt_decoder = _resolve_repo_path(ckpt_decoder)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if any(c in method_name for c in ' /\\'):
        raise ValueError(f"method_name must not contain spaces or path separators: {method_name!r}")

    method_output = os.path.join(output_root, method_name)
    os.makedirs(method_output, exist_ok=True)

    print(f"\n========== run_inference: method={method_name!r} ==========")
    print(f"  images_dir    : {images_dir}")
    print(f"  output dir    : {method_output}")
    print(f"  vlm script    : {vlm_script}")
    print(f"  vlm ckpt      : {ckpt_vlm}")
    if vlm_script == 'vlm_cot':
        print(f"  vlm base      : {vlm_base_model}")
    if vlm_extra_args:
        print(f"  vlm extra args: {vlm_extra_args}")

    sample_names = _list_sample_names(images_dir)
    if not sample_names:
        raise RuntimeError(f"no images found in {images_dir}")

    if resume:
        # Recover any *.skip_resume_<uuid> leftovers from a previous crash so
        # this run starts from a clean state.
        _recover_skip_resume_files(method_output)
        status = _stage_completion(method_output, sample_names, stages)
        print(f"\n  [resume] scanned {len(sample_names)} samples under {method_output}:")
        _print_completion(status)

    print(f"  stages to run : {stages}")

    if 1 in stages:
        _run_stage_with_resume(
            stage=1, label=f'VLM ({vlm_script})',
            sample_names=sample_names, method_output=method_output,
            resume=resume,
            run_fn=lambda fdir: stage_1_vlm(
                fdir, method_output, ckpt_vlm,
                vlm_script=vlm_script,
                base_model=vlm_base_model,
                extra_args=vlm_extra_args,
            ),
            mode='filter_images', images_dir=images_dir,
        )
    if 2 in stages:
        _run_stage_with_resume(
            stage=2, label='Decoder',
            sample_names=sample_names, method_output=method_output,
            resume=resume,
            run_fn=lambda fdir: stage_2_decoder(fdir, method_output, ckpt_decoder),
            mode='filter_images', images_dir=images_dir,
        )
    if 3 in stages:
        _run_stage_with_resume(
            stage=3, label='Split',
            sample_names=sample_names, method_output=method_output,
            resume=resume,
            run_fn=lambda _ignored: stage_3_split(method_output),
            mode='hide_in_output',
            hide_path_fn=lambda n: os.path.join(method_output, n, 'sample.glb'),
        )
    if 4 in stages:
        _run_stage_with_resume(
            stage=4, label='SimReady (URDF)',
            sample_names=sample_names, method_output=method_output,
            resume=resume,
            run_fn=lambda _ignored: stage_4_simready(method_output),
            mode='hide_in_output',
            hide_path_fn=lambda n: os.path.join(method_output, n, 'objs'),
        )

    print(f"\n[done] outputs in {method_output}")
    return method_output


def _run_stage_with_resume(
    *,
    stage: int,
    label: str,
    sample_names: List[str],
    method_output: str,
    resume: bool,
    run_fn,
    mode: str,
    images_dir: Optional[str] = None,
    hide_path_fn=None,
) -> None:
    """Dispatcher that wraps a stage execution with per-sample resume logic.

    `mode='filter_images'`  : build a temp images dir of pending samples and
                              pass its path to `run_fn(fdir)`.
    `mode='hide_in_output'` : temporarily rename each completed sample's
                              gating file/dir under method_output (returned by
                              `hide_path_fn(name)`) so the legacy script's
                              built-in "input present?" check skips it.
    """
    print(f"\n----- Stage {stage}: {label} -----")

    if resume:
        pending = [n for n in sample_names if not _stage_done_for_sample(method_output, n, stage)]
        done = [n for n in sample_names if _stage_done_for_sample(method_output, n, stage)]
        print(f"  pending {len(pending)} / total {len(sample_names)} samples "
              f"({len(done)} already done, will be skipped)")
        if not pending:
            print(f"  [skip] stage {stage}: nothing to do.")
            return
    else:
        pending = sample_names
        done = []

    if mode == 'filter_images':
        if images_dir is None:
            raise ValueError("filter_images mode requires images_dir")
        if resume and done:
            with _filtered_images_dir(images_dir, pending) as fdir:
                run_fn(fdir)
        else:
            run_fn(images_dir)

    elif mode == 'hide_in_output':
        if hide_path_fn is None:
            raise ValueError("hide_in_output mode requires hide_path_fn")
        if resume and done:
            paths = [hide_path_fn(n) for n in done]
            with _hide_paths(paths):
                run_fn(None)
        else:
            run_fn(None)

    else:
        raise ValueError(f"unknown mode: {mode}")


def _parse_stages(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='PhysX-CoT four-stage inference runner')
    parser.add_argument('--images_dir', required=True,
                        help='Folder with input images (Stage-A output).')
    parser.add_argument('--method_name', required=True,
                        help='Identifier of this run; outputs go to {output_root}/{method_name}.')
    parser.add_argument('--output_root', default='./outputs',
                        help='Root for per-method output dirs.')
    parser.add_argument('--ckpt_vlm', default='./pretrain/vlm',
                        help='LoRA adapter directory passed as --adapter_path.')
    parser.add_argument('--ckpt_decoder', default='./pretrain/decoder',
                        help='Path to the TRELLIS decoder checkpoint.')
    parser.add_argument('--vlm_script', default='vlm_cot', choices=['vlm_cot'],
                        help='Structured Physical CoT frontend (1_vlm_cot.py).')
    parser.add_argument('--vlm_base_model', default=None,
                        help='Required when --vlm_script vlm_cot: path to the '
                             'Qwen3-VL base model dir (passed as --base_model).')
    parser.add_argument('--vlm_extra_args', default='',
                        help='Extra args forwarded verbatim to the stage-1 '
                             'script (e.g. \'--sam_feature_dir /path '
                             '--no_auto_extract_sam\'). Use a single quoted string.')
    parser.add_argument('--stages', default='1,2,3,4', type=_parse_stages,
                        help='Comma-separated list of stages to run, e.g. 1,2,3,4.')
    parser.add_argument('--resume', action='store_true',
                        help='Per-sample resume. Within each stage, samples '
                             'whose outputs already exist are skipped; only '
                             'pending samples are reprocessed. Also recovers '
                             'leftover *.skip_resume_<uuid> files from a '
                             'previous crash automatically.')
    parser.add_argument('--status', action='store_true',
                        help='Print per-stage completion status and exit '
                             '(does not run inference).')
    args = parser.parse_args()

    if args.status:
        method_output = os.path.join(args.output_root, args.method_name)
        sample_names = _list_sample_names(args.images_dir)
        status = _stage_completion(method_output, sample_names, args.stages)
        print(f"\nmethod={args.method_name!r}  samples={len(sample_names)}")
        print(f"output_dir={method_output}")
        _print_completion(status)
        return

    import shlex
    extra = shlex.split(args.vlm_extra_args) if args.vlm_extra_args else []

    if args.vlm_script == 'vlm_cot' and not args.vlm_base_model:
        parser.error("--vlm_script vlm_cot requires --vlm_base_model "
                     "(the Qwen3-VL base model dir).")

    run_inference(
        images_dir=args.images_dir,
        method_name=args.method_name,
        output_root=args.output_root,
        ckpt_vlm=args.ckpt_vlm,
        ckpt_decoder=args.ckpt_decoder,
        stages=args.stages,
        resume=args.resume,
        vlm_script=args.vlm_script,
        vlm_base_model=args.vlm_base_model,
        vlm_extra_args=extra,
    )


if __name__ == '__main__':
    main()
