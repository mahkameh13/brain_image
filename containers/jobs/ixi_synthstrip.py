#!/usr/bin/env python3
"""Freeze T1 inputs, run SynthStrip, and validate results in the source grid."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

WORK = Path('/work/dldevel/momeni')
SOURCE = WORK / 'datasets/IXI'
DEST = WORK / 'datasets/IXI_skullstripped'
MANIFEST = WORK / 'containers/jobs/ixi_t1_manifest.json'
SIF = WORK / 'containers/synthstrip_1.8-gpu.sif'


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def signature(path):
    st = path.stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns)


def freeze():
    if MANIFEST.exists():
        raise RuntimeError('Manifest already exists; reuse it to keep the input cohort fixed.')
    inputs = []
    for path in sorted(SOURCE.rglob('*')):
        rel = path.relative_to(SOURCE)
        if any(p.startswith('.') for p in rel.parts) or path.is_symlink() or not path.is_file():
            continue
        name = re.sub(r'\.nii(?:\.gz)?$', '', path.name, flags=re.I)
        if name == path.name or not re.search(r'(^|[-_])t1w?($|[-_])', name, re.I):
            continue
        inputs.append(dict(relative=str(rel), **signature(path)))
    if not inputs:
        raise RuntimeError('No real T1 NIfTIs found')
    write_json(MANIFEST, dict(source=str(SOURCE), destination=str(DEST), inputs=inputs))
    print(f'Manifest: {MANIFEST}; T1 inputs: {len(inputs)}', flush=True)


def paths(relative):
    rel = Path(relative)
    stem = re.sub(r'\.nii(?:\.gz)?$', '', rel.name, flags=re.I)
    folder = DEST / rel.parent
    return (folder / (stem + '_brain.nii.gz'),
            folder / (stem + '_mask.nii.gz'),
            folder / (stem + '_synthstrip.json'))


def validate(src, brain, mask):
    import nibabel as nib
    import numpy as np
    ims = [nib.load(str(p)) for p in (src, brain, mask)]
    if len(ims[0].shape) != 3 or any(im.shape != ims[0].shape for im in ims):
        raise ValueError('Expected matching 3D shapes')
    if any(not np.allclose(im.affine, ims[0].affine, atol=1e-4, rtol=0) for im in ims):
        raise ValueError('Output affine differs from input')
    x, y, m = [np.asarray(im.dataobj, dtype=np.float32) for im in ims]
    if any(not np.isfinite(a).all() for a in (x, y, m)):
        raise ValueError('Nonfinite voxel values')
    if not np.all((m == 0) | (m == 1)) or not 0 < m.sum() < m.size:
        raise ValueError('Mask must be binary, nonempty, and not cover the entire image')
    inside = m.astype(bool)
    if not np.allclose(y[inside], x[inside], rtol=1e-5, atol=1e-4):
        raise ValueError('Brain intensities changed inside mask')
    if np.any(y[~inside] != 0):
        raise ValueError('Brain output is not zero outside mask')
    return dict(shape=list(x.shape), mask_voxels=int(m.sum()),
                mask_fraction=float(m.mean()),
                mask_volume_ml=float(m.sum() * abs(np.linalg.det(ims[0].affine[:3, :3])) / 1000))


def run(args):
    records = json.loads(MANIFEST.read_text())['inputs']
    if args.limit:
        records = records[:args.limit]
    hasher = hashlib.sha256()
    with SIF.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    image_sha = hasher.hexdigest()
    if not args.verify:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; stopping before processing')
        torch.ones(1, device='cuda').sum().item()
        print(f'GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}', flush=True)
    results = []
    consecutive_failures = 0
    for index, rec in enumerate(records, 1):
        relative = rec['relative']
        src = SOURCE / relative
        brain, mask, sidecar = paths(relative)
        started = time.monotonic()
        try:
            if signature(src) != {k: rec[k] for k in ('size', 'mtime_ns')}:
                raise ValueError('Input changed since manifest creation')
            previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            reusable = (previous.get('image_sha256') == image_sha and
                        previous.get('input') == rec and brain.exists() and mask.exists())
            if args.verify or reusable:
                if not reusable:
                    raise ValueError('Missing output or matching provenance')
                qc = validate(src, brain, mask)
                status = 'verified' if args.verify else 'reused'
            else:
                brain.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.synthstrip-', dir=brain.parent) as tmp:
                    tb, tm = Path(tmp) / 'brain.nii.gz', Path(tmp) / 'mask.nii.gz'
                    cmd = ['mri_synthstrip', '-i', str(src), '-o', str(tb), '-m', str(tm),
                           '-g', '-t', str(args.threads), '-f', '0']
                    with brain.with_name(brain.name.replace('_brain.nii.gz', '_synthstrip.log')).open('w') as log:
                        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                    qc = validate(src, tb, tm)
                    tb.replace(brain)
                    tm.replace(mask)
                write_json(sidecar, dict(input=rec, image_sha256=image_sha,
                    image='docker://freesurfer/synthstrip:1.8-gpu', gpu=True,
                    arguments=['-g', '-t', str(args.threads), '-f', '0'], qc=qc))
                status = 'created'
            result = dict(relative=relative, status=status, seconds=round(time.monotonic()-started, 2), **qc)
            consecutive_failures = 0
        except Exception as exc:
            result = dict(relative=relative, status='failed', error=str(exc))
            consecutive_failures += 1
        results.append(result)
        print(f'[{index}/{len(records)}] {json.dumps(result)}', flush=True)
        if consecutive_failures >= 3 and not args.verify:
            print('Stopping after three consecutive failures', flush=True)
            break
    failed = sum(r['status'] == 'failed' for r in results)
    summary = dict(expected=len(records), checked=len(results), succeeded=len(results)-failed,
                   failed=failed, complete=len(results)==len(records) and failed==0,
                   image_sha256=image_sha, results=results)
    tag = 'verification' if args.verify else 'run'
    suffix = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    report = DEST / f'{tag}-{suffix}.json'
    write_json(report, summary)
    print(f'Report: {report}; succeeded={summary["succeeded"]}/{len(records)}', flush=True)
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.prepare:
        freeze()
    else:
        raise SystemExit(run(args))
