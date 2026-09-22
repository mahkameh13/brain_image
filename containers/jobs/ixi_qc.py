#!/usr/bin/env python3
"""Export orthogonal mask overlays from validated cases, or render their PNG."""
import argparse
import json
from pathlib import Path
import numpy as np

WORK = Path('/work/dldevel/momeni')
DEST = WORK / 'datasets/IXI_skullstripped'


def export(path):
    import nibabel as nib
    provenance = sorted(DEST.rglob('*_synthstrip.json'))
    if not provenance:
        raise RuntimeError('No validated cases available')
    rows = [(p, json.loads(p.read_text())) for p in provenance]
    by_volume = sorted(rows, key=lambda r: r[1]['qc']['mask_volume_ml'])
    selected = [rows[0], rows[len(rows)//2], rows[-1], by_volume[0], by_volume[-1]]
    selected = list({str(p): (p, r) for p, r in selected}.values())
    arrays = {}
    labels = []
    for i, (p, rec) in enumerate(selected):
        source = WORK / 'datasets/IXI' / rec['input']['relative']
        mask = p.with_name(p.name.replace('_synthstrip.json', '_mask.nii.gz'))
        x = nib.as_closest_canonical(nib.load(str(source))).get_fdata(dtype=np.float32)
        m = nib.as_closest_canonical(nib.load(str(mask))).get_fdata(dtype=np.float32)
        center = np.array(np.nonzero(m)).mean(axis=1).astype(int)
        for axis in range(3):
            arrays[f'x{i}_{axis}'] = np.take(x, center[axis], axis=axis).T
            arrays[f'm{i}_{axis}'] = np.take(m, center[axis], axis=axis).T
        labels.append(rec['input']['relative'])
    arrays['labels'] = np.array(labels)
    np.savez_compressed(path, **arrays)
    print(path)


def render(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.load(path)
    labels = data['labels']
    fig, axes = plt.subplots(len(labels), 3, figsize=(10, 3.3*len(labels)), squeeze=False)
    for i, label in enumerate(labels):
        for axis in range(3):
            ax = axes[i, axis]
            x, m = data[f'x{i}_{axis}'], data[f'm{i}_{axis}']
            ax.imshow(x, cmap='gray', origin='lower', vmin=np.percentile(x, 1), vmax=np.percentile(x, 99))
            ax.contour(m, levels=[0.5], colors=['#00e5ff'], linewidths=0.7)
            ax.axis('off')
            ax.set_title(('Sagittal', 'Coronal', 'Axial')[axis], fontsize=9)
        axes[i, 1].set_title(label + '\nCoronal', fontsize=8)
    fig.suptitle('IXI SynthStrip — cyan brain-mask boundary\nRepresentative cases and mask-volume extremes', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    target = path.with_suffix('.png')
    fig.savefig(target, dpi=160, facecolor='white')
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'render'))
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    (export if args.mode == 'export' else render)(args.path)
