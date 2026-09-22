#!/usr/bin/env python3
"""Re-read every expected output and verify input/provenance/content consistency."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from ixi_synthstrip import SOURCE, DEST, MANIFEST, SIF, paths, signature, validate, write_json


def check(item):
    record, image_sha = item
    try:
        source = SOURCE / record['relative']
        brain, mask, sidecar = paths(record['relative'])
        if signature(source) != {k: record[k] for k in ('size', 'mtime_ns')}:
            raise ValueError('Input changed since manifest creation')
        provenance = json.loads(sidecar.read_text())
        if provenance.get('input') != record or provenance.get('image_sha256') != image_sha:
            raise ValueError('Provenance mismatch')
        return dict(relative=record['relative'], status='verified', **validate(source, brain, mask))
    except Exception as exc:
        return dict(relative=record['relative'], status='failed', error=str(exc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=1, help='1-based first manifest entry')
    parser.add_argument('--limit', type=int, help='verify the first N manifest entries')
    parser.add_argument('--tag', default='verification', help='report filename prefix')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    records = manifest['inputs']
    actual = set()
    for p in SOURCE.rglob('*'):
        relative = p.relative_to(SOURCE)
        if not p.is_file() or p.is_symlink() or any(s.startswith('.') for s in relative.parts):
            continue
        name = re.sub(r'\.nii(?:\.gz)?$', '', p.name, flags=re.I)
        if name != p.name and re.search(r'(^|[-_])t1w?($|[-_])', name, re.I):
            actual.add(str(relative))
    expected = {r['relative'] for r in records}
    if actual != expected or len(expected) != len(records):
        raise RuntimeError('Manifest does not exactly cover current unique T1 inputs')
    if not 1 <= args.start <= len(records):
        raise ValueError('--start must be between 1 and manifest size')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    records = records[args.start - 1:None if args.limit is None else args.start - 1 + args.limit]
    hasher = hashlib.sha256()
    with SIF.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            hasher.update(chunk)
    sha = hasher.hexdigest()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, ((r, sha) for r in records)))
    failed = sum(r['status'] == 'failed' for r in results)
    report = dict(expected=len(records), checked=len(results), succeeded=len(results)-failed,
                  failed=failed, complete=failed==0, manifest_coverage_exact=True,
                  range_start=args.start, range_end=args.start + len(records) - 1,
                  image_sha256=sha, results=results)
    destination = DEST / (args.tag + '-' + os.environ.get('SLURM_JOB_ID', str(os.getpid())) + '.json')
    write_json(destination, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    print('Report:', destination, flush=True)
    for result in results:
        if result['status'] == 'failed':
            print(json.dumps(result), flush=True)
    raise SystemExit(1 if failed else 0)
