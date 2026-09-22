#!/usr/bin/env python3
"""Recursively convert NIfTI volumes (.nii/.nii.gz) to PNG triptychs.

Each PNG shows the middle sagittal, coronal, and axial slices. The directory
layout below INPUT is reproduced below OUTPUT.
"""

from __future__ import annotations

import argparse
import binascii
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import math
import os
from pathlib import Path
import struct
import sys
import zlib

import numpy as np


NIFTI_DTYPES = {
    2: "u1",     # unsigned char
    4: "i2",     # signed short
    8: "i4",     # signed int
    16: "f4",    # float
    64: "f8",    # double
    256: "i1",   # signed char
    512: "u2",   # unsigned short
    768: "u4",   # unsigned int
    1024: "i8",  # signed long long
    1280: "u8",  # unsigned long long
}


def read_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a scalar NIfTI-1 file without requiring nibabel."""
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as handle:
        header = handle.read(348)
        if len(header) != 348:
            raise ValueError("file is too small to contain a NIfTI-1 header")

        if struct.unpack("<I", header[:4])[0] == 348:
            endian = "<"
        elif struct.unpack(">I", header[:4])[0] == 348:
            endian = ">"
        else:
            raise ValueError("not a supported NIfTI-1 file")

        dims = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dims[0])
        if not 3 <= ndim <= 7:
            raise ValueError(f"expected at least 3 dimensions, found {ndim}")
        shape = tuple(int(value) for value in dims[1 : ndim + 1])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        if datatype not in NIFTI_DTYPES:
            raise ValueError(f"unsupported NIfTI datatype code {datatype}")

        pixdim = struct.unpack(endian + "8f", header[76:108])
        spacing = tuple(abs(float(value)) or 1.0 for value in pixdim[1:4])
        offset = max(348, int(struct.unpack(endian + "f", header[108:112])[0]))
        slope = float(struct.unpack(endian + "f", header[112:116])[0])
        intercept = float(struct.unpack(endian + "f", header[116:120])[0])
        if not math.isfinite(slope) or slope == 0:
            slope = 1.0
        if not math.isfinite(intercept):
            intercept = 0.0

        handle.seek(offset)
        count = int(np.prod(shape))
        dtype = np.dtype(endian + NIFTI_DTYPES[datatype])
        raw = handle.read(count * dtype.itemsize)
        data = np.frombuffer(raw, dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"truncated voxel data: expected {count}, found {data.size}")

    volume = data.reshape(shape, order="F")
    while volume.ndim > 3:
        volume = volume[..., 0]
    volume = volume.astype(np.float32)
    volume *= slope
    volume += intercept
    return volume, spacing


def middle_of_foreground(volume: np.ndarray, axis: int) -> int:
    other_axes = tuple(value for value in range(3) if value != axis)
    occupied = np.any(np.isfinite(volume) & (volume != 0), axis=other_axes)
    indices = np.flatnonzero(occupied)
    if indices.size == 0:
        return volume.shape[axis] // 2
    return int((indices[0] + indices[-1]) // 2)


def intensity_limits(volume: np.ndarray) -> tuple[float, float]:
    flat = volume.ravel()
    step = max(1, flat.size // 1_000_000)
    sample = flat[::step]
    sample = sample[np.isfinite(sample) & (sample != 0)]
    if sample.size == 0:
        return 0.0, 1.0

    unique = np.unique(sample)
    if unique.size <= 8:
        low, high = float(sample.min()), float(sample.max())
        low = min(0.0, low)
    else:
        low, high = (float(value) for value in np.percentile(sample, (0.5, 99.5)))
        low = min(0.0, low)
    if not math.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def to_uint8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.nan_to_num((image.astype(np.float32) - low) / (high - low))
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
    columns = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
    return image[np.ix_(rows, columns)]


def fit_square(image: np.ndarray, physical_ratio: float, size: int) -> np.ndarray:
    """Resize into a square while preserving the physical width/height ratio."""
    physical_ratio = max(0.05, min(20.0, physical_ratio))
    if physical_ratio >= 1.0:
        width = size
        height = max(1, round(size / physical_ratio))
    else:
        height = size
        width = max(1, round(size * physical_ratio))
    resized = resize_nearest(image, height, width)
    panel = np.zeros((size, size), dtype=np.uint8)
    top = (size - height) // 2
    left = (size - width) // 2
    panel[top : top + height, left : left + width] = resized
    return panel


def make_triptych(volume: np.ndarray, spacing: tuple[float, float, float], size: int) -> np.ndarray:
    low, high = intensity_limits(volume)
    x = middle_of_foreground(volume, 0)
    y = middle_of_foreground(volume, 1)
    z = middle_of_foreground(volume, 2)

    # rot90 gives conventional portrait-oriented display panels.
    sagittal = np.rot90(volume[x, :, :])
    coronal = np.rot90(volume[:, y, :])
    axial = np.rot90(volume[:, :, z])

    views = (
        (sagittal, (volume.shape[1] * spacing[1]) / (volume.shape[2] * spacing[2])),
        (coronal, (volume.shape[0] * spacing[0]) / (volume.shape[2] * spacing[2])),
        (axial, (volume.shape[0] * spacing[0]) / (volume.shape[1] * spacing[1])),
    )
    panels = [fit_square(to_uint8(view, low, high), ratio, size) for view, ratio in views]
    separator = np.full((size, 4), 48, dtype=np.uint8)
    return np.concatenate((panels[0], separator, panels[1], separator, panels[2]), axis=1)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def write_grayscale_png(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def output_name(source: Path) -> str:
    name = source.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7] + ".png"
    return source.stem + ".png"


def convert_one(job: tuple[str, str, int, bool]) -> tuple[str, str]:
    source_value, destination_value, size, overwrite = job
    source = Path(source_value)
    destination = Path(destination_value)
    if destination.exists() and not overwrite:
        return "skipped", source_value
    volume, spacing = read_nifti(source)
    write_grayscale_png(destination, make_triptych(volume, spacing, size))
    return "converted", source_value


def find_inputs(root: Path, kind: str) -> list[Path]:
    if root.is_file():
        candidates = [root]
    else:
        candidates = [
            path for path in root.rglob("*")
            if path.is_file() and (path.name.lower().endswith(".nii.gz") or path.suffix.lower() == ".nii")
        ]
    if kind == "brain":
        candidates = [path for path in candidates if "_brain" in path.name.lower()]
    elif kind == "mask":
        candidates = [path for path in candidates if "_mask" in path.name.lower()]
    return sorted(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A NIfTI file or directory to scan recursively")
    parser.add_argument("output", type=Path, help="Directory in which PNG files will be written")
    parser.add_argument("--kind", choices=("all", "brain", "mask"), default="all", help="Files to convert (default: all)")
    parser.add_argument("--size", type=int, default=256, help="Pixel size of each view (default: 256)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Parallel worker count (default: up to 4)")
    parser.add_argument("--overwrite", action="store_true", help="Replace PNG files that already exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if not source_root.exists():
        print(f"error: input does not exist: {source_root}", file=sys.stderr)
        return 2
    if args.size < 32:
        print("error: --size must be at least 32", file=sys.stderr)
        return 2

    sources = find_inputs(source_root, args.kind)
    if not sources:
        print("No matching NIfTI files found.")
        return 0
    relative_root = source_root if source_root.is_dir() else source_root.parent
    jobs = []
    for source in sources:
        relative = source.relative_to(relative_root)
        destination = output_root / relative.parent / output_name(source)
        jobs.append((str(source), str(destination), args.size, args.overwrite))

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    print(f"Found {len(jobs)} file(s); writing PNGs below {output_root}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(convert_one, job): job[0] for job in jobs}
        for completed, future in enumerate(as_completed(future_map), start=1):
            source = future_map[future]
            try:
                status, _ = future.result()
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                print(f"FAILED {source}: {error}", file=sys.stderr, flush=True)
            if completed % 25 == 0 or completed == len(jobs):
                print(f"Progress: {completed}/{len(jobs)}", flush=True)

    print(
        f"Done: {counts['converted']} converted, {counts['skipped']} skipped, {counts['failed']} failed.",
        flush=True,
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
