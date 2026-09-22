#!/usr/bin/env python3
import gzip
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


DTYPES = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


def read_nifti(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        header = handle.read(348)
        endian = "<" if struct.unpack("<I", header[:4])[0] == 348 else ">"
        dims = struct.unpack(endian + "8h", header[40:56])
        shape = tuple(dims[1 : dims[0] + 1])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        offset = int(struct.unpack(endian + "f", header[108:112])[0])
        slope = struct.unpack(endian + "f", header[112:116])[0] or 1.0
        intercept = struct.unpack(endian + "f", header[116:120])[0]
        handle.seek(offset)
        dtype = np.dtype(endian + DTYPES[datatype])
        data = np.frombuffer(handle.read(), dtype=dtype, count=int(np.prod(shape)))
    return data.reshape(shape, order="F").astype(np.float32) * slope + intercept


def axial_preview(volume: np.ndarray) -> np.ndarray:
    while volume.ndim > 3:
        volume = volume[..., 0]
    occupied = np.argwhere(volume != 0)
    z = int(np.median(occupied[:, 2])) if occupied.size else volume.shape[2] // 2
    image = np.rot90(volume[:, :, z])
    foreground = image[image > 0]
    if foreground.size:
        low, high = np.percentile(foreground, (1, 99.5))
    else:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        high = low + 1
    return (np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def write_pgm(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode())
        handle.write(image.tobytes())


def main() -> None:
    output = Path(sys.argv[1])
    inputs = [Path(value) for value in sys.argv[2:]]
    with tempfile.TemporaryDirectory(prefix="skullstrip-preview-") as tmp:
        panels = []
        for source in inputs:
            subject = source.name.split("-")[0]
            pgm = Path(tmp) / f"{subject}.pgm"
            png = Path(tmp) / f"{subject}.png"
            write_pgm(pgm, axial_preview(read_nifti(source)))
            subprocess.run(
                [
                    "convert", str(pgm), "-resize", "300x300", "-background", "black",
                    "-gravity", "center", "-extent", "320x320", "-gravity", "south",
                    "-fill", "white", "-undercolor", "#000000a0", "-pointsize", "22",
                    "-annotate", "+0+8", subject, str(png),
                ],
                check=True,
            )
            panels.append(str(png))
        subprocess.run(
            ["montage", *panels, "-tile", "3x2", "-geometry", "+8+8", "-background", "#202020", str(output)],
            check=True,
        )


if __name__ == "__main__":
    main()
