"""Shared, versioned frame measurements used by board analysis and page grouping.

The cache stores packed ink/foreground masks and per-frame ink sharpness.
It is written in bounded chunks and atomically replaced so interrupted runs
never expose a partially written chunk as valid data.
"""
import os
from pathlib import Path

import numpy as np

from .storage import key

FEATURE_CACHE_VERSION = "2026-11-09.board-features-1"
FEATURE_CHUNK_SIZE = 128


def feature_cache_dir(root, frames, config):
    signature = [frames, config.board, config.analysis_width, config.roi]
    return Path(root) / "cache" / "board-features" / key(
        signature, FEATURE_CACHE_VERSION
    )


def write_feature_chunk(folder, start, inks, masks, sharpness, shape):
    """Atomically store one chunk of packed per-frame measurements."""
    if not inks:
        return

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"features-{start:06d}.npz"
    temporary = path.with_name(path.stem + ".tmp.npz")

    try:
        np.savez_compressed(
            temporary,
            version=np.asarray(FEATURE_CACHE_VERSION),
            ink=np.stack(inks).astype(np.uint8, copy=False),
            mask=np.stack(masks).astype(np.uint8, copy=False),
            sharp=np.asarray(sharpness, dtype=np.float32),
            shape=np.asarray(shape, dtype=np.int32),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_feature_observations(root, frames, config):
    """Return the complete shared cache, or None when it is missing/invalid."""
    if not frames:
        return None

    folder = feature_cache_dir(root, frames, config)
    ink_chunks, mask_chunks, sharp_chunks = [], [], []
    expected_shape = None
    pixel_count = None

    for start in range(0, len(frames), FEATURE_CHUNK_SIZE):
        path = folder / f"features-{start:06d}.npz"
        if not path.is_file():
            return None

        expected_rows = min(FEATURE_CHUNK_SIZE, len(frames) - start)
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["version"].item()) != FEATURE_CACHE_VERSION:
                    return None

                ink = np.asarray(data["ink"], dtype=np.uint8)
                mask = np.asarray(data["mask"], dtype=np.uint8)
                sharp = np.asarray(data["sharp"], dtype=np.float32)
                shape = tuple(int(value) for value in data["shape"].tolist())
        except (OSError, ValueError, KeyError, EOFError):
            return None

        if len(shape) != 2 or min(shape) <= 0:
            return None
        count = int(shape[0] * shape[1])
        packed_width = (count + 7) // 8

        if (
            ink.ndim != 2
            or mask.shape != ink.shape
            or ink.shape != (expected_rows, packed_width)
            or sharp.shape != (expected_rows,)
        ):
            return None
        if expected_shape is not None and shape != expected_shape:
            return None

        expected_shape = shape
        pixel_count = count
        ink_chunks.append(ink.copy())
        mask_chunks.append(mask.copy())
        sharp_chunks.append(sharp.copy())

    return {
        "ink": np.concatenate(ink_chunks, axis=0),
        "mask": np.concatenate(mask_chunks, axis=0),
        "sharp": np.concatenate(sharp_chunks, axis=0),
        "shape": expected_shape,
    }
