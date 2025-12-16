from collections.abc import Iterator

import numpy as np


def random_array_generator(shape: tuple[int, int], dtype: np.dtype = np.float32, seed: int = 0) -> Iterator[np.ndarray]:  # type: ignore
    """Yield an infinite, deterministic stream of 2D arrays of the given shape and dtype.

    Example:
        # With a fixed seed (default 0), the sequence is reproducible
        gen = array_2d_generator((480, 640), dtype=np.float32, seed=42)
        a = next(gen)  # -> np.ndarray of shape (480, 640), dtype float32
    """
    rng = np.random.default_rng(seed)
    np_dtype = np.dtype(dtype)

    while True:
        if np_dtype.kind == "f":
            array = rng.random(shape, dtype=np_dtype)
        elif np_dtype.kind in ("i", "u"):
            # Use a safe, small range for dummy integer data
            high = min(1000, np.iinfo(np_dtype).max)
            array = rng.integers(low=0, high=high, size=shape, dtype=np_dtype)
        else:
            msg = f"Unsupported dtype kind '{np_dtype.kind}'. Use float or int types."
            raise ValueError(msg)
        yield array


class Deterministic3DPointCloudDataset:
    def __init__(self, size: int, seed: int = 0):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, sample: int) -> dict[str, np.ndarray]:
        return {
            "vertices_3d": np.random.rand(100, 3).astype(np.float32),
            "intensities": np.random.rand(100).astype(np.float32),
        }
