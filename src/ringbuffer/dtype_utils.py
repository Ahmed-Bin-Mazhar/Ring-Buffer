from __future__ import annotations

import numpy as np

_SUPPORTED_BYTE_SIZES = {1: np.uint8, 2: np.int16, 4: np.float32, 8: np.complex64}


def resolve_dtype(dtype=None, bytes_per_sample=None):
    """Resolve a supported NumPy dtype from either dtype or bytes_per_sample."""
    if dtype is None and bytes_per_sample is None:
        return np.dtype(np.int16)

    if bytes_per_sample is not None:
        if not isinstance(bytes_per_sample, (int, np.integer)):
            raise TypeError("bytes_per_sample must be an integer")
        if bytes_per_sample <= 0:
            raise ValueError("bytes_per_sample must be greater than 0")
        if bytes_per_sample not in _SUPPORTED_BYTE_SIZES:
            raise ValueError(
                f"Unsupported bytes_per_sample={bytes_per_sample}. Supported values are 1, 2, 4, and 8."
            )

        if dtype is None:
            return np.dtype(_SUPPORTED_BYTE_SIZES[bytes_per_sample])

        if isinstance(dtype, (int, np.integer)):
            if int(dtype) != int(bytes_per_sample):
                raise ValueError("dtype and bytes_per_sample conflict")
            return np.dtype(_SUPPORTED_BYTE_SIZES[int(dtype)])

        resolved = np.dtype(dtype)
        if resolved.itemsize != int(bytes_per_sample):
            raise ValueError(
                f"dtype itemsize ({resolved.itemsize}) does not match bytes_per_sample ({bytes_per_sample})"
            )
        return resolved

    if isinstance(dtype, (int, np.integer)):
        value = int(dtype)
        if value not in _SUPPORTED_BYTE_SIZES:
            raise ValueError(
                f"Unsupported bytes_per_sample value {value}. Supported values are 1, 2, 4, and 8."
            )
        return np.dtype(_SUPPORTED_BYTE_SIZES[value])

    return np.dtype(dtype)
