# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""URL adapter for virtualizing NIfTI slice images.

Extracts a single 2D axial slice from an uncompressed ``.nii`` file, normalizes
it to uint8, and returns PNG bytes — all without loading the full 3D volume.

URL format::

    nifti-slice:///path/to/volume.nii?z=77&dtype=int16&offset=2880&w=240&h=240&vmax=386

Query parameters:

- ``z``: Slice index along the third axis.
- ``dtype``: On-disk voxel data type (``int16``, ``uint8``, ``float32``).
- ``offset``: Byte offset where voxel data begins in the ``.nii`` file.
- ``w``, ``h``: First two spatial dimensions (width, height).
- ``vmax``: Upper intensity value for windowing (maps to 255).

The adapter seeks directly to the slice bytes (Fortran / column-major order),
reads only what is needed, and encodes the result as a grayscale PNG.
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from PIL import Image
from tlcurl.url import Url
from tlcurl.url_adapter import UrlAdapter

_DTYPE_MAP = {
    "int16": np.dtype(np.int16),
    "uint8": np.dtype(np.uint8),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


class NiftiSliceUrlAdapter(UrlAdapter):
    """Read-only adapter that renders NIfTI axial slices as PNG images.

    Handles the ``nifti-slice`` scheme. All parameters needed for raw byte
    access are encoded in the URL query string so the adapter never needs to
    parse the NIfTI header itself.
    """

    def schemes(self) -> list[str]:
        return ["nifti-slice"]

    def read_binary_content_from_url(self, url: Url) -> bytes:
        """Read a single slice from a NIfTI file and return it as PNG bytes."""
        nii_path = url.path
        params = parse_qs(url.query)

        z = int(params["z"][0])
        dtype = _DTYPE_MAP[params["dtype"][0]]
        data_offset = int(params["offset"][0])
        w = int(params["w"][0])
        h = int(params["h"][0])
        vmax = float(params["vmax"][0])

        itemsize = dtype.itemsize
        slice_bytes = w * h * itemsize
        file_offset = data_offset + z * slice_bytes

        with open(nii_path, "rb") as f:
            f.seek(file_offset)
            raw = f.read(slice_bytes)

        # Reshape in Fortran order (NIfTI stores data column-major)
        arr = np.frombuffer(raw, dtype=dtype).reshape(w, h, order="F")

        # Normalize to uint8 using pre-computed vmax
        if vmax > 0:
            normalized = np.clip(arr.astype(np.float32) / vmax * 255.0, 0, 255).astype(np.uint8)
        else:
            normalized = np.zeros((w, h), dtype=np.uint8)

        img = Image.fromarray(normalized, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def exists(self, url: Url) -> bool:
        """Check whether the underlying .nii file exists."""
        return Path(url.path).exists()

    def supports_relativization(self) -> bool:
        """Prevent relativization — these URLs must remain absolute."""
        return False
