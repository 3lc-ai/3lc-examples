# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""URL adapter for virtualizing DigiFarm field-delineation .npz tiles.

Each tile is a single ``.npz`` archive containing a ``patch`` array of shape
``(H, W, 5)`` (uint8) where channels are ``[R, G, B, extent_class, contour_class]``.
The adapter exposes three views of one tile as PNG bytes — without copying or
re-materializing the data on disk.

URL format::

    digifarm-npz:///abs/path/to/tile.npz?view=rgb
    digifarm-npz:///abs/path/to/tile.npz?view=extent
    digifarm-npz:///abs/path/to/tile.npz?view=contour

Query parameters:

- ``view``: One of ``rgb``, ``extent``, ``contour``.

Mask views (``extent``, ``contour``) are writable. Writes land in a sidecar PNG
next to the tile (e.g. ``tile.extent.edit.png``); subsequent reads prefer the
sidecar over the original channel in the .npz. The source archive is never
modified.
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from PIL import Image
from tlc import Url, UrlAdapter

_VIEWS = frozenset({"rgb", "extent", "contour"})
_WRITABLE_VIEWS = frozenset({"extent", "contour"})


def _sidecar_path(npz_path: Path, view: str) -> Path:
    """Location of the edit sidecar PNG for a given tile/view."""
    return npz_path.parent / f"{npz_path.stem}.{view}.edit.png"


def _parse_view(url: Url) -> str:
    view = parse_qs(url.query).get("view", ["rgb"])[0]
    if view not in _VIEWS:
        msg = f"Unknown view {view!r}; expected one of {sorted(_VIEWS)}"
        raise ValueError(msg)
    return view


class DigiFarmNpzUrlAdapter(UrlAdapter):
    """URL adapter that renders DigiFarm tile views as PNGs and virtualizes mask edits."""

    def schemes(self) -> list[str]:
        return ["digifarm-npz"]

    def read_binary_content_from_url(self, url: Url) -> bytes:
        view = _parse_view(url)
        npz_path = Path(url.path)

        # Prefer the edit sidecar if one exists for this mask view.
        if view in _WRITABLE_VIEWS:
            sidecar = _sidecar_path(npz_path, view)
            if sidecar.exists():
                return sidecar.read_bytes()

        with np.load(npz_path) as data:
            patch: np.ndarray = data["patch"]

        if view == "rgb":
            img = Image.fromarray(patch[..., :3], mode="RGB")
        elif view == "extent":
            img = Image.fromarray(patch[..., 3], mode="L")
        else:
            img = Image.fromarray(patch[..., 4], mode="L")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def write_binary_content_to_url(self, url: Url, content: bytes) -> Url:
        view = _parse_view(url)
        if view not in _WRITABLE_VIEWS:
            msg = f"View {view!r} is read-only; only {sorted(_WRITABLE_VIEWS)} can be written"
            raise PermissionError(msg)
        sidecar = _sidecar_path(Path(url.path), view)
        sidecar.write_bytes(content)
        return url

    def is_writable(self, url: Url) -> bool:
        try:
            return _parse_view(url) in _WRITABLE_VIEWS
        except ValueError:
            return False

    def exists(self, url: Url) -> bool:
        return Path(url.path).exists()

    def supports_relativization(self) -> bool:
        return False
