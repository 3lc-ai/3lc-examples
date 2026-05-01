# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""URL adapter for virtualizing KITTI Velodyne LiDAR point cloud files.

KITTI .bin files store interleaved [x, y, z, intensity] float32 values per point.
This adapter reads from the original .bin files and returns de-interleaved components
(vertices or intensity) on the fly, so the data never needs to be copied.

URL format::

    kitti-velodyne:///absolute/path/to/velodyne/003526.bin?component=vertices
    kitti-velodyne:///absolute/path/to/velodyne/003526.bin?component=intensity

The adapter applies the KITTI alignment matrix ``R_ALIGN = diag(1, -1, 1)`` to vertex
coordinates so that LiDAR orientation matches the camera left/right convention.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from tlc import Url, UrlAdapter

# Alignment matrix to match LiDAR orientation with camera left/right
_R_ALIGN = np.diag([1.0, -1.0, 1.0]).astype(np.float32)


class KittiVelodyneUrlAdapter(UrlAdapter):
    """Read-only adapter that extracts components from KITTI Velodyne .bin files.

    Handles the ``kitti-velodyne`` scheme. The query parameter ``component``
    selects which data to return:

    - ``vertices``: The (x, y, z) coordinates, flattened to (3N,) float32,
      with the KITTI alignment rotation applied.
    - ``intensity``: The reflectance values, flattened to (N,) float32.
    """

    def schemes(self) -> list[str]:
        return ["kitti-velodyne"]

    def read_binary_content_from_url(self, url: Url) -> bytes:
        """Read a KITTI .bin file and return the requested component as raw float32 bytes."""
        bin_path = url.path
        query = parse_qs(url.query)

        components = query.get("component")
        if not components:
            msg = f"Missing 'component' query parameter in URL: {url}"
            raise ValueError(msg)
        component = components[0]

        # Read the raw KITTI binary: float32 × [x, y, z, intensity] per point
        raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        if component == "vertices":
            vertices = raw[:, :3]
            # Apply alignment: flip y-axis so LiDAR matches camera orientation
            aligned = (_R_ALIGN @ vertices.T).T
            return aligned.astype(np.float32, copy=False).reshape(-1).tobytes()
        elif component == "intensity":
            return raw[:, 3].astype(np.float32, copy=False).reshape(-1).tobytes()
        else:
            msg = f"Unknown component '{component}', expected 'vertices' or 'intensity'"
            raise ValueError(msg)

    def exists(self, url: Url) -> bool:
        """Check whether the underlying .bin file exists."""
        return Path(url.path).exists()

    def supports_relativization(self) -> bool:
        """Prevent relativization — these URLs must remain absolute."""
        return False
