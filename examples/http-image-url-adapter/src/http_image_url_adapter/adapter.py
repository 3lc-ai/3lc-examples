# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

"""Toy URL adapter plugin demonstrating how to fetch external images over HTTP(S).

This is a minimal example showing how to create a third-party URL adapter plugin
that integrates with 3LC via the ``tlc.url_adapters`` entry point group.

Install with ``pip install .`` (from this directory) and the adapter will
be discovered automatically by 3LC. Any table whose image column contains
``img-https://...`` URLs will have its images fetched on the fly.

.. note::
   This is intentionally simple — a production adapter would add caching,
   retries, timeouts, and authentication.
"""

from __future__ import annotations

from urllib.request import urlopen

from tlcurl.url import Url
from tlcurl.url_adapter import UrlAdapter


class HttpImageUrlAdapter(UrlAdapter):
    """Read-only adapter that fetches images over HTTP(S).

    Handles the ``img-http`` and ``img-https`` schemes by stripping the
    ``img-`` prefix and delegating to :func:`urllib.request.urlopen`.

    Example::

        from tlcurl.url import Url

        url = Url("img-https://picsum.photos/id/10/400/300")
        image_bytes = url.read()  # fetches the JPEG from picsum.photos
    """

    def schemes(self) -> list[str]:
        return ["img-http", "img-https"]

    def read_binary_content_from_url(self, url: Url) -> bytes:
        """Fetch the image bytes from the given URL."""
        real_url = self._to_real_url(url)
        with urlopen(real_url) as response:
            return response.read()

    def exists(self, url: Url) -> bool:
        """Check whether the remote resource exists (HEAD request)."""
        import urllib.request

        real_url = self._to_real_url(url)
        req = urllib.request.Request(real_url, method="HEAD")
        try:
            with urlopen(req) as response:
                return response.status == 200
        except Exception:
            return False

    @staticmethod
    def _to_real_url(url: Url) -> str:
        """Strip the ``img-`` prefix to get a standard HTTP(S) URL."""
        real_scheme = url.scheme.removeprefix("img-")
        return f"{real_scheme}://{url.path}"
