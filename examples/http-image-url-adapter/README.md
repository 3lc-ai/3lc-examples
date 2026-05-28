# http-image-url-adapter

Toy example showing how to write a 3LC URL adapter plugin that fetches images over HTTP(S).

This is **not** a production-ready adapter — it has no caching, retries, or authentication. Its purpose is to demonstrate the URL adapter entry-point mechanism so you can build your own adapter for any external data source.

## Quick start

```bash
# Install the adapter plugin (editable mode recommended for development)
pip install -e .

# Create a small demo table with images from picsum.photos
create-http-image-demo-table
```

After running `create-http-image-demo-table`, open the 3LC Dashboard to browse the table. The images are fetched on the fly from picsum.photos whenever 3LC reads a row.

## How it works

### 1. Custom URL scheme

The adapter registers `img-http` and `img-https` schemes. When 3LC encounters a URL like:

```
img-https://picsum.photos/id/10/400/300
```

it strips the `img-` prefix and fetches `https://picsum.photos/id/10/400/300` over plain HTTP(S).

### 2. Entry-point discovery

The plugin declares an entry point in `pyproject.toml`:

```toml
[project.entry-points."tlc.url_adapters"]
img-http = "http_image_url_adapter:HttpImageUrlAdapter"
```

At import time, 3LC calls `UrlAdapterRegistry.discover_entrypoint_adapters()` which finds all installed packages with the `tlc.url_adapters` entry-point group and registers their adapters. No manual registration needed.

### 3. Using it in Python

```python
from tlcurl.url import Url

url = Url("img-https://picsum.photos/id/10/400/300")
image_bytes = url.read()  # fetches the JPEG from picsum.photos
```

## Building your own adapter

Subclass `tlcurl.url_adapter.UrlAdapter` and implement:

- `schemes()` — which URL schemes to handle
- `read_binary_content_from_url(url)` — fetch the bytes
- `exists(url)` — check if the resource exists

Then declare an entry point so 3LC discovers it automatically.
