import json
import sys
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as ServerClass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.append(str(Path(__file__).parent))

from data_sources import Deterministic3DPointCloudDataset

## Constants

lookup_table_path = Path(__file__).parent / "bulk_data/3/lookup_table.json"
with open(lookup_table_path) as f:
    lookup_table = json.load(f)

dataset = Deterministic3DPointCloudDataset(size=10)


def handle_get_request(path: str, query_params: dict, request_headers: dict) -> bytes:
    """
    Return a chunk of binary data for the given key.
    """
    # Example placeholder response; safe to replace
    key = path.lstrip("/").replace(".raw", "")
    pieces = lookup_table[key]

    data = []

    ranges = sorted(lookup_table[key].keys(), key=lambda x: int(x.split("-")[0]))
    for range in ranges:
        val = pieces[range]
        arr = dataset[val["sample"]][val["attribute"]]
        data.append(arr.tobytes())
    return b"".join(data)


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        # Convert headers to a plain dict with string keys/values
        headers_dict = {k: v for k, v in self.headers.items()}

        try:
            payload = handle_get_request(parsed.path, query, headers_dict)
            # Ensure payload is bytes; allow simple ergonomic returns
            if isinstance(payload, bytes):
                body = payload
            elif isinstance(payload, str):
                body = payload.encode("utf-8")
            else:
                # Fallback: serialize unknown payloads to JSON bytes
                body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            # Minimal error response as bytes
            body = f"error: {exc}".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # Quiet default logging a bit if desired; keep default for now
    # def log_message(self, format: str, *args) -> None:
    #     return


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 2233
    server_address = (host, port)
    httpd = ServerClass(server_address, RequestHandler)
    print(f"HTTP server listening on http://{host}:{port} (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        httpd.server_close()
