import sys
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as ServerClass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.append(str(Path(__file__).parent))

from data_sources import Deterministic3DPointCloudDataset

dataset = Deterministic3DPointCloudDataset(size=10)


def handle_get_request(path: str, query_params: dict, request_headers: dict) -> bytes:
    """
    Return a chunk of binary data for the given key.
    """
    key = path.lstrip("/").replace(".raw", "")
    _, sample_idx, attribute = key.split("-")  # parse our own url format; sample-0-intensities.raw
    arr = dataset[sample_idx][attribute]
    return arr.tobytes()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        # Convert headers to a plain dict with string keys/values
        headers_dict = {k: v for k, v in self.headers.items()}

        try:
            body = handle_get_request(parsed.path, query, headers_dict)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            # Minimal error response as bytes
            body = f"error: {exc}".encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


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
