#!/usr/bin/env python3
"""Tiny local OpenAI-compatible proxy for the Meituan basicai gateway.

asibench's judge transport (litellm) cannot add custom headers, but basicai
requires ``limit-price`` on every call.  Run this on the machine executing
asibench and point ASIBENCH_JUDGE_API_BASE at it::

    python eval/judge_proxy_basicai.py --port 8799
    ASIBENCH_JUDGE_API_BASE=http://127.0.0.1:8799/v1
    ASIBENCH_JUDGE_API_KEY_ENV=ASIBENCH_JUDGE_API_KEY   # any non-empty value

The proxy replaces Authorization with the real app key (read from
/tmp/basicai_key or BASICAI_KEY_FILE) and injects ``limit-price: 2500``.
It listens on loopback only and never logs the key.
"""
import argparse
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("BASICAI_UPSTREAM", "https://basicaiservice.sankuai.com/basicai/v1")
LIMIT_PRICE = os.environ.get("BASICAI_LIMIT_PRICE", "2500")
# ASI-Bench's scorer asks for openai/gpt-5.5 by default; map every judge call to
# the model the user selected (gpt-5.6-sol). Empty string disables rewriting.
MODEL_OVERRIDE = os.environ.get("BASICAI_MODEL_OVERRIDE", "gpt-5.6-sol")
KEY_FILE = os.environ.get("BASICAI_KEY_FILE", "/tmp/basicai_key")

with open(KEY_FILE) as f:
    API_KEY = f.read().strip()

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
    "content-length", "authorization",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter, no headers
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        if body and MODEL_OVERRIDE and b'"model"' in body[:4096]:
            try:
                import json
                payload = json.loads(body)
                if isinstance(payload, dict) and "model" in payload:
                    payload["model"] = MODEL_OVERRIDE
                    # gpt-5.6-sol rejects temperature != 1 ("only the default
                    # (1) value is supported"); asibench scorers default to 0.0.
                    payload.pop("temperature", None)
                    body = json.dumps(payload).encode()
            except Exception:
                pass
        path = self.path
        # accept clients configured with either "...:8799" or "...:8799/v1"
        if UPSTREAM.rstrip("/").endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        url = UPSTREAM.rstrip("/") + path
        req = urllib.request.Request(url, data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in HOP_BY_HOP:
                req.add_header(k, v)
        req.add_header("Authorization", "Bearer " + API_KEY)
        req.add_header("limit-price", LIMIT_PRICE)
        if body is not None:
            req.add_header("Content-Length", str(len(body)))
        try:
            resp = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        except Exception as e:
            data = ("proxy upstream error: %s" % type(e).__name__).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except Exception:
            pass

    do_POST = _proxy
    do_GET = _proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("judge proxy on http://%s:%d -> %s" % (a.host, a.port, UPSTREAM), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
