#!/usr/bin/env python3
"""Хөгжүүлэлтийн энгийн статик сервер.

Ажиллуулах:
    python3 serve.py            # http://localhost:8000
    python3 serve.py 8080       # өөр порт дээр
"""
import http.server
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        # Хөгжүүлэлтийн явцад кэш хийхгүй байх
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Сервер ажиллаж байна:  http://localhost:{PORT}")
        print("Зогсоохын тулд Ctrl+C дарна уу.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nЗогслоо.")
