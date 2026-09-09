#!/usr/bin/env python3
"""Upload a large file to a Colab VM in small pieces, with progress.

`colab upload` reads the whole file, base64-encodes it, and sends one JSON PUT.
A 2GB zip becomes a ~2.7GB payload, prints nothing while that happens, and
often dies with an SSL error. This helper uploads 4MB pieces instead.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


CHUNK_SIZE = 4 * 1024 * 1024
SESSIONS_PATH = Path.home() / ".config/colab-cli/sessions.json"
RETRIES = 6


def log(message: str) -> None:
    print(message, flush=True)


def load_session(name: str) -> tuple[str, str]:
    if not SESSIONS_PATH.exists():
        raise SystemExit(f"Missing Colab session file: {SESSIONS_PATH}")

    sessions = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
    session = sessions.get(name)
    if not session:
        raise SystemExit(f"Colab session '{name}' was not found.")

    return session["url"].rstrip("/"), session["token"]


def put_bytes(base_url: str, token: str, remote_path: str, data: bytes) -> None:
    quoted = urllib.parse.quote(remote_path.strip("/"), safe="/")
    query = urllib.parse.urlencode(
        {
            "authuser": "0",
            "colab-runtime-proxy-token": token,
        }
    )
    url = f"{base_url}/api/contents/{quoted}?{query}"
    payload = json.dumps(
        {
            "name": Path(remote_path).name,
            "path": remote_path,
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        }
    ).encode("ascii")

    last_error = None
    for attempt in range(1, RETRIES + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == RETRIES:
                break
            log(f"  retry {attempt}/{RETRIES - 1} after a network error")
            time.sleep(min(2**attempt, 16))

    reason = getattr(last_error, "reason", last_error)
    raise SystemExit(
        f"Upload of {remote_path} failed: {type(last_error).__name__}: {reason}"
    )


def upload_parts(session_name: str, local_path: Path, prefix: str) -> int:
    base_url, token = load_session(session_name)
    size = local_path.stat().st_size
    if size == 0:
        raise SystemExit(f"{local_path} is empty")

    part_count = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
    log(
        f"Uploading {local_path.name} ({size / (1024 * 1024):.1f} MB) "
        f"in {part_count} chunks of 4MB"
    )

    sent = 0
    part = 0
    with local_path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            part += 1
            remote_path = f"{prefix}.{part:04d}"
            put_bytes(base_url, token, remote_path, chunk)
            sent += len(chunk)
            percent = 100 * sent / size
            log(
                f"  {part}/{part_count}  {sent / (1024 * 1024):.1f}/"
                f"{size / (1024 * 1024):.1f} MB  ({percent:.0f}%)"
            )

    return part_count


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: colab_upload.py SESSION LOCAL_PATH REMOTE_PREFIX"
        )

    session_name, local_name, prefix = sys.argv[1:4]
    local_path = Path(local_name)
    if not local_path.is_file():
        raise SystemExit(f"Missing file: {local_path}")

    count = upload_parts(session_name, local_path, prefix)
    log(f"Uploaded {count} parts as {prefix}.0001...")


if __name__ == "__main__":
    main()
