#!/usr/bin/env python3
"""Mount Google Drive on a Colab CLI session.

`colab drivemount` posts to /tun/m/credentials-propagation without the
Tunnel Frontend header. The Colab front door then returns the generic HTML
400 page, the kernel is left waiting, and train.py never sees the photos.

This helper sends the same headers the keep-alive ping already uses, then
falls back to the browser notebook if propagation still fails.
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from urllib.parse import quote

from colab_cli.auth import get_credentials
from colab_cli.client import COLAB_TUNNEL_HEADER
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime


COLAB_HOST = "https://colab.research.google.com"
TFE_HEADERS = {
    "Accept": "application/json",
    "X-Colab-Client-Agent": "colab-cli",
    COLAB_TUNNEL_HEADER["key"]: COLAB_TUNNEL_HEADER["value"],
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_colab_json(text: str) -> dict:
    body = text.strip()
    if body.startswith(")]}'"):
        body = body.split("\n", 1)[-1]
    return json.loads(body)


def session_url(endpoint: str) -> str:
    backend_path = f"/tun/m/{endpoint}"
    return (
        f"{COLAB_HOST}/notebooks/empty.ipynb"
        f"?dbu={quote(backend_path, safe='')}"
        f"#datalabBackendUrl={COLAB_HOST}{backend_path}"
    )


def output_text(outputs: list) -> str:
    chunks = []
    for item in outputs or []:
        if item.get("text"):
            chunks.append(item["text"])
        elif item.get("output_type") == "error":
            ename = item.get("ename", "Error")
            evalue = item.get("evalue", "")
            chunks.append(f"{ename}: {evalue}\n")
            traceback = item.get("traceback") or []
            if traceback:
                chunks.append("".join(traceback) + "\n")
    return "".join(chunks)


def drive_is_mounted(runtime: ColabRuntime, quiet: bool = False) -> bool:
    outputs = runtime.execute_code(
        "from pathlib import Path\n"
        "print('DRIVE_MOUNTED' if Path('/content/drive/MyDrive').is_dir() "
        "else 'DRIVE_NOT_MOUNTED', flush=True)\n",
        timeout=60,
    )
    text = output_text(outputs)
    mounted = "DRIVE_MOUNTED" in text
    if text and (mounted or not quiet):
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    return mounted


def send_colab_reply(deserialize_msg: dict, wsclient) -> None:
    msg_id = deserialize_msg.get("metadata", {}).get("colab_msg_id")
    reply = wsclient.session.msg(
        "input_reply",
        {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
    )
    if "header" in deserialize_msg:
        reply["parent_header"] = deserialize_msg["header"]
    wsclient.stdin_channel.send(reply)


def propagate_drive_credentials(endpoint: str) -> bool:
    url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{endpoint}"
    params = {
        "authuser": "0",
        "authtype": "dfs_ephemeral",
        "version": "2",
        "dryrun": "true",
        "propagate": "true",
        "record": "false",
    }
    creds = get_credentials(
        state.client_oauth_config,
        provider=state.auth_provider,
    )

    log(f"Requesting Drive credentials from {url}")
    get_resp = creds.request("GET", url, params=params, headers=TFE_HEADERS)
    log(f"  GET {get_resp.status_code}")
    if get_resp.status_code != 200:
        log(get_resp.text[:500])
        return False

    token = parse_colab_json(get_resp.text).get("token")
    if not token:
        log("Drive credential GET did not return an XSRF token.")
        return False

    post_headers = dict(TFE_HEADERS)
    post_headers["X-Goog-Colab-Token"] = token
    post_resp = creds.request(
        "POST",
        url,
        params=params,
        headers=post_headers,
        data={"file_id": "empty.ipynb"},
    )
    log(f"  POST dryrun {post_resp.status_code}")
    if post_resp.status_code != 200:
        log(post_resp.text[:500])
        return False

    payload = parse_colab_json(post_resp.text)
    if not payload.get("success"):
        uri = payload.get("unauthorized_redirect_uri")
        if not uri:
            log("Drive auth did not return a consent URL.")
            return False
        log("Open this Google Drive consent URL, finish the page, then press Enter:")
        log("")
        log(uri)
        log("")
        webbrowser.open(uri)
        if sys.stdin.isatty():
            input("Press Enter after Google says you can close the tab: ")
        else:
            with open("/dev/tty", encoding="utf-8") as tty:
                tty.readline()

    params["dryrun"] = "false"
    log("Propagating Drive credentials onto the VM...")
    post_resp = creds.request(
        "POST",
        url,
        params=params,
        headers=post_headers,
        data={"file_id": "empty.ipynb"},
    )
    log(f"  POST propagate {post_resp.status_code}")
    if post_resp.status_code != 200:
        log(post_resp.text[:500])
        return False
    return True


def wait_for_browser_mount(runtime: ColabRuntime, endpoint: str) -> bool:
    connect_url = session_url(endpoint)
    log("CLI Drive propagation failed. Mount Drive in the Colab tab instead.")
    log("This session is already running. Open:")
    log("")
    log(connect_url)
    log("")
    log("In a cell run:")
    log("  from google.colab import drive")
    log("  drive.mount('/content/drive')")
    log("Waiting for /content/drive/MyDrive...")
    webbrowser.open(connect_url)

    deadline = time.time() + 900
    while time.time() < deadline:
        if drive_is_mounted(runtime, quiet=True):
            return True
        remaining = int(deadline - time.time())
        log(f"Still waiting for Drive ({remaining}s left)...")
        time.sleep(8)
    return False


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: colab_drivemount.py SESSION")

    name = sys.argv[1]
    session = state.store.get(name)
    if not session:
        log(f"Colab session '{name}' was not found.")
        return 1

    runtime = ColabRuntime(
        session.url,
        session.token,
        session_name=session.name,
        kernel_id=session.kernel_id,
        session_id=session.session_id,
    )
    try:
        if drive_is_mounted(runtime):
            log("Google Drive is already mounted")
            return 0

        result = {"ok": False}

        def drivefs_hook(deserialize_msg, wsclient):
            content = deserialize_msg.get("content", {})
            if content.get("request", {}).get("authType") != "dfs_ephemeral":
                return False
            try:
                result["ok"] = propagate_drive_credentials(session.endpoint)
            except Exception as exc:
                log(f"Drive credential propagation crashed: {exc}")
                result["ok"] = False
            send_colab_reply(deserialize_msg, wsclient)
            return True

        runtime.colab_request_hook = drivefs_hook
        log("Mounting Google Drive on the VM...")
        outputs = runtime.execute_code(
            "from google.colab import drive\ndrive.mount('/content/drive')",
            allow_stdin=True,
            timeout=600,
        )
        text = output_text(outputs)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()

        if result["ok"] and drive_is_mounted(runtime):
            log("Google Drive mounted")
            return 0

        if wait_for_browser_mount(runtime, session.endpoint):
            log("Google Drive mounted from the Colab tab")
            return 0

        log("Google Drive is still not mounted at /content/drive/MyDrive.")
        return 1
    finally:
        runtime.stop()


if __name__ == "__main__":
    sys.exit(main())
