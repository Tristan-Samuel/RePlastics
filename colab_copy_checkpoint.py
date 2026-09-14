#!/usr/bin/env python3
"""Copy the checkpoint onto the VM disk while Drive FUSE is still connected.

Drive FUSE often drops (Errno 107) after a long zip unpack. Call this
immediately after mount, before prepare_drive_data.py.
"""

from pathlib import Path
import time

DEST = Path("/content/models/resnext50_metal_plastic.pt")
CANDIDATES = [
    Path(
        "/content/drive/MyDrive/metal-plastic-sorting/"
        "resnext50_metal_plastic_full_fc_9907.pt"
    ),
    Path(
        "/content/drive/MyDrive/metal-plastic-sorting/"
        "resnext50_metal_plastic.pt"
    ),
    Path("/content/drive/MyDrive/stage1_binary_v2/resnext50_metal_plastic.pt"),
]
MIN_BYTES = 10_000_000


def remount():
    try:
        from google.colab import drive

        print("Remounting Google Drive...", flush=True)
        drive.mount("/content/drive", force_remount=True)
    except Exception as err:
        print(f"Remount failed: {err}", flush=True)


def copy_chunked(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    copied = 0
    with src.open("rb") as fsrc, tmp.open("wb") as fdst:
        while True:
            chunk = fsrc.read(8 * 1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk)
    tmp.replace(dest)
    return copied


def usable(path):
    try:
        return path.is_file() and path.stat().st_size >= MIN_BYTES
    except OSError:
        return False


def main():
    if usable(DEST):
        print(
            f"Using checkpoint already on the VM {DEST} "
            f"({DEST.stat().st_size / (1024 * 1024):.1f} MB)",
            flush=True,
        )
        print("COLAB_STEP_OK", flush=True)
        return

    last_err = None
    copied = False
    for attempt in range(1, 4):
        if attempt > 1:
            remount()
            time.sleep(2)
        for src in CANDIDATES:
            try:
                if not src.is_file():
                    print(f"Missing {src}", flush=True)
                    continue
                print(
                    f"Copying checkpoint from {src} (attempt {attempt})",
                    flush=True,
                )
                nbytes = copy_chunked(src, DEST)
                print(
                    f"Copied {nbytes / (1024 * 1024):.1f} MB from {src}",
                    flush=True,
                )
                copied = True
                last_err = None
                break
            except OSError as err:
                last_err = err
                print(f"Checkpoint copy failed: {err}", flush=True)
                part = DEST.with_suffix(".part")
                if part.exists():
                    part.unlink()
        if copied:
            break

    if not copied or not usable(DEST):
        detail = f" Last error: {last_err}" if last_err else ""
        raise SystemExit(
            "Missing checkpoint. Copy "
            "models/resnext50_metal_plastic.pt to "
            "MyDrive/metal-plastic-sorting/ and re-run."
            + detail
        )
    print("COLAB_STEP_OK", flush=True)


if __name__ == "__main__":
    main()
