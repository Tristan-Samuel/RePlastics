#!/usr/bin/env python3
"""Resize the local Drive export to 256px on the long side and zip it.

Reads the five downloaded folders under ~/Downloads/stage1_binary_v2,
merges unique train/val paths, writes JPEGs to ~/Downloads/stage1_binary_v2_256,
and creates a store-only zip next to that folder.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import zipfile

from PIL import Image, ImageOps


SOURCE_ROOT = Path.home() / "Downloads" / "stage1_binary_v2"
DEST_ROOT = Path.home() / "Downloads" / "stage1_binary_v2_256"
ZIP_PATH = Path.home() / "Downloads" / "stage1_binary_v2_256.zip"
LONG_SIDE = 256
JPEG_QUALITY = 90
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_unique_sources():
    files = {}
    for child in sorted(SOURCE_ROOT.iterdir()):
        if not child.is_dir():
            continue
        for path in child.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.name == ".DS_Store":
                continue
            relative = path.relative_to(child)
            files.setdefault(relative.as_posix(), path)
    return files


def resize_one(job):
    relative, source, dest_root, long_side, quality = job
    destination = dest_root / relative
    destination = destination.with_suffix(".jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > long_side:
            if width >= height:
                new_size = (
                    long_side,
                    max(1, int(height * long_side / width)),
                )
            else:
                new_size = (
                    max(1, int(width * long_side / height)),
                    long_side,
                )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image.save(
            destination,
            format="JPEG",
            quality=quality,
            optimize=True,
        )
    return destination.stat().st_size


def main():
    files = collect_unique_sources()
    if not files:
        raise SystemExit(f"No images under {SOURCE_ROOT}")

    if DEST_ROOT.exists():
        import shutil

        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir(parents=True)

    jobs = [
        (relative, source, DEST_ROOT, LONG_SIDE, JPEG_QUALITY)
        for relative, source in files.items()
    ]
    workers = os.cpu_count() or 4
    print(
        f"Resizing {len(jobs)} unique images -> {DEST_ROOT} "
        f"({workers} workers, long side {LONG_SIDE})",
        flush=True,
    )

    done = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resize_one, job) for job in jobs]
        for future in as_completed(futures):
            total_bytes += future.result()
            done += 1
            if done == 1 or done == len(jobs) or done % 200 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    print(
        f"Wrote {done} JPEGs ({total_bytes / (1024 * 1024):.1f} MB) "
        f"to {DEST_ROOT}",
        flush=True,
    )
    print(f"Zipping {ZIP_PATH} (store-only)...", flush=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for path in DEST_ROOT.rglob("*"):
            if path.is_file():
                archive.write(path, f"stage1_binary_v2_256/{path.relative_to(DEST_ROOT)}")
    zip_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Zip {zip_mb:.1f} MB at {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
