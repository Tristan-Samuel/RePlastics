#!/usr/bin/env python3
"""Resize stage1_binary_v2 to 256px on the long side and zip it.

Reads the live Drive folder (train/val, NonPlastic/Plastic), drops exact
train copies of files that already sit in val, writes JPEGs to
~/Downloads/stage1_binary_v2_256, and creates a store-only zip.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import shutil
import zipfile

from PIL import Image, ImageOps


SOURCE_ROOT = Path(
    "/Users/tristan/Library/CloudStorage/GoogleDrive-intern"
    "@replasticrecycle.com/My Drive/stage1_binary_v2"
)
DEST_ROOT = Path.home() / "Downloads" / "stage1_binary_v2_256"
ZIP_PATH = Path.home() / "Downloads" / "stage1_binary_v2_256.zip"
DRIVE_ZIP = Path(
    "/Users/tristan/Library/CloudStorage/GoogleDrive-intern"
    "@replasticrecycle.com/My Drive/stage1_binary_v2_256.zip"
)
LONG_SIDE = 256
JPEG_QUALITY = 90
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_FOLDERS = (
    ("train", "NonPlastic"),
    ("train", "Plastic"),
    ("val", "NonPlastic"),
    ("val", "Plastic"),
)


def image_files(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name != ".DS_Store"
    )


def collect_unique_sources():
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"Missing Drive dataset: {SOURCE_ROOT}")

    extra = [
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.name not in {"train", "val"}
    ]
    if extra:
        print(f"Ignoring extra top-level names: {extra}", flush=True)

    grouped = defaultdict(list)
    for split, class_name in CLASS_FOLDERS:
        folder = SOURCE_ROOT / split / class_name
        for path in image_files(folder):
            grouped[path.name].append((split, class_name, path))

    files = {}
    dropped = []
    for name, locations in grouped.items():
        splits = {split for split, _, _ in locations}
        if "train" in splits and "val" in splits:
            keep = [item for item in locations if item[0] == "val"]
            dropped.extend(
                f"{split}/{class_name}/{name}"
                for split, class_name, _ in locations
                if split == "train"
            )
            locations = keep
        for split, class_name, path in locations:
            relative = f"{split}/{class_name}/{name}"
            files[relative] = path

    return files, dropped


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
    files, dropped = collect_unique_sources()
    if not files:
        raise SystemExit(f"No images under {SOURCE_ROOT}")

    print(f"Drive source: {SOURCE_ROOT}", flush=True)
    print(f"Unique photos: {len(files)}", flush=True)
    if dropped:
        print(
            f"Dropped {len(dropped)} train copies that are identical in val:",
            flush=True,
        )
        for item in dropped:
            print(f"  {item}", flush=True)

    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir(parents=True)

    jobs = [
        (relative, source, DEST_ROOT, LONG_SIDE, JPEG_QUALITY)
        for relative, source in files.items()
    ]
    workers = min(4, os.cpu_count() or 4)
    print(
        f"Resizing {len(jobs)} images -> {DEST_ROOT} "
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
                archive.write(
                    path,
                    f"stage1_binary_v2_256/{path.relative_to(DEST_ROOT)}",
                )
    zip_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Zip {zip_mb:.1f} MB at {ZIP_PATH}", flush=True)

    DRIVE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ZIP_PATH, DRIVE_ZIP)
    print(f"Copied zip to {DRIVE_ZIP}", flush=True)


if __name__ == "__main__":
    main()
