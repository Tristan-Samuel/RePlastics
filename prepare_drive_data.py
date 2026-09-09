#!/usr/bin/env python3
"""Map a Drive dataset onto the metal/plastic split layout used by train.py.

Expected source (no validation/test yet):

    <source>/train/NonPlastic
    <source>/train/Plastic

NonPlastic is treated as metal. Files are not copied; the dest tree uses
symlinks, then split 70% train / 15% validation / 15% test with the same
seed as split_data.py.
"""

from pathlib import Path
import argparse
import shutil

import torch

from split_data import (
    IMAGE_SUFFIXES,
    RANDOM_SEED,
    split_paths,
)
from trashnet import EXPECTED_CLASSES, ensure_split_layout, split_dirs_for


CLASS_ALIASES = {
    "nonplastic": "metal",
    "metal": "metal",
    "plastic": "plastic",
}

SKIP_DIR_NAMES = {
    "train",
    "validation",
    "val",
    "test",
    "models",
    "__pycache__",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build dest/train, dest/validation, and dest/test "
            "from a Drive folder of NonPlastic and Plastic photos."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Drive dataset root, e.g. "
            "/content/drive/MyDrive/Subsystem_3/"
            "dataset/stage1_binary_v2"
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/content/data_real"),
        help="Where to write the metal/plastic split tree.",
    )
    return parser.parse_args()


def class_key(name):
    return "".join(
        character
        for character in name.lower()
        if character.isalnum()
    )


def mapped_class(name):
    return CLASS_ALIASES.get(class_key(name))


def image_files(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name != ".DS_Store"
    )


def iter_candidate_dirs(source):
    train_dir = source / "train"
    if train_dir.is_dir():
        yield train_dir
    yield source


def find_class_folders(source):
    found = {}

    for root in iter_candidate_dirs(source):
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in SKIP_DIR_NAMES:
                continue

            class_name = mapped_class(child.name)
            if class_name is None or class_name in found:
                continue

            found[class_name] = child

        if all(name in found for name in EXPECTED_CLASSES):
            return found

    return found


def unique_destination(destination_dir, source_path):
    destination = destination_dir / source_path.name
    if not destination.exists():
        return destination
    return destination_dir / f"{source_path.parent.name}_{source_path.name}"


def link_file(source_path, destination):
    destination.symlink_to(source_path.resolve())


def resolve_source(source):
    source = Path(source)
    candidates = [source]
    if not source.is_absolute():
        relative = str(source)
        candidates.extend(
            [
                Path("/content/drive/MyDrive") / relative,
                Path("/content/drive") / relative,
            ]
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find Drive dataset at {source}. "
        "Mount Drive first and pass the folder that contains "
        "train/NonPlastic and train/Plastic."
    )


def main():
    args = parse_args()
    source = resolve_source(args.source)
    dest = args.dest

    print(f"Drive source: {source}", flush=True)
    print(f"Split dest: {dest}", flush=True)

    class_folders = find_class_folders(source)
    missing = [
        name
        for name in EXPECTED_CLASSES
        if name not in class_folders
    ]
    if missing:
        found = ", ".join(
            f"{path.name} -> {name}"
            for name, path in class_folders.items()
        ) or "nothing"
        raise SystemExit(
            f"Missing mapped class folders {missing} under {source}. "
            f"Found: {found}. Expected train/NonPlastic and train/Plastic."
        )

    if dest.exists():
        print(f"Replacing existing {dest}", flush=True)
        shutil.rmtree(dest)

    ensure_split_layout(dest)
    dest_splits = split_dirs_for(dest)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    copied = {"train": 0, "validation": 0, "test": 0}

    for class_name in EXPECTED_CLASSES:
        folder = class_folders[class_name]
        paths = image_files(folder)
        print(
            f"  {folder} ({len(paths)} images) -> {class_name}",
            flush=True,
        )

        if not paths:
            raise SystemExit(f"No images in {folder}")

        training_paths, validation_paths, test_paths = split_paths(
            paths,
            generator,
        )
        splits = (
            (dest_splits["train"], training_paths, "train"),
            (dest_splits["validation"], validation_paths, "validation"),
            (dest_splits["test"], test_paths, "test"),
        )

        for split_dir, split_paths_for_class, split_name in splits:
            destination_dir = split_dir / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            for source_path in split_paths_for_class:
                destination = unique_destination(
                    destination_dir,
                    source_path,
                )
                if destination.exists():
                    raise FileExistsError(
                        f"{destination} already exists"
                    )
                link_file(source_path, destination)
                copied[split_name] += 1

    print(flush=True)
    print("Linked images into:", flush=True)
    print(f"  train: {copied['train']}", flush=True)
    print(f"  validation: {copied['validation']}", flush=True)
    print(f"  test: {copied['test']}", flush=True)
    print(
        "Original Drive files are unchanged. "
        "NonPlastic is mapped to metal.",
        flush=True,
    )


if __name__ == "__main__":
    main()
