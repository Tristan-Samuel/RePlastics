#!/usr/bin/env python3
"""Map a Drive dataset onto the metal/plastic split layout used by train.py.

Expected source:

    <source>/train/NonPlastic
    <source>/train/Plastic
    <source>/val/NonPlastic   (optional)
    <source>/val/Plastic      (optional)

NonPlastic is treated as metal. Files are not copied; dest uses symlinks.

If Drive already has train/ and val/, val is kept as validation and 15% of
train is held out as test. If only train/ exists, split 70/15/15.
"""

from pathlib import Path
import argparse
import shutil

import torch

from split_data import (
    IMAGE_SUFFIXES,
    RANDOM_SEED,
    TEST_PERCENT,
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
            "/content/drive/MyDrive/stage1_binary_v2"
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


def mapped_class_folders(root):
    found = {}
    if not root.is_dir():
        return found

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIR_NAMES:
            continue

        class_name = mapped_class(child.name)
        if class_name is None or class_name in found:
            continue

        found[class_name] = child

    return found


def discover_source_splits(source):
    result = {}
    aliases = (
        ("train", "train"),
        ("val", "validation"),
        ("validation", "validation"),
        ("test", "test"),
    )

    for folder_name, split_name in aliases:
        folders = mapped_class_folders(source / folder_name)
        if folders and split_name not in result:
            result[split_name] = folders

    if "train" not in result:
        folders = mapped_class_folders(source)
        if folders:
            result["train"] = folders

    return result


def unique_destination(destination_dir, source_path):
    destination = destination_dir / source_path.name
    if not destination.exists():
        return destination
    return destination_dir / f"{source_path.parent.name}_{source_path.name}"


def link_file(source_path, destination):
    destination.symlink_to(source_path.resolve())


def link_paths(paths, split_dir, class_name, copied, split_name):
    destination_dir = split_dir / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    for source_path in paths:
        destination = unique_destination(destination_dir, source_path)
        if destination.exists():
            raise FileExistsError(f"{destination} already exists")
        link_file(source_path, destination)
        copied[split_name] += 1


def holdout_test(paths, generator):
    permutation = torch.randperm(
        len(paths),
        generator=generator,
    ).tolist()
    shuffled = [paths[index] for index in permutation]
    test_size = int(len(shuffled) * TEST_PERCENT)
    return shuffled[test_size:], shuffled[:test_size]


def require_classes(folders, label):
    missing = [
        name
        for name in EXPECTED_CLASSES
        if name not in folders
    ]
    if missing:
        found = ", ".join(
            f"{path.name} -> {name}"
            for name, path in folders.items()
        ) or "nothing"
        raise SystemExit(
            f"Missing mapped class folders {missing} in {label}. "
            f"Found: {found}. Expected NonPlastic and Plastic."
        )


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

    source_splits = discover_source_splits(source)
    if "train" not in source_splits:
        raise SystemExit(
            f"No train/NonPlastic and train/Plastic under {source}."
        )

    require_classes(source_splits["train"], "train")
    if "validation" in source_splits:
        require_classes(source_splits["validation"], "validation")
    if "test" in source_splits:
        require_classes(source_splits["test"], "test")

    if dest.exists():
        print(f"Replacing existing {dest}", flush=True)
        shutil.rmtree(dest)

    ensure_split_layout(dest)
    dest_splits = split_dirs_for(dest)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    copied = {"train": 0, "validation": 0, "test": 0}

    if "validation" in source_splits and "test" in source_splits:
        print("Using Drive train / val / test as-is.", flush=True)
        for class_name in EXPECTED_CLASSES:
            for split_name in ("train", "validation", "test"):
                folder = source_splits[split_name][class_name]
                paths = image_files(folder)
                print(
                    f"  {folder} ({len(paths)}) -> {split_name}/{class_name}",
                    flush=True,
                )
                link_paths(
                    paths,
                    dest_splits[split_name],
                    class_name,
                    copied,
                    split_name,
                )
    elif "validation" in source_splits:
        print(
            "Found Drive train/val. Keeping val; "
            "holding out 15% of train as test.",
            flush=True,
        )
        for class_name in EXPECTED_CLASSES:
            train_folder = source_splits["train"][class_name]
            val_folder = source_splits["validation"][class_name]
            train_paths = image_files(train_folder)
            val_paths = image_files(val_folder)
            if not train_paths:
                raise SystemExit(f"No images in {train_folder}")
            training_paths, test_paths = holdout_test(
                train_paths,
                generator,
            )
            print(
                f"  {class_name}: {len(training_paths)} train, "
                f"{len(val_paths)} val, {len(test_paths)} test",
                flush=True,
            )
            link_paths(
                training_paths,
                dest_splits["train"],
                class_name,
                copied,
                "train",
            )
            link_paths(
                val_paths,
                dest_splits["validation"],
                class_name,
                copied,
                "validation",
            )
            link_paths(
                test_paths,
                dest_splits["test"],
                class_name,
                copied,
                "test",
            )
    else:
        print("No Drive val split; using 70/15/15 from train.", flush=True)
        for class_name in EXPECTED_CLASSES:
            folder = source_splits["train"][class_name]
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
            link_paths(
                training_paths,
                dest_splits["train"],
                class_name,
                copied,
                "train",
            )
            link_paths(
                validation_paths,
                dest_splits["validation"],
                class_name,
                copied,
                "validation",
            )
            link_paths(
                test_paths,
                dest_splits["test"],
                class_name,
                copied,
                "test",
            )

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
