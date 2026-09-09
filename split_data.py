from pathlib import Path
import argparse
import shutil

import torch

from trashnet import (
    EXPECTED_CLASSES,
    ensure_split_layout,
    split_dirs_for,
)


VALIDATION_PERCENT = 0.15
TEST_PERCENT = 0.15
RANDOM_SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split dest/metal and dest/plastic into "
            "70% train / 15% validation / 15% test."
        )
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root. Looks for dest/metal and "
            "dest/plastic, then writes dest/train, "
            "dest/validation, and dest/test. Use "
            "data_real for in-the-wild photos so the "
            "original data/ tree is left alone."
        ),
    )
    return parser.parse_args()


def new_source_folders(data_dir):
    folders = []

    for class_name in EXPECTED_CLASSES:
        folder = Path(data_dir) / class_name

        if folder.is_dir():
            folders.append(folder)

    return folders


def image_files(folder):
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_paths(paths, generator):
    permutation = torch.randperm(
        len(paths),
        generator=generator,
    ).tolist()

    shuffled = [
        paths[index]
        for index in permutation
    ]

    test_size = int(len(shuffled) * TEST_PERCENT)
    validation_size = int(len(shuffled) * VALIDATION_PERCENT)

    test_paths = shuffled[:test_size]
    validation_paths = shuffled[
        test_size:test_size + validation_size
    ]
    training_paths = shuffled[test_size + validation_size:]

    return training_paths, validation_paths, test_paths


def hoist_splits_out_of_trashnet(data_dir):
    data_dir = Path(data_dir)

    if data_dir.name != "trashnet":
        return

    parent = data_dir.parent
    dest_splits = split_dirs_for(data_dir)

    for split_name, split_dir in dest_splits.items():
        destination = parent / split_name

        if destination.exists():
            raise FileExistsError(
                f"Cannot move {split_dir} to {destination}: "
                "that folder already exists."
            )

        shutil.move(str(split_dir), str(destination))
        print(f"Moved {split_dir} -> {destination}")

    shutil.rmtree(data_dir)
    print(f"Removed empty wrapper {data_dir}")


def main():
    args = parse_args()
    data_dir = args.dest
    dest_splits = split_dirs_for(data_dir)
    source_folders = new_source_folders(data_dir)

    if source_folders:
        ensure_split_layout(data_dir)

        paths_by_class = {
            class_name: []
            for class_name in EXPECTED_CLASSES
        }

        print(f"Dataset root: {data_dir}")
        print("New folders:")

        for folder in source_folders:
            class_name = folder.name
            files = image_files(folder)

            print(f"  {folder}: {len(files)} -> {class_name}")

            paths_by_class[class_name].extend(files)

        generator = torch.Generator().manual_seed(RANDOM_SEED)

        copied = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        for class_name in EXPECTED_CLASSES:
            paths = paths_by_class[class_name]

            if not paths:
                continue

            (
                training_paths,
                validation_paths,
                test_paths,
            ) = split_paths(paths, generator)

            splits = (
                (dest_splits["train"], training_paths, "train"),
                (
                    dest_splits["validation"],
                    validation_paths,
                    "validation",
                ),
                (dest_splits["test"], test_paths, "test"),
            )

            for split_dir, split_paths_for_class, split_name in splits:
                destination_dir = split_dir / class_name
                destination_dir.mkdir(parents=True, exist_ok=True)

                for source_path in split_paths_for_class:
                    destination = destination_dir / source_path.name

                    if destination.exists():
                        raise FileExistsError(
                            f"{destination} already exists. "
                            "Rename the new file before splitting."
                        )

                    shutil.copy2(source_path, destination)
                    copied[split_name] += 1

        for folder in source_folders:
            shutil.rmtree(folder)

        print()
        print("Added new images into:")
        print(f"  train: {copied['train']} images")
        print(f"  validation: {copied['validation']} images")
        print(f"  test: {copied['test']} images")
    else:
        print(
            f"No extra metal/plastic folders to split "
            f"in {data_dir}."
        )

    print()
    hoist_splits_out_of_trashnet(data_dir)


if __name__ == "__main__":
    main()
