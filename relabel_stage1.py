#!/usr/bin/env python3
"""Review and fix Drive stage1 labels; optionally move some val into train.

Drive layout (do not change split names):

    stage1_binary_v2/train|val / NonPlastic|Plastic

NonPlastic is metal. Moves stay in the same split (train stays train).
Nothing is written unless you pass --write.

Typical order:
  1. python relabel_stage1.py apply --known          # dry-run the two lotion bottles
  2. python relabel_stage1.py suggest                # model review queue (optional)
  3. Edit relabel_proposals.tsv, delete foil trays
  4. python relabel_stage1.py apply --file relabel_proposals.tsv --write
  5. python relabel_stage1.py promote-val            # dry-run half of val -> train
  6. python relabel_stage1.py promote-val --write
  7. python resize_stage1.py                         # rebuild 224 zip onto Drive
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


SOURCE_ROOT = Path(
    "/Users/tristan/Library/CloudStorage/GoogleDrive-intern"
    "@replasticrecycle.com/My Drive/stage1_binary_v2"
)
LOCKED_TEST_FILE = Path("stage1_locked_test_stems.txt")
LOCKED_VAL_FILE = Path("stage1_locked_val_stems.txt")
PROPOSALS_FILE = Path("relabel_proposals.tsv")
PROPOSALS_PNG = Path("relabel_proposals.png")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_FOLDERS = ("NonPlastic", "Plastic")
SPLITS = ("train", "val")
OTHER_CLASS = {"NonPlastic": "Plastic", "Plastic": "NonPlastic"}
FOLDER_TO_LABEL = {"NonPlastic": "metal", "Plastic": "plastic"}
LABEL_TO_FOLDER = {"metal": "NonPlastic", "plastic": "Plastic"}

# Visually confirmed: plastic bottles sitting in NonPlastic/val.
KNOWN_FLIPS = (
    (
        "val",
        "NonPlastic",
        "Others_capture_00060_20251126_160916_498646.jpg",
        "Plastic",
        "Aveeno lotion bottle",
    ),
    (
        "val",
        "NonPlastic",
        "Others_capture_00300_20251204_124035_096915.png",
        "Plastic",
        "white lotion bottle",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fix Drive NonPlastic/Plastic labels and optionally "
            "move half of val into train."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SOURCE_ROOT,
        help="Drive dataset root with train/val class folders.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    suggest = sub.add_parser(
        "suggest",
        help="Score Drive photos and write a review TSV/PNG. Does not move files.",
    )
    suggest.add_argument(
        "--model",
        type=Path,
        default=Path("models/resnext50_metal_plastic_full_fc_9907.pt"),
        help="Checkpoint to score with.",
    )
    suggest.add_argument(
        "--min-conf",
        type=float,
        default=0.9,
        help="Only propose disagreements at least this confident (default 0.9).",
    )
    suggest.add_argument(
        "--folder",
        choices=("NonPlastic", "Plastic", "both"),
        default="NonPlastic",
        help="Which Drive class to scan (default NonPlastic).",
    )
    suggest.add_argument(
        "--out",
        type=Path,
        default=PROPOSALS_FILE,
        help="TSV of proposed moves.",
    )
    suggest.add_argument(
        "--png",
        type=Path,
        default=PROPOSALS_PNG,
        help="Contact sheet of proposed moves.",
    )
    suggest.add_argument(
        "--max-grid",
        type=int,
        default=48,
        help="Max images on the PNG (default 48).",
    )

    apply_cmd = sub.add_parser(
        "apply",
        help="Move listed files to the other class, same split. Dry-run unless --write.",
    )
    apply_cmd.add_argument(
        "--file",
        type=Path,
        default=None,
        help="TSV from suggest, or any TSV with split/src_class/filename/new_class.",
    )
    apply_cmd.add_argument(
        "--known",
        action="store_true",
        help="Include the two confirmed lotion bottles in NonPlastic/val.",
    )
    apply_cmd.add_argument(
        "--write",
        action="store_true",
        help="Actually move files on Drive. Default is dry-run.",
    )

    promote = sub.add_parser(
        "promote-val",
        help="Move a fraction of each Drive val class into train. Dry-run unless --write.",
    )
    promote.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of each val class to move into train (default 0.5).",
    )
    promote.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed so the same photos move if you re-run.",
    )
    promote.add_argument(
        "--write",
        action="store_true",
        help="Actually move files on Drive. Default is dry-run.",
    )
    return parser.parse_args()


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


def load_lock(path: Path):
    locked = {name: set() for name in ("metal", "plastic")}
    if not path.is_file():
        return locked
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        class_name, stem = line.split("\t", 1)
        locked[class_name].add(stem)
    return locked


def write_lock(path: Path, locked):
    lines = []
    for class_name in ("metal", "plastic"):
        for stem in sorted(locked[class_name]):
            lines.append(f"{class_name}\t{stem}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_proposal_row(row):
    if "filename" in row:
        split = row["split"].strip()
        src_class = row["src_class"].strip()
        filename = row["filename"].strip()
        new_class = row["new_class"].strip()
        reason = (row.get("reason") or "").strip()
    else:
        values = [row[key] for key in row]
        split, src_class, filename, new_class = values[:4]
        reason = values[4] if len(values) > 4 else ""
    if src_class not in CLASS_FOLDERS or new_class not in CLASS_FOLDERS:
        raise SystemExit(f"Bad class in row: {row}")
    if split not in SPLITS:
        raise SystemExit(f"Bad split in row: {row}")
    if src_class == new_class:
        return None
    return split, src_class, filename, new_class, reason


def collect_apply_rows(args):
    rows = []
    if args.known:
        rows.extend(KNOWN_FLIPS)
    if args.file is not None:
        if not args.file.is_file():
            raise SystemExit(f"Missing TSV: {args.file}")
        with args.file.open(encoding="utf-8", newline="") as handle:
            sample = handle.read(2048)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
            has_header = csv.Sniffer().has_header(sample)
            if has_header:
                reader = csv.DictReader(handle, dialect=dialect)
                for raw in reader:
                    parsed = parse_proposal_row(raw)
                    if parsed is not None:
                        rows.append(parsed)
            else:
                reader = csv.reader(handle, dialect=dialect)
                for raw in reader:
                    if not raw or raw[0].startswith("#"):
                        continue
                    parsed = parse_proposal_row(
                        {
                            "split": raw[0],
                            "src_class": raw[1],
                            "filename": raw[2],
                            "new_class": raw[3],
                            "reason": raw[4] if len(raw) > 4 else "",
                        }
                    )
                    if parsed is not None:
                        rows.append(parsed)
    if not rows:
        raise SystemExit("Pass --known and/or --file TSV.")
    return rows


def move_class_file(root, split, src_class, filename, new_class, write):
    source = root / split / src_class / filename
    destination = root / split / new_class / filename
    if not source.is_file():
        raise SystemExit(f"Missing {source}")
    if destination.exists():
        raise SystemExit(f"Already exists: {destination}")
    print(
        f"  {split}/{src_class}/{filename} -> {split}/{new_class}/{filename}",
        flush=True,
    )
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return source.stem, src_class, new_class


def apply_moves(args):
    root = args.root
    rows = collect_apply_rows(args)
    print(
        f"{'Writing' if args.write else 'Dry-run'} {len(rows)} label move(s) under {root}",
        flush=True,
    )
    moved = []
    for split, src_class, filename, new_class, reason in rows:
        extra = f"  ({reason})" if reason else ""
        if extra:
            print(extra, flush=True)
        moved.append(
            move_class_file(
                root, split, src_class, filename, new_class, args.write
            )
        )

    lock_path = LOCKED_TEST_FILE
    locked = load_lock(lock_path)
    lock_changed = False
    for stem, src_class, new_class in moved:
        old_label = FOLDER_TO_LABEL[src_class]
        new_label = FOLDER_TO_LABEL[new_class]
        if stem in locked[old_label]:
            locked[old_label].remove(stem)
            locked[new_label].add(stem)
            lock_changed = True
            print(
                f"  locked test stem {stem}: {old_label} -> {new_label}",
                flush=True,
            )
    if lock_changed and args.write:
        write_lock(lock_path, locked)
        print(f"Updated {lock_path}", flush=True)
    if not args.write:
        print("Re-run with --write to move the files on Drive.", flush=True)


def promote_val(args):
    root = args.root
    if not 0 < args.fraction < 1:
        raise SystemExit("--fraction must be between 0 and 1.")

    locked_val = load_lock(LOCKED_VAL_FILE)
    has_lock = any(locked_val.values())
    rng = random.Random(args.seed)
    planned = []
    remaining_lock = {name: set() for name in ("metal", "plastic")}

    for folder in CLASS_FOLDERS:
        paths = image_files(root / "val" / folder)
        label = FOLDER_TO_LABEL[folder]
        if has_lock:
            keepers = [
                path for path in paths if path.stem in locked_val[label]
            ]
            movable = [
                path for path in paths if path.stem not in locked_val[label]
            ]
        else:
            keepers = []
            movable = list(paths)
        rng.shuffle(movable)
        n_move = int(len(paths) * args.fraction)
        n_move = min(n_move, len(movable))
        to_train = sorted(movable[:n_move], key=lambda path: path.name)
        stay = keepers + movable[n_move:]
        remaining_lock[label].update(path.stem for path in stay)
        print(
            f"  val/{folder}: {len(paths)} now, move {len(to_train)} to train, "
            f"keep {len(stay)} in val",
            flush=True,
        )
        skipped = []
        keep_from_collisions = []
        for path in to_train:
            destination = root / "train" / folder / path.name
            if destination.exists():
                skipped.append(path)
                keep_from_collisions.append(path)
                continue
            planned.append((folder, path))
        remaining_lock[label].update(path.stem for path in keep_from_collisions)
        if skipped:
            print(
                f"    skip {len(skipped)} already in train/{folder} "
                "(leave in val)",
                flush=True,
            )

    print(
        f"{'Writing' if args.write else 'Dry-run'} {len(planned)} val -> train move(s)",
        flush=True,
    )
    shown = planned if len(planned) <= 20 else planned[:10]
    for folder, path in shown:
        print(f"  val/{folder}/{path.name} -> train/{folder}/{path.name}", flush=True)
    if len(planned) > 20:
        print(f"  ... {len(planned) - 10} more", flush=True)
    if args.write:
        for folder, path in planned:
            shutil.move(str(path), str(root / "train" / folder / path.name))

    if args.write:
        write_lock(LOCKED_VAL_FILE, remaining_lock)
        print(f"Locked remaining val stems in {LOCKED_VAL_FILE}", flush=True)
        print(
            "Rebuild the zip next: python resize_stage1.py",
            flush=True,
        )
    else:
        print("Re-run with --write to move the files on Drive.", flush=True)


def load_model(model_path):
    import torch
    from torch import nn
    from torchvision.models import resnext50_32x4d

    from preprocess import inference_transform, load_rgb

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    candidates = [
        model_path,
        Path("models/resnext50_metal_plastic.pt"),
    ]
    checkpoint_path = next((path for path in candidates if path.is_file()), model_path)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint: {model_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    model = resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device, idx_to_class, inference_transform(), load_rgb


def suggest(args):
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from preprocess import tensor_to_display

    root = args.root
    folders = CLASS_FOLDERS if args.folder == "both" else (args.folder,)
    paths = []
    for split in SPLITS:
        for folder in folders:
            paths.extend(image_files(root / split / folder))
    if not paths:
        raise SystemExit(f"No images under {root}")

    model, device, idx_to_class, preprocess, load_rgb = load_model(args.model)
    rows = []
    print(
        f"Scoring {len(paths)} photos with min-conf {args.min_conf:.0%}...",
        flush=True,
    )
    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            image = load_rgb(path)
            tensor = preprocess(image).to(device)
            probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
            ranked = sorted(
                (
                    (idx_to_class[i], probabilities[i].item())
                    for i in range(probabilities.numel())
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            predicted, confidence = ranked[0]
            true_class = FOLDER_TO_LABEL[path.parent.name]
            if index == 1 or index == len(paths) or index % 100 == 0:
                print(f"  {index}/{len(paths)}", flush=True)
            if predicted == true_class or confidence < args.min_conf:
                continue
            relative = path.relative_to(root)
            split, src_class = relative.parts[:2]
            new_class = LABEL_TO_FOLDER[predicted]
            rows.append(
                {
                    "split": split,
                    "src_class": src_class,
                    "filename": path.name,
                    "new_class": new_class,
                    "confidence": f"{confidence:.4f}",
                    "reason": f"model {true_class}->{predicted}",
                    "tensor": tensor.detach().cpu(),
                    "ranked": ranked,
                    "true": true_class,
                }
            )

    rows.sort(key=lambda item: float(item["confidence"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "src_class",
                "filename",
                "new_class",
                "confidence",
                "reason",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row[key] for key in writer.fieldnames}
            )
    print(
        f"Wrote {len(rows)} proposals to {args.out}. "
        "Delete foil trays and other true metals before apply.",
        flush=True,
    )

    grid = rows[: args.max_grid]
    if not grid:
        return
    columns = min(4, len(grid))
    plot_rows = int(np.ceil(len(grid) / columns))
    figure, axes = plt.subplots(
        plot_rows,
        columns,
        figsize=(2.6 * columns, 2.6 * plot_rows),
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, row in zip(axes, grid):
        title = (
            f"{row['true']}->{FOLDER_TO_LABEL[row['new_class']]} "
            f"{float(row['confidence']):.0%}\n{row['filename'][:28]}"
        )
        axis.imshow(
            np.asarray(tensor_to_display(row["tensor"])),
            origin="upper",
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=7)
        axis.set_axis_off()
    for axis in axes[len(grid) :]:
        axis.set_axis_off()
    figure.suptitle(
        "Proposed label flips (review; foil trays are usually correct metal)",
        fontsize=11,
    )
    args.png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png, dpi=140)
    plt.close(figure)
    print(f"Wrote {args.png}", flush=True)


def main():
    args = parse_args()
    if args.command != "suggest" and not args.root.is_dir():
        raise SystemExit(f"Missing Drive dataset: {args.root}")
    if args.command == "suggest":
        suggest(args)
    elif args.command == "apply":
        apply_moves(args)
    else:
        promote_val(args)


if __name__ == "__main__":
    main()
