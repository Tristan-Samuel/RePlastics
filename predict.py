import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw, ImageFont

from checkpoints import lab_class, read_checkpoint
from decide import (
    UNCERTAIN,
    decide,
    frozen_rule_applies,
    scaled_probabilities,
    scoring_temperature,
    scoring_threshold,
)
from preprocess import (
    inference_transform,
    load_rgb,
    readable_image_paths,
    resin_from_path,
    tensor_to_display,
    true_class_from_path,
)
from trashnet import (
    EXPECTED_CLASSES,
    assert_split_layout,
    split_dirs_for,
)


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
LEGACY_MODEL_PATH = Path("resnext50_metal_plastic.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VALIDATION_MISTAKES_PNG = Path("misclassified_validation.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score 224×224 model inputs and overlay "
            "metal/plastic percentages."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root with train/validation/test "
            "class folders. Used when --images is omitted."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Photo files or folders to score. "
            "Folders are searched recursively. "
            "Skips the test-split layout."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=9,
        help="How many photos to show (default: 9). Ignored with --mistakes-out.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score every image instead of a random sample.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the labeled grid PNG.",
    )
    parser.add_argument(
        "--mistakes-out",
        type=Path,
        default=None,
        help=(
            "Score every labeled image and write only the "
            "errors to this PNG. Never writes "
            "misclassified_validation.png."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save/print only; do not open a window.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Defaults to "
            "models/resnext50_metal_plastic.pt, then "
            "resnext50_metal_plastic.pt in the current "
            "directory."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Hold the frame as uncertain when the winning "
            "score is below this value. Every checkpoint "
            "defaults to the threshold and temperature in "
            "decide.py. This flag overrides the threshold "
            "for one run."
        ),
    )
    return parser.parse_args()


args = parse_args()

if args.mistakes_out is not None:
    if args.mistakes_out.resolve() == VALIDATION_MISTAKES_PNG.resolve():
        raise SystemExit(
            "Refusing to overwrite misclassified_validation.png. "
            "Pass a different --mistakes-out path."
        )

if args.model is not None:
    MODEL_PATH = args.model
elif DEFAULT_MODEL_PATH.exists():
    MODEL_PATH = DEFAULT_MODEL_PATH
elif LEGACY_MODEL_PATH.exists():
    MODEL_PATH = LEGACY_MODEL_PATH
else:
    MODEL_PATH = DEFAULT_MODEL_PATH

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def label_font(size=16):
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def paint_overlay(canvas, text):
    canvas = canvas.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = label_font()
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=2,
    )
    text_width = right - left
    text_height = bottom - top
    pad = 6
    x = 4
    y = 4
    draw.rounded_rectangle(
        [x, y, x + text_width + 2 * pad, y + text_height + 2 * pad],
        radius=6,
        fill=(0, 0, 0, 180),
    )
    draw.multiline_text(
        (x + pad - left, y + pad - top),
        text,
        font=font,
        fill="white",
        spacing=2,
    )
    return canvas.convert("RGB")


def collect_image_paths(locations):
    paths = []
    for location in locations:
        if location.is_file():
            if location.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(location)
            continue
        if not location.is_dir():
            raise FileNotFoundError(f"No such file or folder: {location}")
        for path in sorted(location.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
    return paths


def overlay_text(ranked, true_class=None, resin=None, decision=None):
    lines = [f"{name}  {score:.0%}" for name, score in ranked]
    if decision == UNCERTAIN:
        lines.insert(0, "uncertain")
    if true_class is not None:
        true_line = f"true {true_class}"
        if resin:
            true_line += f" ({resin})"
        lines.append(true_line)
    elif resin:
        lines.append(resin)
    return "\n".join(lines)


def save_grid(cells, title, out_path, show):
    sample_count = len(cells)
    if sample_count == 0:
        print("No images to plot.")
        return
    columns = min(4, sample_count)
    rows = int(np.ceil(sample_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.6 * columns, 2.6 * rows),
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, cell in zip(axes, cells):
        axis.imshow(np.asarray(cell), origin="upper", interpolation="nearest")
        axis.set_axis_off()
        axis.set_aspect("equal")
    for axis in axes[sample_count:]:
        axis.set_axis_off()
    figure.suptitle(title, fontsize=12)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=160)
        print(f"Wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model, class_to_idx, idx_to_class = read_checkpoint(MODEL_PATH, device)

preprocess = inference_transform()
if frozen_rule_applies(MODEL_PATH):
    temperature = scoring_temperature()
    threshold = scoring_threshold(args.threshold)
else:
    temperature = 1.0
    threshold = None if args.threshold is None else float(args.threshold)
    print(
        "No frozen hold rule for this checkpoint. "
        "Scores are the raw softmax."
    )
print(f"Temperature {temperature:.6f}")
if threshold is None:
    print("Uncertain threshold is unset; the winning class is used.")
else:
    print(f"Uncertain when the winning score is below {threshold:.2f}")

if args.images:
    image_paths = collect_image_paths(args.images)
    source_label = "selected photos"
else:
    assert_split_layout(args.data_dir)
    test_dir = split_dirs_for(args.data_dir)["test"]
    image_paths = [
        path
        for class_name in EXPECTED_CLASSES
        for path in sorted((test_dir / class_name).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    source_label = f"{test_dir} (test split)"

image_paths = readable_image_paths(image_paths)
if not image_paths:
    raise FileNotFoundError(
        "No images found. Pass --images or a dataset with a test split."
    )

report_mistakes = args.mistakes_out is not None
if report_mistakes or args.all:
    sample_paths = image_paths
else:
    sample_count = min(max(1, args.count), len(image_paths))
    if args.images and len(image_paths) <= sample_count:
        sample_paths = image_paths
    else:
        sample_paths = random.sample(image_paths, sample_count)

mistakes = []
held = []
cells = []

skipped = []
with torch.no_grad():
    for image_path in sample_paths:
        try:
            image = load_rgb(image_path)
        except OSError as error:
            print(f"skip {image_path}: {error}")
            skipped.append(image_path)
            continue
        image_tensor = preprocess(image).to(device)
        logits = model(image_tensor.unsqueeze(0))
        probabilities = scaled_probabilities(logits, temperature)[0]
        ranked = sorted(
            (
                (idx_to_class[index], probabilities[index].item())
                for index in range(probabilities.numel())
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        decision, confidence = decide(probabilities, threshold, idx_to_class)
        predicted_class = ranked[0][0]
        true_class = true_class_from_path(image_path)
        resin = resin_from_path(image_path)
        detail = " | ".join(
            f"{name} {score:.2%}" for name, score in ranked
        )
        extras = []
        if decision == UNCERTAIN:
            extras.append("uncertain")
        if true_class is not None:
            extras.append(f"true {true_class}")
        if resin:
            extras.append(resin)
        if extras:
            print(f"{image_path}: {detail} | {' | '.join(extras)}")
        else:
            print(f"{image_path}: {detail}")

        is_mistake = (
            true_class is not None
            and lab_class(predicted_class) != true_class
            and decision != UNCERTAIN
        )
        record = {
            "path": image_path,
            "true": true_class,
            "resin": resin,
            "predicted": predicted_class,
            "decision": decision,
            "confidence": confidence,
            "ranked": ranked,
            "tensor": image_tensor.detach().cpu(),
        }
        if is_mistake:
            mistakes.append(record)
        if true_class is not None and decision == UNCERTAIN:
            held.append(record)

        if not report_mistakes:
            cells.append(
                paint_overlay(
                    tensor_to_display(image_tensor.cpu()),
                    overlay_text(ranked, true_class, resin, decision),
                )
            )

if skipped:
    print(f"Skipped {len(skipped)} unreadable image(s) during scoring.")

if report_mistakes:
    scored_paths = [path for path in sample_paths if path not in skipped]
    unlabeled = sum(
        1 for path in scored_paths if true_class_from_path(path) is None
    )
    if unlabeled:
        print(
            f"{unlabeled} images had no metal/plastic folder in the path "
            "and were skipped for the mistake list."
        )
    mistakes.sort(key=lambda item: item["confidence"], reverse=True)
    txt_path = args.mistakes_out.with_suffix(".txt")
    args.mistakes_out.parent.mkdir(parents=True, exist_ok=True)
    labeled_by_resin = Counter()
    for path in scored_paths:
        if true_class_from_path(path) is None:
            continue
        labeled_by_resin[resin_from_path(path) or "unknown"] += 1
    wrong_by_resin = Counter(
        mistake["resin"] or "unknown" for mistake in mistakes
    )
    with txt_path.open("w", encoding="utf-8") as report:
        resin_keys = [
            key
            for key in ("HDPE", "PET", "PP", "OTHERS", "unknown")
            if labeled_by_resin.get(key)
        ]
        if resin_keys:
            report.write("# mistakes by resin\n")
            print("Mistakes by resin:")
            for key in resin_keys:
                total = labeled_by_resin[key]
                wrong = wrong_by_resin.get(key, 0)
                line = f"{key}: {wrong}/{total} ({wrong / total:.1%})"
                print(f"  {line}")
                report.write(line + "\n")
            report.write("\n")
        for mistake in mistakes:
            resin_note = (
                f" | resin: {mistake['resin']}" if mistake["resin"] else ""
            )
            line = (
                f"{mistake['path']} | true: {mistake['true']}{resin_note} | "
                f"pred: {mistake['predicted']} "
                f"({mistake['confidence']:.2%})"
            )
            print(line)
            report.write(line + "\n")
        if held:
            report.write("\n# held\n")
            print("Held:")
            for item in sorted(held, key=lambda row: row["confidence"]):
                resin_note = (
                    f" | resin: {item['resin']}" if item["resin"] else ""
                )
                line = (
                    f"{item['path']} | true: {item['true']}{resin_note} | "
                    f"argmax: {item['predicted']} "
                    f"({item['confidence']:.2%}) | uncertain"
                )
                print(line)
                report.write(line + "\n")
    labeled_count = len(scored_paths) - unlabeled
    print(f"{len(mistakes)} mistakes / {labeled_count} labeled")
    if threshold is None:
        print("No uncertain threshold; the winning class is used.")
    else:
        print(f"{len(held)} held / {labeled_count} labeled")
    print(f"Wrote {txt_path}")
    cells = [
        paint_overlay(
            tensor_to_display(mistake["tensor"]),
            overlay_text(
                mistake["ranked"],
                mistake["true"],
                mistake["resin"],
            ),
        )
        for mistake in mistakes
    ]
    save_grid(
        cells,
        f"Mistakes on the 224×224 model input\n{source_label}",
        args.mistakes_out,
        show=not args.no_show,
    )
else:
    scored_paths = [path for path in sample_paths if path not in skipped]
    labeled_count = sum(
        1
        for path in scored_paths
        if true_class_from_path(path) is not None
    )
    if threshold is None:
        print("No uncertain threshold; the winning class is used.")
    else:
        print(f"{len(held)} held / {labeled_count} labeled")
    save_grid(
        cells,
        f"224×224 model input\n{source_label}",
        args.out,
        show=not args.no_show,
    )
