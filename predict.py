import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from PIL import ImageDraw, ImageFont
from torchvision.models import resnext50_32x4d

from preprocess import (
    inference_transform,
    load_rgb,
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


def overlay_text(ranked, true_class=None):
    lines = [f"{name}  {score:.0%}" for name, score in ranked]
    if true_class is not None:
        lines.append(f"true {true_class}")
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

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {
    index: name
    for name, index in class_to_idx.items()
}

model = resnext50_32x4d(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

preprocess = inference_transform()

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
cells = []

with torch.no_grad():
    for image_path in sample_paths:
        image = load_rgb(image_path)
        image_tensor = preprocess(image).to(device)
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]
        ranked = sorted(
            (
                (idx_to_class[index], probabilities[index].item())
                for index in range(probabilities.numel())
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        predicted_class, confidence = ranked[0]
        true_class = true_class_from_path(image_path)
        detail = " | ".join(
            f"{name} {score:.2%}" for name, score in ranked
        )
        if true_class is not None:
            print(f"{image_path}: {detail} | true {true_class}")
        else:
            print(f"{image_path}: {detail}")

        is_mistake = (
            true_class is not None and predicted_class != true_class
        )
        if is_mistake:
            mistakes.append(
                {
                    "path": image_path,
                    "true": true_class,
                    "predicted": predicted_class,
                    "confidence": confidence,
                    "ranked": ranked,
                    "tensor": image_tensor.detach().cpu(),
                }
            )

        if not report_mistakes:
            cells.append(
                paint_overlay(
                    tensor_to_display(image_tensor.cpu()),
                    overlay_text(ranked, true_class),
                )
            )

if report_mistakes:
    unlabeled = sum(
        1 for path in sample_paths if true_class_from_path(path) is None
    )
    if unlabeled:
        print(
            f"{unlabeled} images had no metal/plastic folder in the path "
            "and were skipped for the mistake list."
        )
    mistakes.sort(key=lambda item: item["confidence"], reverse=True)
    txt_path = args.mistakes_out.with_suffix(".txt")
    args.mistakes_out.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as report:
        for mistake in mistakes:
            line = (
                f"{mistake['path']} | true: {mistake['true']} | "
                f"pred: {mistake['predicted']} "
                f"({mistake['confidence']:.2%})"
            )
            print(line)
            report.write(line + "\n")
    print(
        f"{len(mistakes)} mistakes / {len(sample_paths) - unlabeled} labeled"
    )
    print(f"Wrote {txt_path}")
    cells = [
        paint_overlay(
            tensor_to_display(mistake["tensor"]),
            overlay_text(mistake["ranked"], mistake["true"]),
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
    save_grid(
        cells,
        f"224×224 model input\n{source_label}",
        args.out,
        show=not args.no_show,
    )
