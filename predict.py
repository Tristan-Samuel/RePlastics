import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models import (
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
)

from trashnet import (
    EXPECTED_CLASSES,
    assert_split_layout,
    split_dirs_for,
)


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
LEGACY_MODEL_PATH = Path("resnext50_metal_plastic.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score photos and overlay metal/plastic "
            "percentages on the images."
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
        help="How many photos to show (default: 9).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the labeled grid PNG.",
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

if args.model is not None:
    MODEL_PATH = args.model
elif DEFAULT_MODEL_PATH.exists():
    MODEL_PATH = DEFAULT_MODEL_PATH
elif LEGACY_MODEL_PATH.exists():
    MODEL_PATH = LEGACY_MODEL_PATH
else:
    MODEL_PATH = DEFAULT_MODEL_PATH

DISPLAY_COUNT = max(1, args.count)
CELL_SIZE = 360

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def label_font(size=20):
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def square_cell(image):
    """Keep the photo upright and unstretched in a square tile."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fitted = ImageOps.contain(
        image,
        (CELL_SIZE, CELL_SIZE),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (20, 20, 20))
    canvas.paste(
        fitted,
        (
            (CELL_SIZE - fitted.width) // 2,
            (CELL_SIZE - fitted.height) // 2,
        ),
    )
    return canvas


def paint_overlay(canvas, text):
    """Draw the prediction on the photo so it cannot drift off the image."""
    canvas = canvas.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = label_font()
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=4,
    )
    text_width = right - left
    text_height = bottom - top
    pad = 8
    x = 12
    y = 12
    draw.rounded_rectangle(
        [x, y, x + text_width + 2 * pad, y + text_height + 2 * pad],
        radius=8,
        fill=(0, 0, 0, 180),
    )
    draw.multiline_text(
        (x + pad - left, y + pad - top),
        text,
        font=font,
        fill="white",
        spacing=4,
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


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Re-create model architecture
# ---------------------------------------------------------

model = resnext50_32x4d(
    weights=None
)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ---------------------------------------------------------
# Images to score
# ---------------------------------------------------------

weights = ResNeXt50_32X4D_Weights.DEFAULT
preprocess = weights.transforms()

if args.images:
    test_paths = collect_image_paths(args.images)
    source_label = "selected photos"
else:
    assert_split_layout(args.data_dir)
    test_dir = split_dirs_for(args.data_dir)["test"]
    test_paths = [
        path
        for class_name in EXPECTED_CLASSES
        for path in sorted((test_dir / class_name).iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    source_label = f"{test_dir} (random test split)"

if not test_paths:
    raise FileNotFoundError(
        "No images found. Pass --images or a dataset with a test split."
    )

sample_count = min(DISPLAY_COUNT, len(test_paths))
if args.images and len(test_paths) <= DISPLAY_COUNT:
    sample_paths = test_paths
    sample_count = len(sample_paths)
else:
    sample_paths = random.sample(test_paths, sample_count)


# ---------------------------------------------------------
# Predict and popup
# ---------------------------------------------------------

rows = int(np.ceil(sample_count / 3))
columns = min(3, sample_count)

figure, axes = plt.subplots(
    rows,
    columns,
    figsize=(3.4 * columns, 3.4 * rows),
    layout="constrained",
)

axes = np.atleast_1d(axes).ravel()

with torch.no_grad():

    for axis, image_path in zip(axes, sample_paths):

        image = Image.open(image_path)

        image_tensor = preprocess(image.convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        ranked = sorted(
            (
                (idx_to_class[index], probabilities[index].item())
                for index in range(probabilities.numel())
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        overlay_lines = [
            f"{name}  {score:.0%}"
            for name, score in ranked
        ]
        overlay_text = "\n".join(overlay_lines)
        cell = paint_overlay(
            square_cell(image),
            overlay_text,
        )

        axis.imshow(
            np.asarray(cell),
            origin="upper",
            interpolation="bilinear",
        )
        axis.set_axis_off()
        axis.set_aspect("equal")

        detail = " | ".join(
            f"{name} {score:.2%}"
            for name, score in ranked
        )
        print(f"{image_path}: {detail}")

for axis in axes[sample_count:]:
    axis.set_axis_off()

figure.suptitle(
    f"Predictions with class percentages\n{source_label}",
    fontsize=14,
)

if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")

if not args.no_show:
    plt.show()
else:
    plt.close(figure)
