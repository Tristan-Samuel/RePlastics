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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show predictions on random test images."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root with train/validation/test "
            "class folders."
        ),
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

DISPLAY_COUNT = 9
CELL_SIZE = 360

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def label_font(size=22):
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
# Random test images
# ---------------------------------------------------------

assert_split_layout(args.data_dir)
test_dir = split_dirs_for(args.data_dir)["test"]

weights = ResNeXt50_32X4D_Weights.DEFAULT
preprocess = weights.transforms()

test_paths = [
    path
    for class_name in EXPECTED_CLASSES
    for path in sorted((test_dir / class_name).iterdir())
    if path.is_file()
    and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

if not test_paths:
    raise FileNotFoundError(
        f"No test images found in {test_dir}."
    )

sample_count = min(DISPLAY_COUNT, len(test_paths))
sample_paths = random.sample(test_paths, sample_count)


# ---------------------------------------------------------
# Predict and popup
# ---------------------------------------------------------

rows = 3
columns = 3

figure, axes = plt.subplots(
    rows,
    columns,
    figsize=(10, 10),
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

        confidence, predicted_index = (
            probabilities.max(dim=0)
        )

        predicted_class = idx_to_class[
            predicted_index.item()
        ]

        overlay_text = (
            f"{predicted_class}\n"
            f"{confidence.item():.0%}"
        )
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

        print(
            f"{image_path.name}: "
            f"{predicted_class} "
            f"({confidence.item():.2%})"
        )

for axis in axes[sample_count:]:
    axis.set_axis_off()

figure.suptitle("Random test predictions", fontsize=16)
plt.show()
