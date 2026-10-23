from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms


INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_ALIASES = {
    "nonplastic": "metal",
    "metal": "metal",
    "plastic": "plastic",
    "hdpe": "plastic",
    "pet": "plastic",
    "pp": "plastic",
    "others": "metal",
    "other": "metal",
}

RESIN_ALIASES = {
    "hdpe": "HDPE",
    "pet": "PET",
    "pp": "PP",
    "others": "OTHERS",
    "other": "OTHERS",
}

RESIN_TO_CLASS = {
    "HDPE": "plastic",
    "PET": "plastic",
    "PP": "plastic",
    "OTHERS": "metal",
}

RESIN_TAGS = ("OTHERS", "HDPE", "PET", "PP")


def class_key(name):
    return "".join(
        character
        for character in name.lower()
        if character.isalnum()
    )


def mapped_class(name):
    return CLASS_ALIASES.get(class_key(name))


def resin_from_path(path):
    """HDPE / PET / PP / OTHERS from a `_RESIN` filename tag or folder."""
    path = Path(path)
    stem = path.stem.upper()
    for resin in RESIN_TAGS:
        if stem.endswith("_" + resin) or stem.endswith("-" + resin):
            return resin
    for part in reversed(path.parts[:-1]):
        mapped = RESIN_ALIASES.get(class_key(part))
        if mapped is not None:
            return mapped
    return None


def true_class_from_path(path):
    resin = resin_from_path(path)
    if resin is not None:
        return RESIN_TO_CLASS[resin]
    for part in reversed(Path(path).parts):
        mapped = mapped_class(part)
        if mapped is not None:
            return mapped
    return None


def load_rgb(path):
    path = Path(path)
    if path.stat().st_size == 0:
        raise UnidentifiedImageError("empty file")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def readable_image_paths(paths):
    """Drop empty or corrupt files so a camera dump can still be scored."""
    kept = []
    skipped = 0
    for path in paths:
        path = Path(path)
        try:
            if path.stat().st_size == 0:
                raise UnidentifiedImageError("empty file")
            with Image.open(path):
                pass
        except OSError as error:
            print(f"skip {path}: {error}", flush=True)
            skipped += 1
            continue
        kept.append(path)
    if skipped:
        print(f"Skipped {skipped} unreadable image(s).", flush=True)
    return kept


def inference_transform():
    """Scale to 224×224 the same way production should, then ImageNet-normalize."""
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def train_transform():
    """Zoom into the object at train time, still ending at 224×224.

    ImageNet and the current 99% checkpoint used a random resized crop.
    Scale stays at least 60% of the photo so a bottle on the belt is not
    cropped away. Eval and production still use the full-frame 224×224
    resize in inference_transform.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.6, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def tensor_to_display(tensor):
    """Undo ImageNet normalize so the grid shows the 224×224 the model scored."""
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    image = (tensor * std + mean).clamp(0, 1)
    array = (
        image.detach().cpu().permute(1, 2, 0).numpy() * 255
    ).astype("uint8")
    return Image.fromarray(array)
