from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms


INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_ALIASES = {
    "nonplastic": "metal",
    "metal": "metal",
    "plastic": "plastic",
}


def class_key(name):
    return "".join(
        character
        for character in name.lower()
        if character.isalnum()
    )


def mapped_class(name):
    return CLASS_ALIASES.get(class_key(name))


def true_class_from_path(path):
    for part in reversed(Path(path).parts):
        mapped = mapped_class(part)
        if mapped is not None:
            return mapped
    return None


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


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
    """Same 224×224 bilinear scale as production, plus light aug."""
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
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
