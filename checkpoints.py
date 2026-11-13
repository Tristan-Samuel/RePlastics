"""Load a ResNeXt checkpoint, including the official weights-only file.

A training run stores ``model_state_dict`` and ``class_to_idx``.
``models/replastics_best.pt`` is the state dict itself. The sidecar
``models/replastics_best.metadata.json`` names the head HDPE, PET, PP.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnext50_32x4d


RESIN_HEAD = ("HDPE", "PET", "PP")
BELT_HEAD = ("metal", "plastic")


def lab_class(name):
    """Metal or plastic, which is how this lab reads a resin call."""
    if name in RESIN_HEAD:
        return "plastic"
    return name


def resin_head(idx_to_class):
    names = [idx_to_class[index] for index in range(len(idx_to_class))]
    return tuple(names) == RESIN_HEAD


def is_raw_state_dict(blob):
    if not isinstance(blob, dict):
        return False
    if "model_state_dict" in blob or "class_to_idx" in blob:
        return False
    weight = blob.get("fc.weight")
    return hasattr(weight, "ndim") and weight.ndim == 2


def metadata_path(model_path):
    path = Path(model_path)
    candidates = (
        path.with_suffix(".metadata.json"),
        path.parent / "model_metadata.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_metadata(model_path):
    path = metadata_path(model_path)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def class_names_for(blob, metadata):
    listed = metadata.get("class_names") if isinstance(metadata, dict) else None
    if (
        isinstance(listed, list)
        and listed
        and all(isinstance(name, str) and name.strip() for name in listed)
    ):
        return [name.strip() for name in listed]
    if not is_raw_state_dict(blob):
        stored = blob.get("class_to_idx") if isinstance(blob, dict) else None
        if isinstance(stored, dict) and stored:
            return [
                str(name)
                for name, _index in sorted(
                    stored.items(), key=lambda item: int(item[1])
                )
            ]
        return None
    width = int(blob["fc.weight"].shape[0])
    if width == len(RESIN_HEAD):
        return list(RESIN_HEAD)
    if width == len(BELT_HEAD):
        return list(BELT_HEAD)
    return None


def read_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location=device, weights_only=False)
    metadata = read_metadata(path)
    if is_raw_state_dict(blob):
        names = class_names_for(blob, metadata)
        width = int(blob["fc.weight"].shape[0])
        if names is None or len(names) != width:
            raise ValueError(
                f"{path.name} is weights only and its class list does not "
                f"match the {width}-wide head."
            )
        class_to_idx = {name: index for index, name in enumerate(names)}
        state = blob
    else:
        if not isinstance(blob, dict):
            raise ValueError(f"{path.name} is not a checkpoint.")
        stored = blob.get("class_to_idx")
        state = blob.get("model_state_dict")
        if not isinstance(stored, dict) or not stored or state is None:
            raise ValueError(
                f"{path.name} has no class_to_idx and model_state_dict."
            )
        class_to_idx = {
            str(name): int(index) for name, index in stored.items()
        }
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    model = resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, class_to_idx, idx_to_class
