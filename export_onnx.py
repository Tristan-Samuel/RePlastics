"""Export the belt ResNeXt checkpoint to a fixed-shape ONNX graph.

The graph is ``model.forward`` only: one ImageNet-normalized 224×224
image in, two logits out. Resize, temperature, and the 0.97 hold stay
in the camera process. See docs/jetson.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnext50_32x4d

from decide import frozen_rule_applies, scoring_temperature, scoring_threshold
from preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    inference_transform,
    load_rgb,
)
from trashnet import EXPECTED_CLASSES, split_dirs_for


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
DEFAULT_ONNX_PATH = Path("models/resnext50_metal_plastic.onnx")
DEFAULT_DATA_DIR = Path("data_belt_stage1")
OPSET = 17
INPUT_NAME = "image"
OUTPUT_NAME = "logits"
LOGIT_ATOL = 1e-4
CHECK_PER_CLASS = 2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the 2-class ResNeXt checkpoint to ONNX "
            "for the Jetson TensorRT build."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Checkpoint path. Default: "
            "models/resnext50_metal_plastic.pt."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=(
            "ONNX path. The JSON sidecar is this path "
            "plus .json."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Score a few JPEGs with PyTorch and ONNX Runtime. "
            "Stop if logits differ by more than 1e-4."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Dataset root used by --check when --images "
            "is omitted. Reads test/metal and test/plastic."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help="Photos to score for --check. Folders are not searched.",
    )
    return parser.parse_args()


def sidecar_path(onnx_path):
    return onnx_path.parent / (onnx_path.name + ".json")


def load_checkpoint(model_path):
    if not model_path.is_file():
        raise SystemExit(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    class_to_idx = checkpoint.get("class_to_idx")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise SystemExit(f"{model_path} has no class_to_idx.")
    if state_dict is None:
        raise SystemExit(f"{model_path} has no model_state_dict.")
    class_to_idx = {
        str(name): int(index) for name, index in class_to_idx.items()
    }
    if len(class_to_idx) != 2:
        raise SystemExit(
            f"{model_path} has {len(class_to_idx)} classes. "
            "This export is the 2-class belt head."
        )
    model = resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_to_idx


def scoring_contract(model_path):
    """Temperature and threshold that travel with this checkpoint."""
    if frozen_rule_applies(model_path):
        return scoring_temperature(), scoring_threshold(None)
    return 1.0, None


def build_sidecar(model_path, class_to_idx):
    temperature, threshold = scoring_contract(model_path)
    return {
        "checkpoint": model_path.resolve().name,
        "input": {
            "name": INPUT_NAME,
            "dtype": "float32",
            "layout": "NCHW",
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        },
        "output": {
            "name": OUTPUT_NAME,
            "dtype": "float32",
            "shape": [1, len(class_to_idx)],
        },
        "class_to_idx": class_to_idx,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "temperature": temperature,
        "threshold": threshold,
    }


def export_graph(model, onnx_path):
    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Export needs the onnx package: "
            "pip install onnx onnxruntime"
        ) from error
    dummy = torch.zeros(
        1,
        3,
        INPUT_SIZE,
        INPUT_SIZE,
        dtype=torch.float32,
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # CPU graph, static batch 1. JetPack TensorRT accepts this opset
    # from the legacy exporter. The dynamo exporter is a different graph.
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamo=False,
    )


def check_images(data_dir, images):
    if images:
        missing = [path for path in images if not path.is_file()]
        if missing:
            raise SystemExit(
                "Missing image(s): "
                + ", ".join(str(path) for path in missing)
            )
        return list(images)
    chosen = []
    for class_name in EXPECTED_CLASSES:
        folder = split_dirs_for(data_dir)["test"] / class_name
        if not folder.is_dir():
            raise SystemExit(
                f"Missing {folder}. Pass --images to check "
                "specific photos."
            )
        found = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(found) < CHECK_PER_CLASS:
            raise SystemExit(
                f"{folder} has {len(found)} image(s); "
                f"the logit check wants {CHECK_PER_CLASS}."
            )
        chosen.extend(found[:CHECK_PER_CLASS])
    return chosen


def compare_logits(model, onnx_path, image_paths):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise SystemExit(
            "The logit check needs onnxruntime: "
            "pip install onnx onnxruntime"
        ) from error
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    preprocess = inference_transform()
    worst = 0.0
    checked = 0
    with torch.no_grad():
        for image_path in image_paths:
            try:
                image = load_rgb(image_path)
            except OSError as error:
                print(f"skip {image_path}: {error}")
                continue
            tensor = preprocess(image).unsqueeze(0)
            torch_logits = model(tensor).detach().cpu().numpy()
            onnx_logits = session.run(
                None,
                {INPUT_NAME: tensor.numpy()},
            )[0]
            diff = float(np.max(np.abs(torch_logits - onnx_logits)))
            worst = max(worst, diff)
            checked += 1
            print(f"{image_path}: max abs logit diff {diff:.3e}")
    if checked == 0:
        raise SystemExit("No readable images for the logit check.")
    if worst > LOGIT_ATOL:
        raise SystemExit(
            f"ONNX logits differ from PyTorch by {worst:.3e}, "
            f"above {LOGIT_ATOL:.0e}. "
            "Leave this file off the Jetson."
        )
    print(
        f"Checked {checked} image(s). "
        f"Worst max abs logit diff {worst:.3e}."
    )


def main():
    args = parse_args()
    model, class_to_idx = load_checkpoint(args.model)
    export_graph(model, args.out)
    payload = build_sidecar(args.model, class_to_idx)
    sidecar = sidecar_path(args.out)
    sidecar.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {sidecar}")
    print(f"class_to_idx {class_to_idx}")
    print(
        f"temperature {payload['temperature']:.6f}  "
        f"threshold {payload['threshold']}"
    )
    if args.check:
        compare_logits(
            model,
            args.out,
            check_images(args.data_dir, args.images),
        )


if __name__ == "__main__":
    main()
