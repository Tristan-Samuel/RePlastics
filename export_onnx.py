"""Export the belt ResNeXt checkpoint to a fixed-shape ONNX graph.

The graph is ``model.forward`` only: one ImageNet-normalized 224×224
image in, two logits out. Resize, temperature, and the hold stay
in the camera process. See docs/jetson.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checkpoints import read_checkpoint
from decide import (
    decide,
    frozen_rule_applies,
    scaled_probabilities,
    scoring_temperature,
    scoring_threshold,
)
from preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    inference_transform,
    load_rgb,
)
from trashnet import EXPECTED_CLASSES, split_dirs_for


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
DEFAULT_DATA_DIR = Path("data_belt_stage1")
OPSET = 17
INPUT_NAME = "image"
OUTPUT_NAME = "logits"
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
        default=None,
        help=(
            "ONNX path. Default: the checkpoint path with "
            ".onnx. The JSON sidecar is this path plus .json."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Score the test split with PyTorch and ONNX Runtime. "
            "Print accuracy and stop if a label changes. "
            "With --images, only those photos are compared."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Score an ONNX file already on disk. "
            "Do not write a new graph or JSON."
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
    try:
        model, class_to_idx, _idx_to_class = read_checkpoint(model_path, "cpu")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if len(class_to_idx) != 2:
        raise SystemExit(
            f"{model_path.name} has {len(class_to_idx)} classes. "
            "This export is the 2-class belt head."
        )
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


def test_rows(data_dir):
    """Every labeled test photo, in class order then filename order."""
    rows = []
    for class_name in EXPECTED_CLASSES:
        folder = split_dirs_for(data_dir)["test"] / class_name
        if not folder.is_dir():
            raise SystemExit(
                f"Missing {folder}. The accuracy check reads "
                "test/metal and test/plastic."
            )
        found = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not found:
            raise SystemExit(f"{folder} has no images.")
        rows.extend((path, class_name) for path in found)
    return rows


def classification_scores(true_names, pred_names, class_names):
    """Overall accuracy, balanced accuracy, and per-class correct/total."""
    total = len(true_names)
    correct = sum(
        true == pred for true, pred in zip(true_names, pred_names)
    )
    parts = []
    recalls = []
    for name in class_names:
        class_total = sum(true == name for true in true_names)
        class_correct = sum(
            true == name and pred == name
            for true, pred in zip(true_names, pred_names)
        )
        if class_total == 0:
            continue
        recalls.append(class_correct / class_total)
        parts.append(f"{name} {class_correct}/{class_total}")
    balanced = sum(recalls) / len(recalls) if recalls else 0.0
    accuracy = correct / total if total else 0.0
    return accuracy, balanced, parts


def _show_path(path):
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_scores(title, true_names, pred_names, class_names):
    accuracy, balanced, parts = classification_scores(
        true_names,
        pred_names,
        class_names,
    )
    detail = " | ".join(parts)
    print(
        f"{title}: accuracy {accuracy:.2%} | "
        f"balanced {balanced:.2%}"
        + (f" | {detail}" if detail else ""),
        flush=True,
    )
    return accuracy, balanced


def compare_test_split(model, onnx_path, data_dir, class_to_idx, contract):
    """Score every test photo in PyTorch and in ONNX Runtime.

    Accuracy here is argmax on the class index. The hold is reported
    separately, with the temperature and threshold stored in the JSON.
    """
    missing = [
        name for name in EXPECTED_CLASSES if name not in class_to_idx
    ]
    if missing:
        raise SystemExit(
            f"class_to_idx has no {', '.join(missing)}."
        )
    rows = test_rows(data_dir)
    temperature, threshold = contract
    idx_to_class = {
        index: name for name, index in class_to_idx.items()
    }
    class_names = [
        name
        for name, _index in sorted(
            class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise SystemExit(
            "The accuracy check needs onnxruntime: "
            "pip install onnx onnxruntime"
        ) from error
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    preprocess = inference_transform()
    total = len(rows)
    print(
        f"Scoring {total} test image(s) in {data_dir} "
        "with PyTorch and ONNX Runtime.",
        flush=True,
    )
    if threshold is None:
        print(
            f"Temperature {temperature:.6f}. No hold threshold.",
            flush=True,
        )
    else:
        print(
            f"Temperature {temperature:.6f}. "
            f"Hold when the winning score is under {threshold}.",
            flush=True,
        )
    true_names = []
    torch_preds = []
    onnx_preds = []
    torch_holds = []
    onnx_holds = []
    mismatches = []
    worst = 0.0
    skipped = 0
    with torch.no_grad():
        for image_path, class_name in rows:
            try:
                image = load_rgb(image_path)
            except OSError as error:
                print(f"skip {image_path}: {error}", flush=True)
                skipped += 1
                continue
            tensor = preprocess(image).unsqueeze(0)
            torch_logits = model(tensor).detach().cpu().numpy()
            onnx_logits = session.run(
                None,
                {INPUT_NAME: tensor.numpy()},
            )[0]
            diff = float(np.max(np.abs(torch_logits - onnx_logits)))
            worst = max(worst, diff)
            torch_index = int(np.argmax(torch_logits, axis=-1).reshape(-1)[0])
            onnx_index = int(np.argmax(onnx_logits, axis=-1).reshape(-1)[0])
            torch_name = idx_to_class[torch_index]
            onnx_name = idx_to_class[onnx_index]
            torch_probs = scaled_probabilities(
                torch.from_numpy(torch_logits),
                temperature,
            )[0]
            onnx_probs = scaled_probabilities(
                torch.from_numpy(onnx_logits),
                temperature,
            )[0]
            torch_hold, _torch_score = decide(
                torch_probs,
                threshold,
                idx_to_class,
            )
            onnx_hold, _onnx_score = decide(
                onnx_probs,
                threshold,
                idx_to_class,
            )
            true_names.append(class_name)
            torch_preds.append(torch_name)
            onnx_preds.append(onnx_name)
            torch_holds.append(torch_hold)
            onnx_holds.append(onnx_hold)
            checked = len(true_names)
            if torch_name != onnx_name or torch_hold != onnx_hold:
                mismatches.append(
                    (
                        _show_path(image_path),
                        torch_name,
                        onnx_name,
                        torch_hold,
                        onnx_hold,
                    )
                )
            if checked % 25 == 0 or checked == total - skipped:
                print(
                    f"  test batch {checked}/{total - skipped}",
                    flush=True,
                )
    if skipped:
        print(f"Skipped {skipped} unreadable image(s).", flush=True)
    if not true_names:
        raise SystemExit("No readable test images for the accuracy check.")
    print(
        "Argmax accuracy on this test split "
        "(not the scores stored in the checkpoint):",
        flush=True,
    )
    _print_scores("PyTorch", true_names, torch_preds, class_names)
    _print_scores("ONNX", true_names, onnx_preds, class_names)
    argmax_disagree = sum(
        left != right for left, right in zip(torch_preds, onnx_preds)
    )
    hold_disagree = sum(
        left != right for left, right in zip(torch_holds, onnx_holds)
    )
    torch_uncertain = sum(label == "uncertain" for label in torch_holds)
    onnx_uncertain = sum(label == "uncertain" for label in onnx_holds)
    print(
        f"Argmax disagreements: {argmax_disagree}. "
        f"Hold disagreements: {hold_disagree}. "
        f"Uncertain: PyTorch {torch_uncertain}, ONNX {onnx_uncertain}.",
        flush=True,
    )
    print(
        f"Worst max abs logit diff {worst:.3e}.",
        flush=True,
    )
    shown = mismatches[:8]
    for path, torch_name, onnx_name, torch_hold, onnx_hold in shown:
        print(
            f"disagree {path}: argmax {torch_name} vs {onnx_name}, "
            f"hold {torch_hold} vs {onnx_hold}",
            flush=True,
        )
    hidden = len(mismatches) - len(shown)
    if hidden > 0:
        print(f"and {hidden} more disagreement(s).", flush=True)
    if argmax_disagree or hold_disagree:
        raise SystemExit(
            "ONNX labels do not match PyTorch on this test split. "
            "Leave this file off the Jetson."
        )
    print(
        f"ONNX matches PyTorch on {len(true_names)} test image(s): "
        "same argmax, same hold. "
        f"Worst logit diff {worst:.3e}.",
        flush=True,
    )


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
    print(
        f"Checked {checked} image(s). "
        f"Worst max abs logit diff {worst:.3e}."
    )


def _run_check(args, model, class_to_idx):
    if args.images:
        compare_logits(
            model,
            args.out,
            check_images(args.data_dir, args.images),
        )
        return
    compare_test_split(
        model,
        args.out,
        args.data_dir,
        class_to_idx,
        scoring_contract(args.model),
    )


def main():
    args = parse_args()
    if args.out is None:
        args.out = Path(args.model).with_suffix(".onnx")
    model, class_to_idx = load_checkpoint(args.model)
    if args.check_only:
        if not args.out.is_file():
            raise SystemExit(f"ONNX file not found: {args.out}")
        print(f"Checking {args.out}")
        print(f"class_to_idx {class_to_idx}")
        _run_check(args, model, class_to_idx)
        return
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
        _run_check(args, model, class_to_idx)


if __name__ == "__main__":
    main()
