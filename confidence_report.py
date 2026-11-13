"""Freeze temperature and the uncertain threshold on validation, then
score the locked test split once.

The threshold is chosen only from validation. Test numbers are written
after that choice and are not used to change it.
"""

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from checkpoints import read_checkpoint
from decide import scaled_probabilities
from preprocess import (
    inference_transform,
    load_rgb,
    mapped_class,
    readable_image_paths,
    train_transform,
    true_class_from_path,
)
from trashnet import EXPECTED_CLASSES


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
DEFAULT_DATA_DIR = Path("data_belt_stage1")
DEFAULT_OUT = Path("analysis/belt-stage1/confidence")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_HOLD_RATE = 0.05
N_BINS = 10
AUGMENT_PASSES = 5
AUGMENT_SEED = 0
CRUSHED_CAN = "capture_01896_20260929_141207_925981"
BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a temperature on validation, choose the uncertain "
            "threshold, and score the locked test split once."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset root with validation/ and test/ (default: data_belt_stage1).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Folder for plots, logits, and summary.md.",
    )
    parser.add_argument(
        "--rerun-test",
        action="store_true",
        help="Score the locked test split again. Refused when a finished test record exists.",
    )
    return parser.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path, device):
    try:
        model, class_to_idx, idx_to_class = read_checkpoint(path, device)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if set(class_to_idx) != set(EXPECTED_CLASSES):
        names = ", ".join(
            idx_to_class[index] for index in range(len(idx_to_class))
        )
        raise SystemExit(
            f"{Path(path).name} scores {names}. "
            "This report is the metal/plastic head."
        )
    return model, class_to_idx, idx_to_class


def split_paths(data_dir, split):
    root = Path(data_dir) / split
    paths = []
    for class_name in EXPECTED_CLASSES:
        folder = root / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
    paths = readable_image_paths(paths)
    if not paths:
        raise FileNotFoundError(f"No images in {root}")
    return paths


def labels_for(paths, class_to_idx):
    labels = []
    for path in paths:
        folder = mapped_class(path.parent.name)
        true_class = true_class_from_path(path)
        if folder is None or true_class is None:
            raise SystemExit(f"No class for {path}")
        if folder != true_class:
            raise SystemExit(
                f"{path} folder is {folder} but the filename maps to {true_class}"
            )
        labels.append(class_to_idx[true_class])
    return torch.tensor(labels, dtype=torch.long)


def score_paths(model, paths, transform, device):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            batch = torch.stack(
                [transform(load_rgb(path)) for path in batch_paths]
            ).to(device)
            chunks.append(model(batch).cpu())
            done = min(start + BATCH_SIZE, len(paths))
            print(f"  scored {done}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def class_names(indices, idx_to_class):
    return np.array([idx_to_class[int(index)] for index in indices])


def fit_temperature(logits, labels):
    """One scalar T minimizing unweighted NLL of softmax(logits / T)."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_t = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_t],
        lr=0.5,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.detach().exp())
    if not np.isfinite(temperature) or temperature <= 0:
        raise SystemExit(f"Temperature fit failed: {temperature}")
    return temperature


def mean_nll(logits, labels, temperature):
    log_probs = torch.log_softmax(logits.float() / temperature, dim=1)
    picked = log_probs[torch.arange(len(labels)), labels.long()]
    return float(-picked.mean())


def max_confidence(probabilities):
    return probabilities.max(dim=1).values.numpy()


def prediction_names(probabilities, idx_to_class):
    return class_names(probabilities.argmax(dim=1).tolist(), idx_to_class)


def balanced_accuracy(labels, predictions):
    recalls = []
    detail = {}
    for name in EXPECTED_CLASSES:
        mask = labels == name
        total = int(mask.sum())
        if total == 0:
            detail[name] = {"total": 0, "correct": 0, "recall": None}
            continue
        correct = int((predictions[mask] == name).sum())
        recall = correct / total
        detail[name] = {
            "total": total,
            "correct": correct,
            "recall": recall,
        }
        recalls.append(recall)
    balanced = sum(recalls) / len(recalls) if recalls else None
    return balanced, detail


def sweep_thresholds(confidences, labels, predictions):
    wrong = predictions != labels
    rows = []
    n = len(confidences)
    for step in range(50, 100):
        threshold = step / 100
        held = confidences < threshold
        accepted = ~held
        errors_caught = int((wrong & held).sum())
        balanced, detail = balanced_accuracy(
            labels[accepted], predictions[accepted]
        )
        held_count = int(held.sum())
        rows.append(
            {
                "threshold": threshold,
                "held": held_count,
                "held_rate": held_count / n,
                "errors_caught": errors_caught,
                "errors_total": int(wrong.sum()),
                "accepted": int(accepted.sum()),
                "balanced_accuracy": balanced,
                "metal_recall": detail["metal"]["recall"],
                "plastic_recall": detail["plastic"]["recall"],
            }
        )
    return rows


def choose_threshold(rows):
    """Most errors caught while holding at most 5%. Ties take the lower threshold.

    Returns None when no such threshold catches an error.
    """
    eligible = [
        row
        for row in rows
        if row["held_rate"] <= MAX_HOLD_RATE and row["errors_caught"] > 0
    ]
    if not eligible:
        return None
    best_caught = max(row["errors_caught"] for row in eligible)
    tied = [row for row in eligible if row["errors_caught"] == best_caught]
    return min(tied, key=lambda row: row["threshold"])


def equal_mass_bins(scores, outcomes, bins=N_BINS):
    scores = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    outcomes = outcomes[order]
    ece = 0.0
    rows = []
    n = len(scores)
    for chunk in np.array_split(np.arange(n), bins):
        if len(chunk) == 0:
            continue
        mean_score = float(scores[chunk].mean())
        rate = float(outcomes[chunk].mean())
        ece += (len(chunk) / n) * abs(rate - mean_score)
        rows.append(
            {
                "count": int(len(chunk)),
                "mean_score": mean_score,
                "rate": rate,
            }
        )
    return ece, rows


def brier_score(probabilities, label_index):
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(label_index)), label_index] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def frame_time(path):
    date = None
    for part in Path(path).stem.split("_"):
        if len(part) == 8 and part.isdigit() and part.startswith("2026"):
            date = part
            continue
        if date is not None and len(part) == 6 and part.isdigit():
            return datetime.strptime(date + part, "%Y%m%d%H%M%S")
    return None


def burst_agreement(paths, predictions, window_seconds):
    """Groups of timestamped frames in one class folder, gap at most ``window_seconds``.

    A group of one frame is left out. Agreement is the fraction of the
    remaining groups whose argmax is the same on every frame.
    """
    by_folder = {}
    missing = 0
    for path, prediction in zip(paths, predictions):
        when = frame_time(path)
        if when is None:
            missing += 1
            continue
        by_folder.setdefault(path.parent, []).append((when, prediction))
    groups = 0
    agree = 0
    largest = 0
    for items in by_folder.values():
        items.sort(key=lambda item: item[0])
        current = [items[0]]

        def close():
            nonlocal groups, agree, largest
            if len(current) < 2:
                return
            groups += 1
            largest = max(largest, len(current))
            labels = {item[1] for item in current}
            if len(labels) == 1:
                agree += 1

        for item in items[1:]:
            gap = (item[0] - current[-1][0]).total_seconds()
            if gap <= window_seconds:
                current.append(item)
            else:
                close()
                current = [item]
        close()
    rate = agree / groups if groups else None
    return {
        "window_seconds": window_seconds,
        "groups": groups,
        "agree": agree,
        "rate": rate,
        "largest": largest,
        "missing_timestamp": missing,
    }


def plot_histogram(path, raw_confidence, scaled_confidence, correct):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    bins = np.linspace(0.5, 1.0, 21)
    panels = (
        (axes[0], raw_confidence, "Raw softmax"),
        (axes[1], scaled_confidence, "Temperature-scaled"),
    )
    for axis, confidence, title in panels:
        axis.hist(
            confidence[correct],
            bins=bins,
            label="correct",
            color="tab:green",
            alpha=0.75,
        )
        if np.any(~correct):
            axis.hist(
                confidence[~correct],
                bins=bins,
                label="wrong",
                color="tab:red",
                alpha=0.9,
            )
        axis.set_xlim(0.5, 1.0)
        axis.set_xlabel("winning score")
        axis.set_ylabel("images")
        axis.set_title(title)
        axis.legend()
    figure.suptitle("Validation confidence")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_reliability(path, panels):
    figure, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    for axis, panel in zip(axes.ravel(), panels):
        rows = panel["rows"]
        axis.plot([0, 1], [0, 1], color="0.6", linewidth=1)
        if rows:
            axis.scatter(
                [row["mean_score"] for row in rows],
                [row["rate"] for row in rows],
                s=[max(20, row["count"]) for row in rows],
                zorder=3,
            )
            for row in rows:
                axis.annotate(
                    str(row["count"]),
                    (row["mean_score"], row["rate"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel(panel["xlabel"])
        axis.set_ylabel(panel["ylabel"])
        axis.set_title(f"{panel['title']}\nECE {panel['ece']:.4f}")
    figure.suptitle("Test reliability, equal-mass bins")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_sweep(path, rows, chosen):
    chosen_threshold = None if chosen is None else chosen["threshold"]
    fieldnames = [
        "threshold",
        "held",
        "held_rate",
        "errors_caught",
        "errors_total",
        "accepted",
        "balanced_accuracy",
        "metal_recall",
        "plastic_recall",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            saved = dict(row)
            saved["selected"] = (
                chosen_threshold is not None
                and row["threshold"] == chosen_threshold
            )
            writer.writerow(saved)


def pct(value):
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def count_line(correct, total):
    if total == 0:
        return "0/0"
    return f"{correct}/{total} ({correct / total:.2%})"


def mistake_rows(paths, labels, predictions, raw_confidence, scaled_confidence):
    rows = []
    for index, path in enumerate(paths):
        if predictions[index] == labels[index]:
            continue
        rows.append(
            {
                "path": path,
                "true": labels[index],
                "predicted": predictions[index],
                "raw": float(raw_confidence[index]),
                "scaled": float(scaled_confidence[index]),
            }
        )
    rows.sort(key=lambda row: row["scaled"])
    return rows


def format_mistake(row):
    return (
        f"{row['path'].name} | true {row['true']} | "
        f"pred {row['predicted']} | raw {row['raw']:.2%} | "
        f"scaled {row['scaled']:.2%}"
    )


def accepted_block(confidences, labels, predictions, threshold):
    if threshold is None:
        accepted = np.ones(len(confidences), dtype=bool)
    else:
        accepted = confidences >= threshold
    balanced, detail = balanced_accuracy(
        labels[accepted], predictions[accepted]
    )
    total = int(accepted.sum())
    correct = int((predictions[accepted] == labels[accepted]).sum())
    accuracy = correct / total if total else None
    return {
        "accepted": total,
        "held": int((~accepted).sum()),
        "correct": correct,
        "accuracy": accuracy,
        "balanced": balanced,
        "detail": detail,
    }


def operating_point_lines(title, block, n):
    lines = [
        f"{title}: {block['accepted']} accepted, {block['held']} held "
        f"({block['held'] / n:.2%} of {n}).",
        f"Accuracy on accepted images: {pct(block['accuracy'])} "
        f"({block['correct']}/{block['accepted']}).",
        f"Balanced accuracy on accepted images: {pct(block['balanced'])}.",
    ]
    for name in EXPECTED_CLASSES:
        detail = block["detail"][name]
        lines.append(
            f"{name} recall on accepted: "
            f"{count_line(detail['correct'], detail['total'])}."
        )
    return lines


def main():
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = out_dir / "test_scored.txt"
    if done_flag.exists() and not args.rerun_test:
        raise SystemExit(
            f"{done_flag} already exists, so the locked test split "
            "was scored. Pass --rerun-test to score it again."
        )

    device = pick_device()
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model}", flush=True)
    model, class_to_idx, idx_to_class = load_model(args.model, device)
    metal_index = class_to_idx["metal"]
    eval_transform = inference_transform()

    print("Scoring validation...", flush=True)
    validation_paths = split_paths(args.data_dir, "validation")
    validation_labels = labels_for(validation_paths, class_to_idx)
    validation_logits = score_paths(
        model, validation_paths, eval_transform, device
    )
    torch.save(
        {
            "paths": [str(path) for path in validation_paths],
            "logits": validation_logits,
            "labels": validation_labels,
            "class_to_idx": class_to_idx,
        },
        out_dir / "validation_logits.pt",
    )

    nll_raw = mean_nll(validation_logits, validation_labels, 1.0)
    temperature = fit_temperature(validation_logits, validation_labels)
    nll_scaled = mean_nll(validation_logits, validation_labels, temperature)
    temperature_note = "fitted on validation NLL"
    if nll_scaled > nll_raw + 1e-6:
        temperature = 1.0
        nll_scaled = nll_raw
        temperature_note = (
            "fit did not beat raw NLL, so temperature stays 1"
        )
    print(
        f"Temperature {temperature:.6f} ({temperature_note}); "
        f"validation NLL {nll_raw:.6f} -> {nll_scaled:.6f}",
        flush=True,
    )

    raw_probabilities = scaled_probabilities(validation_logits, 1.0)
    scaled = scaled_probabilities(validation_logits, temperature)
    raw_confidence = max_confidence(raw_probabilities)
    scaled_confidence = max_confidence(scaled)
    label_names = class_names(validation_labels.tolist(), idx_to_class)
    pred_names = prediction_names(scaled, idx_to_class)
    correct = pred_names == label_names
    plot_histogram(
        out_dir / "confidence_hist_validation.png",
        raw_confidence,
        scaled_confidence,
        correct,
    )
    rows = sweep_thresholds(scaled_confidence, label_names, pred_names)
    chosen = choose_threshold(rows)
    write_sweep(out_dir / "threshold_sweep.csv", rows, chosen)
    mistakes = mistake_rows(
        validation_paths,
        label_names,
        pred_names,
        raw_confidence,
        scaled_confidence,
    )
    threshold = None if chosen is None else chosen["threshold"]
    print(
        "Frozen threshold: "
        + ("none (two-class)" if threshold is None else f"{threshold:.2f}"),
        flush=True,
    )
    print(
        "Temperature and threshold are fixed. "
        "The test split is scored once below and does not change them.",
        flush=True,
    )

    # Locked test. Do not refit T or the threshold from anything below.
    print("Scoring test...", flush=True)
    test_paths = split_paths(args.data_dir, "test")
    test_labels = labels_for(test_paths, class_to_idx)
    test_logits = score_paths(model, test_paths, eval_transform, device)
    torch.save(
        {
            "paths": [str(path) for path in test_paths],
            "logits": test_logits,
            "labels": test_labels,
            "class_to_idx": class_to_idx,
            "temperature": temperature,
            "threshold": threshold,
        },
        out_dir / "test_logits.pt",
    )

    test_raw = scaled_probabilities(test_logits, 1.0).numpy()
    test_scaled = scaled_probabilities(test_logits, temperature).numpy()
    test_label_names = class_names(test_labels.tolist(), idx_to_class)
    test_pred = np.array(
        [idx_to_class[int(index)] for index in test_scaled.argmax(axis=1)]
    )
    test_raw_pred = np.array(
        [idx_to_class[int(index)] for index in test_raw.argmax(axis=1)]
    )
    test_confidence = test_scaled.max(axis=1)
    test_raw_confidence = test_raw.max(axis=1)
    test_correct = test_pred == test_label_names
    raw_correct = test_raw_pred == test_label_names
    label_index = test_labels.numpy()
    is_metal = label_index == metal_index

    top_raw_ece, top_raw_bins = equal_mass_bins(
        test_raw_confidence, raw_correct.astype(np.float64)
    )
    top_scaled_ece, top_scaled_bins = equal_mass_bins(
        test_confidence, test_correct.astype(np.float64)
    )
    metal_raw_ece, metal_raw_bins = equal_mass_bins(
        test_raw[:, metal_index], is_metal.astype(np.float64)
    )
    metal_scaled_ece, metal_scaled_bins = equal_mass_bins(
        test_scaled[:, metal_index], is_metal.astype(np.float64)
    )
    brier_raw = brier_score(test_raw, label_index)
    brier_scaled = brier_score(test_scaled, label_index)
    plot_reliability(
        out_dir / "reliability_test.png",
        [
            {
                "title": "Top label, raw",
                "xlabel": "mean winning score",
                "ylabel": "accuracy",
                "ece": top_raw_ece,
                "rows": top_raw_bins,
            },
            {
                "title": "Top label, scaled",
                "xlabel": "mean winning score",
                "ylabel": "accuracy",
                "ece": top_scaled_ece,
                "rows": top_scaled_bins,
            },
            {
                "title": "P(metal), raw",
                "xlabel": "mean P(metal)",
                "ylabel": "fraction metal",
                "ece": metal_raw_ece,
                "rows": metal_raw_bins,
            },
            {
                "title": "P(metal), scaled",
                "xlabel": "mean P(metal)",
                "ylabel": "fraction metal",
                "ece": metal_scaled_ece,
                "rows": metal_scaled_bins,
            },
        ],
    )

    test_block = accepted_block(
        test_confidence, test_label_names, test_pred, threshold
    )
    crushed = None
    for index, path in enumerate(test_paths):
        if CRUSHED_CAN in path.name:
            crushed = {
                "path": path,
                "true": test_label_names[index],
                "predicted": test_pred[index],
                "raw": float(test_raw_confidence[index]),
                "scaled": float(test_confidence[index]),
                "held": threshold is not None
                and float(test_confidence[index]) < threshold,
            }
            break

    bursts = [
        burst_agreement(test_paths, test_pred, window)
        for window in (2, 5)
    ]

    print(
        f"Augmenting test, {AUGMENT_PASSES} passes, seed {AUGMENT_SEED}...",
        flush=True,
    )
    random.seed(AUGMENT_SEED)
    np.random.seed(AUGMENT_SEED)
    torch.manual_seed(AUGMENT_SEED)
    augment = train_transform()
    flip_passes = {name: 0 for name in EXPECTED_CLASSES}
    flip_images = {name: 0 for name in EXPECTED_CLASSES}
    class_totals = {name: 0 for name in EXPECTED_CLASSES}
    image_flipped = np.zeros(len(test_paths), dtype=bool)
    for pass_index in range(AUGMENT_PASSES):
        print(f"Augment pass {pass_index + 1}/{AUGMENT_PASSES}", flush=True)
        aug_logits = score_paths(model, test_paths, augment, device)
        aug_pred = prediction_names(
            scaled_probabilities(aug_logits, temperature), idx_to_class
        )
        disagree = aug_pred != test_pred
        image_flipped |= disagree
        for name in EXPECTED_CLASSES:
            mask = test_label_names == name
            flip_passes[name] += int(disagree[mask].sum())
            class_totals[name] = int(mask.sum())
    for name in EXPECTED_CLASSES:
        mask = test_label_names == name
        flip_images[name] = int(image_flipped[mask].sum())

    lines = [
        "# Confidence report",
        "",
        f"Model: `{args.model}`",
        f"Data: `{args.data_dir}`",
        "",
        "## Validation",
        "",
        f"Images: {len(validation_paths)}.",
        f"Argmax errors after scaling: {len(mistakes)}.",
        f"Temperature: {temperature:.6f} ({temperature_note}).",
        f"Validation NLL: {nll_raw:.6f} raw, {nll_scaled:.6f} scaled.",
        "",
        "Wrong argmax calls, lowest scaled score first:",
        "",
    ]
    if mistakes:
        lines.extend(f"- {format_mistake(row)}" for row in mistakes)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Threshold rule: among scores from 0.50 to 0.99 that hold "
            f"at most {MAX_HOLD_RATE:.0%} of validation images, pick the "
            "one that catches the most wrong argmax calls. Ties take the "
            "lower threshold. If none catch an error, leave sorting two-class.",
            "",
        ]
    )
    if chosen is None:
        lines.append(
            "Chosen threshold: none. No threshold at or below the hold "
            "cap caught a validation error, so the misses are high-confidence "
            "and the default stays two-class."
        )
    else:
        lines.append(
            f"Chosen threshold: {chosen['threshold']:.2f}. "
            f"Holds {chosen['held']} images "
            f"({chosen['held_rate']:.2%}), catches "
            f"{chosen['errors_caught']} of {chosen['errors_total']} errors. "
            f"Balanced accuracy on the rest: {pct(chosen['balanced_accuracy'])}."
        )
    lines.extend(["", "Sweep rows that catch a new error, plus the choice:", ""])
    seen_caught = set()
    for row in rows:
        mark = (
            chosen is not None and row["threshold"] == chosen["threshold"]
        )
        if row["errors_caught"] not in seen_caught or mark:
            seen_caught.add(row["errors_caught"])
            flag = " chosen" if mark else ""
            lines.append(
                f"- {row['threshold']:.2f}: held {row['held']} "
                f"({row['held_rate']:.2%}), caught {row['errors_caught']}, "
                f"balanced accuracy {pct(row['balanced_accuracy'])}{flag}"
            )
    lines.extend(
        [
            "",
            "Full sweep: `threshold_sweep.csv`.",
            "Histogram: `confidence_hist_validation.png`.",
            "",
            "## Test",
            "",
            "Scored once after the temperature and threshold above were fixed.",
            f"Images: {len(test_paths)}.",
            "",
        ]
    )
    if threshold is None:
        lines.append("No hold. Every test image keeps the winning class.")
    else:
        lines.extend(
            operating_point_lines(
                f"Threshold {threshold:.2f} on the scaled winning score",
                test_block,
                len(test_paths),
            )
        )
    wrong_and_held = 0
    correct_and_held = 0
    if threshold is not None:
        held_mask = test_confidence < threshold
        wrong_and_held = int((~test_correct & held_mask).sum())
        correct_and_held = int((test_correct & held_mask).sum())
        lines.append(
            f"Held and wrong: {wrong_and_held}. Held and correct: {correct_and_held}."
        )
    lines.append("")
    if crushed is None:
        lines.append(f"Crushed can `{CRUSHED_CAN}` was not in the test split.")
    else:
        state = "held" if crushed["held"] else "sorted"
        lines.append(
            f"Crushed can `{crushed['path'].name}`: true {crushed['true']}, "
            f"pred {crushed['predicted']}, raw {crushed['raw']:.2%}, "
            f"scaled {crushed['scaled']:.2%}, {state}."
        )
    lines.extend(
        [
            "",
            f"Top-label ECE, {N_BINS} equal-mass bins: "
            f"raw {top_raw_ece:.4f}, scaled {top_scaled_ece:.4f}.",
            f"P(metal) ECE, {N_BINS} equal-mass bins: "
            f"raw {metal_raw_ece:.4f}, scaled {metal_scaled_ece:.4f}.",
            f"Brier score: raw {brier_raw:.6f}, scaled {brier_scaled:.6f}.",
            "Reliability diagram: `reliability_test.png`.",
            "",
            "Burst agreement uses argmax, not the hold. Groups are "
            "timestamped frames in the same class folder. A frame with no "
            "timestamp is left out. A group of one frame is left out.",
            "",
        ]
    )
    for burst in bursts:
        rate = pct(burst["rate"])
        lines.append(
            f"{burst['window_seconds']}s: {burst['agree']}/{burst['groups']} "
            f"groups agree ({rate}), largest group {burst['largest']}, "
            f"{burst['missing_timestamp']} frames without a timestamp."
        )
    lines.extend(
        [
            "",
            f"Augmentation: {AUGMENT_PASSES} passes of the training crop, "
            f"flip, ±15° rotation, and ±20% brightness and contrast, "
            f"seed {AUGMENT_SEED}. "
            "A pass flips when its argmax differs from the full-frame argmax.",
            "",
        ]
    )
    for name in EXPECTED_CLASSES:
        total = class_totals[name]
        passes = total * AUGMENT_PASSES
        lines.append(
            f"{name}: {flip_passes[name]}/{passes} passes flipped "
            f"({pct(flip_passes[name] / passes if passes else None)}), "
            f"{flip_images[name]}/{total} images flipped at least once "
            f"({pct(flip_images[name] / total if total else None)})."
        )
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    done_flag.write_text(
        f"temperature={temperature:.6f}\nthreshold="
        + ("none" if threshold is None else f"{threshold:.2f}")
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
