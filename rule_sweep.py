"""Sweep hold rules on one checkpoint.

Validation chooses the rule. Test is scored after that and does not
move the cutoffs. Probe folders are reported the same way: metal and
plastic should keep their class, and other should be held.

A photo is uncertain when the winning score is under the confidence
cutoff and the distance is still inside the mark. It is others when
the distance is past the mark, unless confidence reaches the keep
line. Both of those are holds. Neither cutoff is written into decide.py.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from confidence_report import load_model, pick_device, split_paths
from decide import scoring_temperature
from open_set import (
    embed_and_logits,
    fit_bank,
    labels_for,
    measure,
    probe_true_class,
    train_for_bank,
)
from web.paths import IMAGE_SUFFIXES


CONF_CUTS = (None, 0.5, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
KEEP_CUTS = (None, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995)
REJECT_QUANTILES = (None, 0.9, 0.95, 0.97, 0.99, 0.995)
REJECT_NAMES = ("cloud", "call", "knn", "maha", "energy")
HOLD_CAP = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep confidence and distance holds.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data_belt_stage1"))
    parser.add_argument("--probes", type=Path, default=Path("data/probes"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def probe_paths(folder):
    paths = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise SystemExit(f"No probe photos in {folder}")
    return paths


def pack(paths, measured, logits, true_names):
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    energy = (-torch.logsumexp(logits, dim=-1)).numpy()
    rows = []
    for index, path in enumerate(paths):
        rows.append(
            {
                "path": path.as_posix(),
                "name": path.name,
                "true": true_names[index],
                "pred": measured["pred"][index],
                "confidence": float(measured["confidence"][index]),
                "margin": float(margin[index]),
                "cloud": float(measured["cloud"][index]),
                "call": float(measured["call"][index]),
                "knn": float(measured["knn"][index]),
                "maha": float(measured["maha"][index]),
                "energy": float(energy[index]),
                "empty": "empty" in path.name.lower(),
            }
        )
    return rows


def reject_mark(rows, name, quantile):
    if quantile is None:
        return None
    correct = [
        row[name]
        for row in rows
        if row["true"] in ("metal", "plastic") and row["pred"] == row["true"]
    ]
    if not correct:
        return None
    return float(np.quantile(np.asarray(correct, dtype=np.float64), quantile))


def decisions(rows, confidence_cut, reject_name, reject_cut, keep_cut):
    labels = []
    for row in rows:
        far = (
            reject_cut is not None
            and reject_name is not None
            and row[reject_name] > reject_cut
        )
        if far and keep_cut is not None and row["confidence"] >= keep_cut:
            far = False
        if far:
            labels.append("others")
        elif confidence_cut is not None and row["confidence"] < confidence_cut:
            labels.append("uncertain")
        else:
            labels.append(row["pred"])
    return labels


def score_known(rows, labels):
    known = [
        (row, label)
        for row, label in zip(rows, labels)
        if row["true"] in ("metal", "plastic")
    ]
    total = len(known)
    accepted = [(row, label) for row, label in known if label in ("metal", "plastic")]
    correct = sum(label == row["true"] for row, label in accepted)
    held = total - len(accepted)
    still_wrong = sum(
        row["pred"] != row["true"] and label in ("metal", "plastic")
        for row, label in known
    )
    caught = sum(
        row["pred"] != row["true"] and label not in ("metal", "plastic")
        for row, label in known
    )
    recalls = []
    per_class = {}
    for name in ("metal", "plastic"):
        class_rows = [(row, label) for row, label in known if row["true"] == name]
        class_accepted = [
            (row, label) for row, label in class_rows if label in ("metal", "plastic")
        ]
        class_correct = sum(label == name for row, label in class_accepted)
        per_class[name] = {
            "n": len(class_rows),
            "accepted": len(class_accepted),
            "correct": class_correct,
        }
        if class_accepted:
            recalls.append(class_correct / len(class_accepted))
        else:
            recalls.append(0.0)
    argmax_correct = sum(row["pred"] == row["true"] for row, _label in known)
    return {
        "n": total,
        "accepted": len(accepted),
        "correct": correct,
        "held": held,
        "hold_rate": held / total if total else 0.0,
        "accuracy": correct / len(accepted) if accepted else 0.0,
        "balanced": float(np.mean(recalls)) if total else 0.0,
        "still_wrong": still_wrong,
        "caught": caught,
        "argmax_accuracy": argmax_correct / total if total else 0.0,
        "per_class": per_class,
    }


def score_probes(rows, labels):
    buckets = {"metal": [], "plastic": [], "other": []}
    for row, label in zip(rows, labels):
        buckets.setdefault(row["true"], []).append((row, label))
    report = {}
    for name, pairs in buckets.items():
        accepted = [label for _row, label in pairs if label in ("metal", "plastic")]
        if name == "other":
            rejected = sum(label not in ("metal", "plastic") for _row, label in pairs)
            empty = [pair for pair in pairs if pair[0]["empty"]]
            junk = [pair for pair in pairs if not pair[0]["empty"]]
            report[name] = {
                "n": len(pairs),
                "rejected": rejected,
                "sorted_metal": sum(label == "metal" for label in accepted),
                "sorted_plastic": sum(label == "plastic" for label in accepted),
                "empty_rejected": sum(
                    label not in ("metal", "plastic") for _row, label in empty
                ),
                "empty_n": len(empty),
                "junk_rejected": sum(
                    label not in ("metal", "plastic") for _row, label in junk
                ),
                "junk_n": len(junk),
            }
        else:
            correct = sum(label == name for _row, label in pairs)
            held = sum(label not in ("metal", "plastic") for _row, label in pairs)
            report[name] = {
                "n": len(pairs),
                "correct": correct,
                "held": held,
                "wrong": len(pairs) - correct - held,
            }
    return report


def rule_name(confidence_cut, reject_name, quantile, keep_cut):
    parts = []
    if confidence_cut is None:
        parts.append("no confidence hold")
    else:
        parts.append(f"confidence {confidence_cut:.3f}")
    if reject_name is None or quantile is None:
        parts.append("no distance")
    else:
        parts.append(f"{reject_name} q{quantile:.3f}")
    if keep_cut is None:
        parts.append("no keep")
    else:
        parts.append(f"keep {keep_cut:.3f}")
    return ", ".join(parts)


def better(candidate, current):
    """Higher accepted accuracy, then balanced, then fewer misses, then less hold."""
    if current is None:
        return True
    keys = (
        ("accuracy", 1),
        ("balanced", 1),
        ("still_wrong", -1),
        ("hold_rate", -1),
    )
    for key, direction in keys:
        left = candidate[key] * direction
        right = current[key] * direction
        if left > right + 1e-12:
            return True
        if left < right - 1e-12:
            return False
    return False


def evaluate(rule, splits):
    confidence_cut, reject_name, quantile, reject_cut, keep_cut = rule
    payload = {
        "name": rule_name(confidence_cut, reject_name, quantile, keep_cut),
        "confidence": confidence_cut,
        "reject": reject_name,
        "quantile": quantile,
        "reject_cut": reject_cut,
        "keep": keep_cut,
    }
    for split, rows in splits.items():
        labels = decisions(rows, confidence_cut, reject_name, reject_cut, keep_cut)
        if split == "probes":
            payload[split] = score_probes(rows, labels)
        else:
            payload[split] = score_known(rows, labels)
        payload[split]["labels"] = labels
    return payload


def drop_labels(payload):
    cleaned = dict(payload)
    for split in ("validation", "test", "probes"):
        block = dict(cleaned[split])
        block.pop("labels", None)
        cleaned[split] = block
    return cleaned


def percent(value):
    return f"{value * 100:.2f}%"


def write_report(path, model, data_dir, baseline, chosen, family_winners, mistakes, probe_note):
    lines = [
        f"# Rule sweep",
        "",
        f"Checkpoint: `{model}`",
        f"Data: `{data_dir}`",
        "",
        "Cutoffs were chosen on validation. Test and probes were scored after that.",
        "Accepted accuracy ignores held photos. The search kept only rules that hold at most 5% of validation.",
        "",
        "## Argmax, no hold",
        "",
        "| Split | Accuracy | Balanced | Metal | Plastic |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        block = baseline[split]
        metal = block["per_class"]["metal"]
        plastic = block["per_class"]["plastic"]
        lines.append(
            f"| {split} | {percent(block['argmax_accuracy'])} | "
            f"{percent(block['balanced'])} | "
            f"{metal['correct']}/{metal['n']} | "
            f"{plastic['correct']}/{plastic['n']} |"
        )
    lines.extend(
        [
            "",
            "## Validation-chosen rule",
            "",
            f"`{chosen['name']}`",
            "",
            "| Split | Accepted accuracy | Balanced | Held | Still wrong | Caught |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("validation", "test"):
        block = chosen[split]
        lines.append(
            f"| {split} | {block['correct']}/{block['accepted']} "
            f"({percent(block['accuracy'])}) | {percent(block['balanced'])} | "
            f"{block['held']}/{block['n']} ({percent(block['hold_rate'])}) | "
            f"{block['still_wrong']} | {block['caught']} |"
        )
    lines.extend(["", "## Best rule in each family", ""])
    lines.append("| Family | Rule | Val accepted | Val held | Test accepted | Test held | Test still wrong |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for payload in family_winners:
        val = payload["validation"]
        test = payload["test"]
        lines.append(
            f"| {payload.get('family', '')} | {payload['name']} | "
            f"{percent(val['accuracy'])} | {percent(val['hold_rate'])} | "
            f"{percent(test['accuracy'])} | {percent(test['hold_rate'])} | "
            f"{test['still_wrong']} |"
        )
    lines.extend(["", "## Test mistakes under argmax", ""])
    if not mistakes:
        lines.append("None.")
    else:
        lines.append("| File | True | Pred | Confidence | Cloud | Call | k-NN | Mahalanobis | Energy |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in mistakes:
            lines.append(
                f"| `{row['name']}` | {row['true']} | {row['pred']} | "
                f"{row['confidence']:.4f} | {row['cloud']:.4f} | {row['call']:.4f} | "
                f"{row['knn']:.4f} | {row['maha']:.3f} | {row['energy']:.3f} |"
            )
    lines.extend(["", "## Probes", "", probe_note, ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out = args.out or Path("analysis") / args.model.stem / "rule-sweep"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Using device: {device}", flush=True)
    model, _class_to_idx, idx_to_class = load_model(args.model, device)
    temperature = scoring_temperature()
    split_rows = {}
    for split in ("train", "validation", "test"):
        paths = split_paths(args.data_dir, split)
        features, logits = embed_and_logits(model, paths, device, split)
        if split == "train":
            counts = {
                "train": len(paths),
                "validation": 0,
                "test": 0,
            }
            fit_features, label_names, class_names = train_for_bank(
                features, paths, idx_to_class
            )
            bank = fit_bank(
                fit_features,
                temperature,
                args.model,
                args.data_dir,
                counts,
                class_names,
            )
        measured = measure(features, logits, bank, idx_to_class)
        split_rows[split] = pack(paths, measured, logits, labels_for(paths))
    probes = probe_paths(args.probes)
    features, logits = embed_and_logits(model, probes, device, "probes")
    measured = measure(features, logits, bank, idx_to_class)
    split_rows["probes"] = pack(
        probes,
        measured,
        logits,
        [probe_true_class(path) for path in probes],
    )
    marks = {}
    for name in REJECT_NAMES:
        marks[name] = {
            quantile: reject_mark(split_rows["validation"], name, quantile)
            for quantile in REJECT_QUANTILES
        }

    rules = [(None, None, None, None, None)]
    for confidence_cut in CONF_CUTS:
        rules.append((confidence_cut, None, None, None, None))
    for reject_name in REJECT_NAMES:
        for quantile, reject_cut in marks[reject_name].items():
            if quantile is None or reject_cut is None:
                continue
            for confidence_cut in CONF_CUTS:
                for keep_cut in KEEP_CUTS:
                    if keep_cut is not None and confidence_cut is not None and keep_cut < confidence_cut:
                        continue
                    rules.append(
                        (confidence_cut, reject_name, quantile, reject_cut, keep_cut)
                    )

    scored = [evaluate(rule, split_rows) for rule in rules]
    eligible = [
        payload
        for payload in scored
        if payload["validation"]["hold_rate"] <= HOLD_CAP + 1e-12
    ]
    chosen = None
    for payload in eligible:
        if better(payload["validation"], None if chosen is None else chosen["validation"]):
            chosen = payload
    if chosen is None:
        raise SystemExit("No rule held at most 5% of validation.")

    families = [("confidence only", None)] + [(name, name) for name in REJECT_NAMES]
    family_winners = []
    for label, reject_name in families:
        winner = None
        for payload in eligible:
            if payload["reject"] != reject_name:
                continue
            if better(
                payload["validation"],
                None if winner is None else winner["validation"],
            ):
                winner = payload
        if winner is not None:
            winner = dict(winner)
            winner["family"] = label
            family_winners.append(winner)

    mistakes = [
        row
        for row in split_rows["test"]
        if row["true"] in ("metal", "plastic") and row["pred"] != row["true"]
    ]
    mistakes.sort(key=lambda row: -row["confidence"])

    probe_lines = [
        "| Rule | Other rejected | Empty rejected | Junk rejected | Metal probes | Plastic probes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    shown = [scored[0], chosen]
    for payload in family_winners:
        if payload["name"] not in {item["name"] for item in shown}:
            shown.append(payload)
    for payload in shown:
        probes_block = payload["probes"]
        other = probes_block["other"]
        metal = probes_block["metal"]
        plastic = probes_block["plastic"]
        probe_lines.append(
            f"| {payload['name']} | {other['rejected']}/{other['n']} | "
            f"{other['empty_rejected']}/{other['empty_n']} | "
            f"{other['junk_rejected']}/{other['junk_n']} | "
            f"{metal['correct']}/{metal['n']} correct, {metal['held']} held, {metal['wrong']} wrong | "
            f"{plastic['correct']}/{plastic['n']} correct, {plastic['held']} held, {plastic['wrong']} wrong |"
        )

    report_rows = [drop_labels(payload) for payload in shown]
    (out / "summary.json").write_text(
        json.dumps(
            {
                "model": args.model.as_posix(),
                "data_dir": args.data_dir.as_posix(),
                "rules_tried": len(rules),
                "eligible": len(eligible),
                "chosen": drop_labels(chosen),
                "shown": report_rows,
                "test_mistakes": mistakes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out / "eligible.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "val_accuracy",
                "val_balanced",
                "val_hold",
                "val_still_wrong",
                "test_accuracy",
                "test_balanced",
                "test_hold",
                "test_still_wrong",
                "other_rejected",
            ],
        )
        writer.writeheader()
        for payload in eligible:
            writer.writerow(
                {
                    "name": payload["name"],
                    "val_accuracy": f"{payload['validation']['accuracy']:.6f}",
                    "val_balanced": f"{payload['validation']['balanced']:.6f}",
                    "val_hold": f"{payload['validation']['hold_rate']:.6f}",
                    "val_still_wrong": payload["validation"]["still_wrong"],
                    "test_accuracy": f"{payload['test']['accuracy']:.6f}",
                    "test_balanced": f"{payload['test']['balanced']:.6f}",
                    "test_hold": f"{payload['test']['hold_rate']:.6f}",
                    "test_still_wrong": payload["test"]["still_wrong"],
                    "other_rejected": payload["probes"]["other"]["rejected"],
                }
            )
    write_report(
        out / "summary.md",
        args.model.as_posix(),
        args.data_dir.as_posix(),
        scored[0],
        chosen,
        family_winners,
        mistakes,
        "\n".join(probe_lines),
    )
    test = chosen["test"]
    base = scored[0]["test"]
    print(
        f"Argmax test {base['argmax_accuracy']:.2%} "
        f"({base['per_class']['metal']['correct']}/{base['per_class']['metal']['n']} metal, "
        f"{base['per_class']['plastic']['correct']}/{base['per_class']['plastic']['n']} plastic)",
        flush=True,
    )
    print(
        f"Chosen: {chosen['name']}",
        flush=True,
    )
    print(
        f"Test accepted {test['accuracy']:.2%} on {test['accepted']}/{test['n']}, "
        f"held {test['hold_rate']:.2%}, still wrong {test['still_wrong']}",
        flush=True,
    )
    print(f"Wrote {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
