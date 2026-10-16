"""Checkpoints, saved analysis folders, and the files inside them."""

import csv
import re
from datetime import datetime
from pathlib import Path

from decide import TEMPERATURE, THRESHOLD, frozen_rule_applies

from web.markdown import section, to_html
from web.paths import DATA_ROOTS, IMAGE_SUFFIXES, RESINS, ROOT

LEGACY_RUNS = {
    "resnext50_metal_plastic_belt_stage1_layer4.pt": (
        "analysis/belt_stage1_layer4",
        "analysis/belt_stage1_layer4_val",
    ),
    "resnext50_metal_plastic_sep25_layer4.pt": (
        "analysis/belt_sep25_layer4",
    ),
}

OTHER_ANALYSES = (
    ("sep16_camera", "analysis/sep16_camera", "Sep 16 camera"),
    ("belt_models", "analysis/belt_models", "Older belt misses"),
)

_META = {}
_MISTAKE_LINE = re.compile(
    r"^(?P<path>.+?) \| true: (?P<true>\w+)"
    r"(?: \| resin: (?P<resin>\w+))? \| "
    r"(?:pred|argmax): (?P<pred>\w+) \((?P<score>[0-9.]+)%\)"
)
_LAYER = re.compile(r"_layer([1-4])$")


def list_models():
    models_dir = ROOT / "models"
    if not models_dir.is_dir():
        return []
    aliases = {}
    files = []
    for path in sorted(models_dir.rglob("*.pt")):
        if path.is_symlink():
            aliases.setdefault(path.resolve(), []).append(
                path.relative_to(models_dir).as_posix()
            )
            continue
        if path.is_file():
            files.append(path)
    models = [_model_record(path, aliases.get(path.resolve(), [])) for path in files]
    models.sort(key=lambda item: (not item["frozen"], -item["mtime"]))
    return models


def model_by_stem(stem):
    for model in list_models():
        if model["stem"] == stem:
            return model
    return None


def other_analyses():
    found = []
    for slug, rel, label in OTHER_ANALYSES:
        path = ROOT / rel
        if path.is_dir():
            found.append({"slug": slug, "rel": rel, "label": label})
    return found


def other_by_slug(slug):
    for item in other_analyses():
        if item["slug"] == slug:
            return item
    return None


def dataset_counts():
    rows = []
    for name in DATA_ROOTS:
        root = ROOT / name
        if not root.is_dir():
            rows.append({"name": name, "present": False})
            continue
        splits = []
        for split in ("train", "validation", "test"):
            metal = _count_images(root / split / "metal")
            plastic = _count_images(root / split / "plastic")
            splits.append(
                {
                    "name": split,
                    "metal": metal,
                    "plastic": plastic,
                    "total": metal + plastic,
                }
            )
        loose = []
        for class_name in ("metal", "plastic"):
            folder = root / class_name
            if folder.is_dir():
                loose.append(
                    {"name": class_name, "count": _count_images(folder)}
                )
        rows.append(
            {
                "name": name,
                "present": True,
                "splits": splits,
                "loose": loose,
            }
        )
    return rows


def present_data_roots():
    return [name for name in DATA_ROOTS if (ROOT / name).is_dir()]


def runs_for(model):
    found = []
    seen = set()
    for rel in LEGACY_RUNS.get(model["filename"], ()):
        path = ROOT / rel
        if path.is_dir() and path.resolve() not in seen:
            found.append(path)
            seen.add(path.resolve())
    base = ROOT / "analysis" / model["stem"]
    if base.is_dir():
        for path in base.iterdir():
            if path.is_dir() and path.resolve() not in seen:
                found.append(path)
                seen.add(path.resolve())
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [_run_brief(path) for path in found]


def confidence_dirs(model, runs):
    found = []
    seen = set()
    for run in runs:
        path = ROOT / run["rel"]
        if _is_confidence_dir(path):
            _add_confidence(found, seen, path)
        child = path / "confidence"
        if child.is_dir():
            _add_confidence(found, seen, child)
    return found


def page_for_model(model, run_rel, confidence_rel):
    runs = runs_for(model)
    confidences = confidence_dirs(model, runs)
    selected = _pick(runs, run_rel)
    if run_rel and selected is None and runs:
        selected = runs[0]
    selected_confidence = _pick(confidences, confidence_rel)
    if selected_confidence is None and selected is not None:
        child = (ROOT / selected["rel"] / "confidence")
        if child.is_dir():
            selected_confidence = _confidence_brief(child)
        elif _is_confidence_dir(ROOT / selected["rel"]):
            selected_confidence = _confidence_brief(ROOT / selected["rel"])
    if selected_confidence is None and confidences:
        selected_confidence = confidences[0]
    run = scan_run(ROOT / selected["rel"]) if selected else None
    if run is not None and selected and _is_confidence_dir(ROOT / selected["rel"]):
        run = _empty_run(run["rel"])
    confidence = None
    if model["frozen"] or selected_confidence is not None:
        confidence = load_confidence(selected_confidence)
    eval_log = latest_eval(model["stem"])
    return {
        "runs": runs,
        "selected": selected,
        "run": run,
        "confidences": confidences,
        "confidence": confidence,
        "eval_log": eval_log,
        "show_uncertain": model["frozen"] or selected_confidence is not None,
    }


def page_for_saved(rel):
    path = ROOT / rel
    return scan_run(path)


def latest_eval(stem):
    folder = ROOT / "analysis" / stem
    if not folder.is_dir():
        return None
    logs = sorted(folder.glob("eval-*.txt"), reverse=True)
    if not logs:
        return None
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    return {"name": logs[0].name, "text": text}


def scan_run(directory):
    rel = directory.resolve().relative_to(ROOT).as_posix()
    errors = []
    correct = []
    map_groups = {}
    filters = []
    space = []
    stats = []
    reports = []
    montages = []
    mistakes = []
    held = []
    other = []
    scoring = ""
    if not directory.is_dir():
        return _empty_run(rel)
    files = sorted(path for path in directory.iterdir() if path.is_file())
    for path in files:
        if path.suffix == ".pt" or path.name.startswith("eval-"):
            continue
        name = path.name
        if name.startswith("cam_WRONG"):
            errors.append(_cam_card(path))
        elif name.startswith("cam_CORRECT"):
            correct.append(_cam_card(path))
        elif name.startswith("maps_"):
            group, layer, card = _map_card(path)
            map_groups.setdefault(group, {"label": _pretty(group), "layers": {}})
            if layer is not None:
                map_groups[group]["layers"][layer] = card
            else:
                map_groups[group]["layers"][0] = card
        elif name.startswith("filters_"):
            filters.append(_image_card(path))
        elif name.startswith("pca_") or name.startswith("tsne_"):
            space.append(_image_card(path))
        elif name.startswith("stats_"):
            stats.append(_image_card(path))
        elif name.endswith(".md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            reports.append({"name": name, "html": to_html(text)})
            if name == "ANALYSIS_REPORT.md" and not scoring:
                scoring = section(text, "Scoring")
        elif "mistake" in name and name.endswith(".txt"):
            parsed = _parse_mistake_file(path)
            mistakes.extend(parsed["mistakes"])
            held.extend(parsed["held"])
        elif name.endswith(".png") and (
            "mistake" in name or name.startswith("misclassified")
        ):
            montages.append(_image_card(path))
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            other.append(_image_card(path))
        elif path.suffix.lower() in {".txt", ".csv", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            other.append({"name": name, "text": text, "image": False})
    groups = []
    for key in sorted(map_groups):
        item = map_groups[key]
        layers = [
            item["layers"][layer]
            for layer in sorted(item["layers"])
        ]
        true, resin = _class_and_resin(key)
        groups.append(
            {
                "label": item["label"],
                "true": true,
                "resin": resin,
                "layers": layers,
            }
        )
    return {
        "rel": rel,
        "errors": errors,
        "correct": correct,
        "maps": groups,
        "filters": filters,
        "space": space,
        "stats": stats,
        "reports": reports,
        "montages": montages,
        "mistakes": mistakes,
        "held": held,
        "other": other,
        "scoring_html": to_html(scoring) if scoring else None,
    }


def load_confidence(brief):
    payload = {
        "temperature": f"{float(TEMPERATURE):.6f}",
        "threshold": f"{float(THRESHOLD):.2f}",
        "folder": None,
        "summary_html": None,
        "histogram": None,
        "reliability": None,
        "sweep": [],
        "scored": "",
        "has_test_record": False,
    }
    if brief is None:
        return payload
    folder = ROOT / brief["rel"]
    payload["folder"] = brief
    payload["has_test_record"] = (folder / "test_scored.txt").is_file()
    summary = folder / "summary.md"
    if summary.is_file():
        payload["summary_html"] = to_html(
            summary.read_text(encoding="utf-8", errors="replace")
        )
    scored = folder / "test_scored.txt"
    if scored.is_file():
        payload["scored"] = scored.read_text(encoding="utf-8", errors="replace").strip()
    histogram = folder / "confidence_hist_validation.png"
    if histogram.is_file():
        payload["histogram"] = _image_card(histogram)
    reliability = folder / "reliability_test.png"
    if reliability.is_file():
        payload["reliability"] = _image_card(reliability)
    sweep_path = folder / "threshold_sweep.csv"
    if sweep_path.is_file():
        payload["sweep"] = _read_sweep(sweep_path)
    return payload


def _model_record(path, alias_names):
    stat = path.stat()
    meta = _checkpoint_meta(path)
    frozen = frozen_rule_applies(path)
    return {
        "stem": path.stem,
        "filename": path.name,
        "rel": path.relative_to(ROOT).as_posix(),
        "aliases": alias_names,
        "size": stat.st_size,
        "size_label": f"{stat.st_size / (1024 * 1024):.1f} MB",
        "mtime": stat.st_mtime,
        "mtime_label": datetime.fromtimestamp(stat.st_mtime).strftime(
            "%d %b %Y %H:%M"
        ),
        "frozen": frozen,
        "threshold": f"{float(THRESHOLD):.2f}" if frozen else None,
        "metrics": meta,
    }


def _checkpoint_meta(path):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key in _META:
        return _META[key]
    meta = {
        "validation": None,
        "balanced": None,
        "test": None,
    }
    try:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        meta = {
            "validation": _as_float(blob.get("best_validation_accuracy")),
            "balanced": _as_float(blob.get("best_balanced_accuracy")),
            "test": _as_float(blob.get("test_accuracy")),
        }
        del blob
    except (OSError, RuntimeError, ValueError, KeyError):
        meta["error"] = "Could not read checkpoint metrics."
    _META[key] = meta
    return meta


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_images(folder):
    if not folder.is_dir():
        return 0
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _run_brief(path):
    return {
        "rel": path.resolve().relative_to(ROOT).as_posix(),
        "label": path.name,
    }


def _confidence_brief(path):
    return _run_brief(path)


def _is_confidence_dir(path):
    return path.name == "confidence" or path.name.startswith("confidence-")


def _add_confidence(found, seen, path):
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    found.append(_confidence_brief(path))


def _pick(items, rel):
    if rel:
        for item in items:
            if item["rel"] == rel:
                return item
        return None
    return items[0] if items else None


def _empty_run(rel):
    return {
        "rel": rel,
        "errors": [],
        "correct": [],
        "maps": [],
        "filters": [],
        "space": [],
        "stats": [],
        "reports": [],
        "montages": [],
        "mistakes": [],
        "held": [],
        "other": [],
        "scoring_html": None,
    }


def _image_card(path):
    return {
        "name": path.name,
        "rel": path.resolve().relative_to(ROOT).as_posix(),
        "label": path.stem.replace("_", " "),
    }


def _cam_card(path):
    card = _image_card(path)
    true, resin, predicted = _parse_cam(path.stem)
    card["true"] = true
    card["resin"] = resin
    card["predicted"] = predicted
    bits = []
    if true:
        bits.append(f"true {true}")
    if predicted:
        bits.append(f"pred {predicted}")
    if resin != "unknown":
        bits.append(resin)
    card["caption"] = " · ".join(bits) if bits else card["label"]
    return card


def _parse_cam(stem):
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "cam" or parts[1] not in {"WRONG", "CORRECT"}:
        return "", "unknown", ""
    true = parts[2] if parts[2] in {"metal", "plastic"} else ""
    index = 3
    resin = "unknown"
    if index < len(parts) and parts[index] in RESINS:
        resin = parts[index]
        index += 1
    predicted = ""
    if index < len(parts) and parts[index] in {"metal", "plastic"}:
        predicted = parts[index]
    return true, resin, predicted


def _map_card(path):
    stem = path.stem
    layer = None
    match = _LAYER.search(stem)
    if match:
        layer = int(match.group(1))
        stem = stem[: match.start()]
    card = _image_card(path)
    card["layer"] = layer
    return stem, layer, card


def _class_and_resin(stem):
    parts = stem.split("_")
    true = ""
    if len(parts) > 1 and parts[1] in {"metal", "plastic"}:
        true = parts[1]
    resin = "unknown"
    for part in parts:
        if part in RESINS:
            resin = part
            break
    return true, resin


def _pretty(stem):
    text = stem
    if text.startswith("maps_"):
        text = text[5:]
    return text.replace("_", " ")


def _parse_mistake_file(path):
    mistakes = []
    held = []
    bucket = mistakes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# held"):
            bucket = held
            continue
        match = _MISTAKE_LINE.match(line.strip())
        if not match:
            continue
        resin = match.group("resin") or "unknown"
        image_rel = _image_rel(match.group("path"))
        bucket.append(
            {
                "path": match.group("path"),
                "true": match.group("true"),
                "resin": resin,
                "predicted": match.group("pred"),
                "score": match.group("score") + "%",
                "image_rel": image_rel,
            }
        )
    return {"mistakes": mistakes, "held": held}


def _image_rel(text):
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(ROOT)
    except (OSError, ValueError):
        return None
    if (
        resolved.is_file()
        and resolved.suffix.lower() in IMAGE_SUFFIXES
        and "models" not in resolved.relative_to(ROOT).parts
    ):
        return resolved.relative_to(ROOT).as_posix()
    return None


def _read_sweep(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                held_rate = float(row["held_rate"])
                balanced = float(row["balanced_accuracy"])
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "threshold": row.get("threshold", ""),
                    "held": row.get("held", ""),
                    "held_rate": f"{held_rate:.2%}",
                    "caught": row.get("errors_caught", ""),
                    "errors_total": row.get("errors_total", ""),
                    "balanced": f"{balanced:.2%}",
                    "chosen": str(row.get("selected", "")).strip().lower() == "true",
                }
            )
    return rows
