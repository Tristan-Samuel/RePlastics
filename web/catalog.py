"""Checkpoints, saved analysis folders, and the files inside them."""

import csv
import re
from datetime import datetime
from pathlib import Path

from decide import (
    TEMPERATURE,
    THRESHOLD,
    frozen_rule_applies,
    scaled_probabilities,
    scoring_temperature,
)

from preprocess import RESIN_ALIASES, resin_from_path

from web.markdown import section, to_html
from web.paths import DATA_ROOTS, IMAGE_SUFFIXES, RESINS, ROOT

LEGACY_RUNS = {
    "resnext50_metal_plastic_belt_stage1_layer4.pt": (
        "analysis/belt-stage1",
    ),
}

_META = {}
_INDEX = {}
_ORIGIN_LABELS = (
    ("stage1", "Stage 1"),
    ("belt", "Belt, 11–17 Sep"),
    ("sep28", "28 Sep cans"),
    ("sep29", "29 Sep cans"),
    ("sep30", "30 Sep cans"),
)
_RESIN_ORDER = ("PET", "HDPE", "PP", "OTHERS")
_CAPTURE = re.compile(r"capture_\d{5}_\d{8}_\d{6}")
_DAY = re.compile(r"2026-\d{2}-\d{2}")
_LAYER_SUFFIX = re.compile(r"_layer[1-4]$")
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


def _folder_label(name):
    text = name.replace("-", " ").replace("_", " ")
    return re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)


def _checkpoint_names():
    models_dir = ROOT / "models"
    if not models_dir.is_dir():
        return []
    names = []
    for path in models_dir.rglob("*.pt"):
        if path.is_symlink() or not path.is_file():
            continue
        names.append(path.name)
    return names


def other_analyses():
    """Analysis folders a clone can open when the checkpoint is not on disk."""
    analysis = ROOT / "analysis"
    if not analysis.is_dir():
        return []
    attached = set()
    for filename in _checkpoint_names():
        for rel in LEGACY_RUNS.get(filename, ()):
            attached.add((ROOT / rel).resolve())
        stem = filename[:-3] if filename.endswith(".pt") else filename
        attached.add((analysis / stem).resolve())
    found = []
    for path in sorted(analysis.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.resolve() in attached:
            continue
        found.append(
            {
                "slug": path.name,
                "rel": path.relative_to(ROOT).as_posix(),
                "label": _folder_label(path.name),
            }
        )
    return found


def other_by_slug(slug):
    for item in other_analyses():
        if item["slug"] == slug:
            return item
    return None


def dataset_counts():
    return [composition(name) for name in DATA_ROOTS]


def composition(name):
    root = ROOT / name
    if name not in DATA_ROOTS or not root.is_dir():
        return {"name": name, "present": False}
    rows = image_index(name)
    splits = []
    for split in ("train", "validation", "test"):
        metal = sum(1 for row in rows if row["split"] == split and row["class_name"] == "metal")
        plastic = sum(1 for row in rows if row["split"] == split and row["class_name"] == "plastic")
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
        count = sum(1 for row in rows if row["split"] == "" and row["class_name"] == class_name)
        if count:
            loose.append({"name": class_name, "count": count})
    resins = []
    for resin in _RESIN_ORDER:
        metal = sum(1 for row in rows if row["resin"] == resin and row["class_name"] == "metal")
        plastic = sum(1 for row in rows if row["resin"] == resin and row["class_name"] == "plastic")
        resins.append(
            {"name": resin, "metal": metal, "plastic": plastic, "total": metal + plastic}
        )
    untagged_metal = sum(1 for row in rows if row["resin"] == "" and row["class_name"] == "metal")
    untagged_plastic = sum(1 for row in rows if row["resin"] == "" and row["class_name"] == "plastic")
    resins.append(
        {
            "name": "No tag",
            "metal": untagged_metal,
            "plastic": untagged_plastic,
            "total": untagged_metal + untagged_plastic,
        }
    )
    origins = _origin_rows(rows)
    metal = sum(row["metal"] for row in splits) + sum(item["count"] for item in loose if item["name"] == "metal")
    plastic = sum(row["plastic"] for row in splits) + sum(item["count"] for item in loose if item["name"] == "plastic")
    return {
        "name": name,
        "present": True,
        "total": metal + plastic,
        "metal": metal,
        "plastic": plastic,
        "splits": splits,
        "loose": loose,
        "resins": resins,
        "origins": origins,
    }


def search_images(name, query, split, class_name, resin, origin, limit=48):
    if name not in DATA_ROOTS:
        return {"total": 0, "shown": 0, "images": []}
    needle = (query or "").strip().lower()
    split = split if split in {"train", "validation", "test"} else ""
    class_name = class_name if class_name in {"metal", "plastic"} else ""
    resin = resin if resin in set(_RESIN_ORDER) or resin == "none" else ""
    hits = []
    for row in image_index(name):
        if split and row["split"] != split:
            continue
        if class_name and row["class_name"] != class_name:
            continue
        if resin == "none" and row["resin"]:
            continue
        if resin and resin != "none" and row["resin"] != resin:
            continue
        if origin and row["origin"] != origin:
            continue
        if needle and needle not in row["name"].lower() and needle not in row["stem"].lower():
            continue
        hits.append(row)
    shown = hits[:limit]
    return {
        "total": len(hits),
        "shown": len(shown),
        "images": [
            {
                "name": row["name"],
                "rel": row["rel"],
                "split": row["split"],
                "class_name": row["class_name"],
                "resin": row["resin"],
                "origin": row["origin"],
                "origin_label": _origin_label(row["origin"]),
            }
            for row in shown
        ],
    }


def link_run_sources(run, dataset_name):
    """Point analysis plots at a dataset photo when the filename identifies one."""
    if not run or dataset_name not in DATA_ROOTS:
        return
    rows = image_index(dataset_name)
    cards = []
    for key in ("errors", "correct", "filters", "space", "stats", "montages", "other"):
        cards.extend(item for item in run.get(key, []) if isinstance(item, dict) and item.get("rel"))
    for group in run.get("maps", []):
        cards.extend(group.get("layers", []))
    for card in cards:
        query = _photo_query(card.get("name") or "")
        card["query"] = query
        match = _unique_match(rows, query)
        if match is None:
            continue
        card["source_rel"] = match["rel"]
        card["source_name"] = match["name"]
        card["source_split"] = match["split"]
        card["source_class"] = match["class_name"]
        card["source_resin"] = match["resin"]
        card["source_origin"] = _origin_label(match["origin"])


def image_index(name):
    root = ROOT / name
    stamp = []
    manifest = root / "SPLIT_MANIFEST.tsv"
    if manifest.is_file():
        stat = manifest.stat()
        stamp.append((stat.st_mtime_ns, stat.st_size))
    for split in ("train", "validation", "test", ""):
        for class_name in ("metal", "plastic"):
            folder = root / class_name if split == "" else root / split / class_name
            if folder.is_dir():
                stat = folder.stat()
                stamp.append((str(folder), stat.st_mtime_ns))
    key = (name, tuple(stamp))
    cached = _INDEX.get(name)
    if cached and cached[0] == key:
        return cached[1]
    origins = _manifest_origins(manifest)
    rows = []
    seen = set()
    for split in ("train", "validation", "test", ""):
        for class_name in ("metal", "plastic"):
            folder = root / class_name if split == "" else root / split / class_name
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if path.name.startswith("."):
                    continue
                rel = path.resolve().relative_to(ROOT).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                rows.append(
                    {
                        "name": path.name,
                        "stem": path.stem,
                        "rel": rel,
                        "split": split,
                        "class_name": class_name,
                        "resin": _photo_resin(path),
                        "origin": origins.get(path.name, ""),
                    }
                )
    rows.sort(key=lambda row: row["name"])
    _INDEX[name] = (key, rows)
    return rows


def _manifest_origins(path):
    origins = {}
    if not path.is_file():
        return origins
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        origins[parts[3]] = parts[2]
    return origins


def _origin_rows(rows):
    if not any(row["origin"] for row in rows):
        return []
    order = [key for key, _label in _ORIGIN_LABELS]
    extra = sorted({row["origin"] for row in rows if row["origin"] and row["origin"] not in order})
    found = []
    for key in order + extra:
        metal = sum(1 for row in rows if row["origin"] == key and row["class_name"] == "metal")
        plastic = sum(1 for row in rows if row["origin"] == key and row["class_name"] == "plastic")
        if metal or plastic:
            found.append(
                {
                    "key": key,
                    "label": _origin_label(key),
                    "metal": metal,
                    "plastic": plastic,
                    "total": metal + plastic,
                }
            )
    return found


def _origin_label(key):
    for name, label in _ORIGIN_LABELS:
        if name == key:
            return label
    return key or ""


def _photo_resin(path):
    resin = resin_from_path(path)
    if resin:
        return resin
    head = Path(path).stem.split("_", 1)[0]
    return RESIN_ALIASES.get(head.lower(), "")


# Published belt-stage1 plots were renamed away from the capture id.
_SLUG_QUERY = {
    "crushed-can": "capture_01896_20260929_141207_925981",
    "foil-tray": "capture_00168_20260917_123444_682681",
    "can-2026-09-29": "capture_01895_20260929_141206_407448",
    "bottle": "HDPE_capture_00014_20251112_120713_685907",
    "belt-frame": "capture_00215_20260914_102957_949156",
}


def _photo_query(filename):
    stem = Path(filename).stem
    capture = _CAPTURE.search(stem)
    if capture:
        return capture.group(0)
    if stem.startswith("maps_"):
        stem = _LAYER_SUFFIX.sub("", stem)
    token = stem.split("_")[-1]
    if token in _SLUG_QUERY:
        return _SLUG_QUERY[token]
    day = _DAY.search(stem)
    if day:
        return day.group(0).replace("-", "")
    if stem.startswith("maps_") or stem.startswith("cam_"):
        return token if len(token) >= 4 else ""
    return ""


def _unique_match(rows, query):
    if not query or len(query) < 4:
        return None
    needle = query.lower()
    hits = [
        row
        for row in rows
        if needle in row["stem"].lower() or needle in row["name"].lower()
    ]
    if len(hits) == 1:
        return hits[0]
    exact = [row for row in hits if row["stem"].lower() == needle]
    if len(exact) == 1:
        return exact[0]
    return None


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
    runs, confidences, selected, selected_confidence = _selection(
        model, run_rel, confidence_rel
    )
    run = scan_run(ROOT / selected["rel"]) if selected else None
    if run is not None and selected and _is_confidence_dir(ROOT / selected["rel"]):
        run = _empty_run(run["rel"])
    confidence = None
    if model["frozen"] or selected_confidence is not None:
        confidence = load_confidence(selected_confidence)
    eval_log = latest_eval(model["stem"])
    if selected_confidence is not None:
        confidence_choice = selected_confidence["rel"]
    else:
        confidence_choice = confidence_rel or ""
    return {
        "runs": runs,
        "selected": selected,
        "run": run,
        "confidences": confidences,
        "confidence": confidence,
        "confidence_choice": confidence_choice,
        "eval_log": eval_log,
        "show_uncertain": model["frozen"] or selected_confidence is not None,
        "has_scores": _logits_present(selected_confidence),
    }


def scores_for_model(model, run_rel, confidence_rel):
    """Per-image winning scores for the selected confidence folder.

    The slider in the browser recomputes holds from these rows. Nothing
    here writes the frozen threshold.
    """
    _runs, _confidences, _selected, selected_confidence = _selection(
        model, run_rel, confidence_rel
    )
    payload = {
        "temperature": round(scoring_temperature(), 6),
        "frozen_threshold": float(THRESHOLD) if THRESHOLD is not None else None,
        "frames": [],
        "stats": {},
    }
    if selected_confidence is None:
        return payload
    folder = ROOT / selected_confidence["rel"]
    frames = []
    temperature = scoring_temperature()
    for split, filename in (
        ("validation", "validation_logits.pt"),
        ("test", "test_logits.pt"),
    ):
        path = folder / filename
        if not path.is_file():
            continue
        split_frames, split_temperature = _frames_from_logits(path, split)
        frames.extend(split_frames)
        if split_temperature is not None:
            temperature = split_temperature
    frames.extend(_frames_from_excluded(folder))
    payload["temperature"] = round(float(temperature), 6)
    payload["frames"] = frames
    payload["stats"] = _stats_by_split(frames)
    return payload


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


def _run_brief(path):
    return {
        "rel": path.resolve().relative_to(ROOT).as_posix(),
        "label": path.name,
    }


def _confidence_brief(path):
    brief = _run_brief(path)
    if path.name == "confidence":
        brief["label"] = path.parent.name
    return brief


def _is_confidence_dir(path):
    return path.name == "confidence" or path.name.startswith("confidence-")


def _add_confidence(found, seen, path):
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    found.append(_confidence_brief(path))


def _selection(model, run_rel, confidence_rel):
    runs = runs_for(model)
    confidences = confidence_dirs(model, runs)
    selected = _pick(runs, run_rel)
    if run_rel and selected is None and runs:
        selected = runs[0]
    for run in runs:
        path = ROOT / run["rel"]
        child = path / "confidence"
        if child.is_dir() and _logits_present(_confidence_brief(child)):
            run["confidence_rel"] = child.resolve().relative_to(ROOT).as_posix()
        elif _is_confidence_dir(path) and _logits_present(run):
            run["confidence_rel"] = run["rel"]
        else:
            run["confidence_rel"] = ""
    selected_confidence = _pick(confidences, confidence_rel)
    explicit_miss = bool(confidence_rel) and selected_confidence is None
    if selected_confidence is None and not explicit_miss and selected is not None:
        child = ROOT / selected["rel"] / "confidence"
        if child.is_dir() and _logits_present(_confidence_brief(child)):
            selected_confidence = _confidence_brief(child)
        elif _is_confidence_dir(ROOT / selected["rel"]) and _logits_present(selected):
            selected_confidence = _confidence_brief(ROOT / selected["rel"])
    if selected_confidence is None and not explicit_miss and confidences:
        selected_confidence = confidences[0]
    return runs, confidences, selected, selected_confidence


def _logits_present(brief):
    if brief is None:
        return False
    folder = ROOT / brief["rel"]
    return (folder / "validation_logits.pt").is_file() or (
        folder / "test_logits.pt"
    ).is_file()


def _frames_from_logits(path, split):
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = blob.get("class_to_idx") or {}
    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    stored = blob.get("temperature")
    if stored is None:
        temperature = scoring_temperature()
    else:
        temperature = float(stored)
        if temperature <= 0:
            temperature = scoring_temperature()
    probabilities = scaled_probabilities(blob["logits"], temperature)
    confidence, picked = probabilities.max(dim=-1)
    labels = blob["labels"]
    frames = []
    for index, image_path in enumerate(blob["paths"]):
        true_index = int(labels[index])
        pred_index = int(picked[index])
        true_name = idx_to_class.get(true_index, str(true_index))
        pred_name = idx_to_class.get(pred_index, str(pred_index))
        text = str(image_path)
        frames.append(
            {
                "split": split,
                "path": text,
                "true": true_name,
                "pred": pred_name,
                "score": round(float(confidence[index]), 6),
                "resin": _resin_name(text),
                "image": _image_rel(text),
            }
        )
    del blob
    return frames, temperature


def _frames_from_excluded(folder):
    candidates = (
        folder / "stage1_excluded" / "scores.tsv",
        folder.parent / "stage1_excluded" / "scores.tsv",
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        return []
    frames = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                score = float(row["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            text = row.get("path") or ""
            frames.append(
                {
                    "split": "excluded",
                    "path": text,
                    "true": "metal",
                    "pred": row.get("argmax") or "",
                    "score": round(score, 6),
                    "resin": _resin_name(text),
                    "image": _image_rel(text),
                }
            )
    return frames


def _stats_by_split(frames):
    grouped = {}
    for frame in frames:
        grouped.setdefault(frame["split"], []).append(frame)
    return {split: _spread_groups(rows) for split, rows in grouped.items()}


def _spread_groups(frames):
    return {
        "all": _spread([frame["score"] for frame in frames]),
        "correct": _spread(
            [
                frame["score"]
                for frame in frames
                if frame["pred"] == frame["true"]
            ]
        ),
        "wrong": _spread(
            [
                frame["score"]
                for frame in frames
                if frame["pred"] != frame["true"]
            ]
        ),
    }


def _spread(values):
    count = len(values)
    if count == 0:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(values) / count
    if count < 2:
        return {"n": count, "mean": mean, "std": None}
    variance = sum((value - mean) ** 2 for value in values) / count
    return {"n": count, "mean": mean, "std": variance ** 0.5}


def _resin_name(text):
    for part in Path(text).stem.split("_"):
        if part in RESINS:
            return part
    return "unknown"


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
