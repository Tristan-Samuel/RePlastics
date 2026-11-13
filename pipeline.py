"""Cache belt checks and open-set scores for the pipeline page.

Validation and test come from the dataset. ``data/probes`` and
``data/Garbage`` are scored too when those folders are on disk. A
matching open-set bank is reused. Test photos are scored into the cache.
They do not choose a cut. This does not write decide.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from belt_checks import (
    CHECKS,
    audited_blank,
    measure_image,
    pattern_distance,
    suggest_cut,
)
from checkpoints import read_checkpoint
from confidence_report import pick_device, split_paths
from decide import scoring_temperature
from open_set import (
    SCORE_KEYS,
    OpenSetError,
    checkpoint_matches,
    display_path,
    embed_and_logits,
    fit_bank,
    labels_for,
    load_bank,
    load_scorer,
    probe_true_class,
    save_bank,
    score_loaded,
    score_paths,
    suggestions_from,
)
from preprocess import readable_image_paths, resin_from_path, true_class_from_path
from trashnet import EXPECTED_CLASSES
from web.paths import IMAGE_SUFFIXES, ROOT


BELT_VERSION = 1


class PipelineError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score belt checks and checkpoints for the pipeline page.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--belt-only",
        action="store_true",
        help="Score belt checks and skip checkpoints.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Replace a cache that is already on disk.",
    )
    return parser.parse_args()


def cache_root(data_dir):
    return Path("analysis") / "pipeline-cache" / Path(data_dir).name


def _image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return readable_image_paths(found)


def _blank_label(path):
    return any("blank" in part.lower() for part in Path(path).parts)


def _label_for(path, split):
    if _blank_label(path):
        return "empty"
    if split in ("validation", "test", "train"):
        return true_class_from_path(path) or ""
    return probe_true_class(path) or true_class_from_path(path) or ""


def collect_records(data_dir):
    records = []
    for split in ("validation", "test"):
        for path in split_paths(data_dir, split):
            records.append((path, split, _label_for(path, split)))
    extras = (
        ("probe", ROOT / "data" / "probes"),
        ("garbage", ROOT / "data" / "Garbage"),
    )
    for split, folder in extras:
        for path in _image_files(folder):
            records.append((path, split, _label_for(path, split)))
    return records


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".next")
    staging.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    staging.replace(path)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def belt_is_fresh(path, data_dir):
    payload = _read_json(path)
    if not payload:
        return False
    return (
        payload.get("ready") is True
        and payload.get("version") == BELT_VERSION
        and payload.get("data_dir") == str(data_dir)
    )


def _pack_scores(scores):
    packed = {}
    for item in CHECKS:
        packed[item["id"]] = _none_or_number(scores.get(item["id"]))
    return packed


def _none_or_number(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, 6)


def _pattern_bank(blank_specs):
    if not blank_specs:
        return None, None
    bank = np.stack(blank_specs).astype(np.float32)
    mean = bank.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return None, None
    return mean / norm, bank


def score_belt(records):
    blank_specs = []
    for path, _split, true_name in records:
        if true_name != "empty" or not audited_blank(path):
            continue
        _scores, spec = measure_image(path)
        if spec is not None:
            blank_specs.append(spec.astype(np.float32))
    mean, bank = _pattern_bank(blank_specs)
    print(f"Empty-belt spectra {len(blank_specs)}", flush=True)
    rows = []
    total = len(records)
    for index, (path, split, true_name) in enumerate(records, start=1):
        scores, spec = measure_image(path)
        if scores is None:
            print(f"skip {path}", flush=True)
            continue
        fourier, nearest = pattern_distance(spec, mean, bank)
        row = {
            "path": display_path(path),
            "split": split,
            "true": true_name or "",
        }
        resin = resin_from_path(path)
        if resin:
            row["resin"] = resin
        row.update(_pack_scores(scores))
        row["fourier"] = _none_or_number(fourier)
        row["nearest_blanks"] = _none_or_number(nearest)
        rows.append(row)
        if index == total or index % 100 == 0:
            print(f"belt {index}/{total}", flush=True)
    return rows, len(blank_specs)


def _values(rows, split_ok, check_id):
    found = []
    for row in rows:
        if not split_ok(row):
            continue
        value = row.get(check_id)
        if value is None:
            continue
        found.append(value)
    return found


def defaults_for(rows):
    blanks = _is_blank_row
    objects = lambda row: row.get("split") == "validation" and row.get("true") in EXPECTED_CLASSES
    defaults = []
    for item in CHECKS:
        copy = dict(item)
        if not item["fixed"]:
            cut = suggest_cut(
                _values(rows, blanks, item["id"]),
                _values(rows, objects, item["id"]),
                item["empty_when"],
            )
            if cut is not None:
                copy["cut"] = round(float(cut), 6)
        defaults.append(copy)
    return defaults


def _is_blank_row(row):
    return row.get("true") == "empty"


def build_belt(data_dir, out, rerun):
    path = Path(out) / "belt.json"
    if path.is_file() and not rerun and belt_is_fresh(path, data_dir):
        print(f"Belt cache already at {path}", flush=True)
        return path
    records = collect_records(data_dir)
    if not records:
        raise PipelineError(f"No photos under {data_dir}")
    print(f"Belt checks on {len(records)} photos", flush=True)
    rows, blank_count = score_belt(records)
    payload = {
        "ready": True,
        "version": BELT_VERSION,
        "data_dir": str(data_dir),
        "blank_spectra": blank_count,
        "defaults": defaults_for(rows),
        "counts": _split_counts(rows),
        "frames": rows,
    }
    _write_json(path, payload)
    print(f"Wrote {path}", flush=True)
    return path


def _split_counts(rows):
    counts = {}
    for row in rows:
        split = row.get("split") or ""
        counts[split] = counts.get(split, 0) + 1
    return counts


def _stat_match(meta, model_path):
    stat = Path(model_path).stat()
    return (
        stat.st_mtime_ns == meta.get("model_mtime_ns")
        and stat.st_size == meta.get("model_size")
    )


def find_open_set(model_path, data_dir):
    path = Path(model_path)
    stems = [path.stem]
    resolved = path.resolve()
    if resolved.stem not in stems:
        stems.append(resolved.stem)
    for stem in stems:
        folder = ROOT / "analysis" / stem / "open-set"
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        if not (folder / "bank.npz").is_file() or not (folder / "scores.json").is_file():
            continue
        meta = _read_json(meta_path)
        if not meta or str(meta.get("data_dir")) != str(data_dir):
            continue
        if not _stat_match(meta, model_path):
            continue
        return folder
    return None


def model_cache_path(out, model_path):
    return Path(out) / "models" / f"{Path(model_path).stem}.json"


def model_is_fresh(path, model_path, data_dir):
    payload = _read_json(path)
    if not payload or payload.get("ready") is not True:
        return False
    if payload.get("data_dir") != str(data_dir):
        return False
    if payload.get("model") != display_path(model_path):
        return False
    return _stat_match(payload, model_path)


def _keep_frame(row, split, true_name):
    kept = {
        "path": row["path"],
        "split": split,
        "true": true_name if true_name is not None else (row.get("true") or ""),
        "pred": row.get("pred") or "",
        "confidence": row.get("confidence"),
        "cloud": row.get("cloud"),
        "call": row.get("call"),
        "knn": row.get("knn"),
        "maha": row.get("maha"),
        "disagree": bool(row.get("disagree")),
        "nearest": row.get("nearest") or "",
    }
    if row.get("resin"):
        kept["resin"] = row["resin"]
    return kept


def _ranges(frames, suggestions):
    ranges = {}
    for key in SCORE_KEYS:
        values = [
            float(row[key])
            for row in frames
            if row.get(key) is not None and np.isfinite(row[key])
        ]
        peak = max(values) if values else 0.0
        mark = suggestions.get(key)
        mark = float(mark) if mark is not None else 0.0
        top = max(peak, mark) * 1.08
        ranges[key] = round(top if top > 0 else 1.0, 6)
    return ranges


def _class_list(idx_to_class):
    return [idx_to_class[index] for index in range(len(idx_to_class))]


def _model_payload(model_path, data_dir, suggestions, suggestion_count, frames, classes):
    stat = Path(model_path).stat()
    suggestions = {
        key: None if suggestions.get(key) is None else round(float(suggestions[key]), 6)
        for key in SCORE_KEYS
    }
    return {
        "ready": True,
        "model": display_path(model_path),
        "data_dir": str(data_dir),
        "suggestions": suggestions,
        "suggestion_count": int(suggestion_count or 0),
        "ranges": _ranges(frames, suggestions),
        "counts": _split_counts(frames),
        "classes": list(classes or []),
        "model_mtime_ns": stat.st_mtime_ns,
        "model_size": stat.st_size,
        "frames": frames,
    }


def _score_extras(model, device, idx_to_class, bank, frames):
    for split, folder in (
        ("probe", ROOT / "data" / "probes"),
        ("garbage", ROOT / "data" / "Garbage"),
    ):
        paths = _image_files(folder)
        if not paths:
            continue
        print(f"{split}: {len(paths)} photos", flush=True)
        rows = score_loaded(model, device, idx_to_class, bank, paths)
        for row, path in zip(rows, paths):
            frames.append(_keep_frame(row, split, _label_for(path, split)))
    return frames


def _reuse_open_set(folder, model_path, data_dir):
    print(f"Reusing open-set bank {folder}", flush=True)
    payload = _read_json(folder / "scores.json")
    if not payload:
        raise PipelineError(f"Could not read {folder / 'scores.json'}")
    frames = []
    for row in payload.get("frames") or []:
        split = row.get("split") or ""
        if split not in ("validation", "test"):
            continue
        frames.append(_keep_frame(row, split, row.get("true") or ""))
    model, device, idx_to_class, bank = load_scorer(model_path, folder)
    if not checkpoint_matches(bank, model_path):
        raise PipelineError("This checkpoint changed after the bank was built.")
    frames = _score_extras(model, device, idx_to_class, bank, frames)
    suggestions = payload.get("suggestions") or {}
    return _model_payload(
        model_path,
        data_dir,
        suggestions,
        payload.get("suggestion_count") or 0,
        frames,
        _class_list(idx_to_class),
    )


def _build_model(model_path, data_dir):
    device = pick_device()
    print(f"Device {device}", flush=True)
    try:
        model, _class_to_idx, idx_to_class = read_checkpoint(model_path, device)
    except (OSError, ValueError) as exc:
        raise PipelineError(str(exc)) from exc
    temperature = scoring_temperature()
    counts = {}
    frames = []
    train_paths = split_paths(data_dir, "train")
    counts["train"] = len(train_paths)
    print(f"train: {len(train_paths)} photos", flush=True)
    features, _logits = embed_and_logits(model, train_paths, device, "train")
    bank = fit_bank(
        features,
        labels_for(train_paths),
        temperature,
        model_path,
        data_dir,
        counts,
    )
    print(f"Fit centroids on {len(train_paths)} train photos.", flush=True)
    for split in ("validation", "test"):
        paths = split_paths(data_dir, split)
        counts[split] = len(paths)
        print(f"{split}: {len(paths)} photos", flush=True)
        rows = score_paths(model, paths, device, idx_to_class, bank, split)
        if split == "validation":
            suggestions, correct = suggestions_from(rows)
            bank.suggestions = suggestions
            bank.suggestion_count = correct
        for row in rows:
            frames.append(_keep_frame(row, split, row.get("true") or ""))
    bank.counts = counts
    bank_dir = cache_root(data_dir) / "banks" / Path(model_path).stem
    save_bank(bank, bank_dir)
    meta = {
        "model": display_path(model_path),
        "data_dir": str(data_dir),
        "temperature": bank.temperature,
        "suggestions": {
            key: None if bank.suggestions.get(key) is None else float(bank.suggestions[key])
            for key in SCORE_KEYS
        },
        "suggestion_count": bank.suggestion_count,
        "counts": counts,
        "knn_k": bank.knn_k,
        "pca_dim": int(bank.pca_components.shape[0]),
        "model_mtime_ns": bank.model_mtime_ns,
        "model_size": bank.model_size,
    }
    (bank_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    frames = _score_extras(model, device, idx_to_class, bank, frames)
    return _model_payload(
        model_path,
        data_dir,
        bank.suggestions,
        bank.suggestion_count,
        frames,
        _class_list(idx_to_class),
    )


def build_model(model_path, data_dir, out, rerun):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise PipelineError(f"Checkpoint not found: {model_path}")
    dest = model_cache_path(out, model_path)
    if dest.is_file() and not rerun and model_is_fresh(dest, model_path, data_dir):
        print(f"Model cache already at {dest}", flush=True)
        return dest
    folder = find_open_set(model_path, data_dir)
    try:
        if folder is not None:
            payload = _reuse_open_set(folder, model_path, data_dir)
        else:
            payload = _build_model(model_path, data_dir)
    except OpenSetError as exc:
        raise PipelineError(str(exc)) from exc
    _write_json(dest, payload)
    print(f"Wrote {dest}", flush=True)
    return dest


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} is not on disk.")
    out = cache_root(data_dir)
    try:
        build_belt(data_dir, out, args.rerun)
        if args.belt_only:
            return
        if not args.model:
            return
        for model_path in args.model:
            build_model(model_path, data_dir, out, args.rerun)
    except (PipelineError, FileNotFoundError, OpenSetError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
