"""Belt presence gate.

A frame is empty when no object is large enough to sort. Area is the
largest blob after the slot grid is averaged away. A second score is
cosine distance to a centroid fit on audited blank frames.

Validation picks both cuts. Test is scored once and does not move them.
This does not write decide.py and does not change the 0.85 hold.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from confidence_report import IMAGE_SUFFIXES, load_model, pick_device, split_paths
from open_set import display_path, embed_and_logits, l2_normalize, publish
from preprocess import readable_image_paths, true_class_from_path


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
DEFAULT_BLANKS = Path("data/probes/blanks")

# Measured on data/probes/blanks against data_belt_stage1 validation.
# Width 64 averages a belt slot to about a pixel on a full camera frame.
# Deviation 28 left a gap: audited blanks topped out at 158, and the
# cut that caught all of them held 4 of 805 validation photos.
DOWNSCALE_WIDTH = 64
DEVIATION_THRESHOLD = 28
SIGMA_DIVISOR = 40.0
ILLUMINATION_WIDTH = 480

HARD_HOLD_CAP = 0.01
AREA_LOOSE_QUANTILE = 0.05
BLANK_CUT_QUANTILE = 0.01
OUTLIER_MAD = 3.0


class PresenceError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a belt presence gate on blanks and validation.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_belt_stage1"),
        help="Dataset root with train/, validation/, and test/.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Checkpoint path. Used for the blank centroid only.",
    )
    parser.add_argument(
        "--blanks",
        type=Path,
        default=DEFAULT_BLANKS,
        help="Folder of empty-belt photos.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Folder for the scores. Default: analysis/<stem>/presence.",
    )
    parser.add_argument(
        "--rerun-test",
        action="store_true",
        help="Score the locked test split again and replace this folder.",
    )
    return parser.parse_args()


def default_out(model_path):
    return Path("analysis") / Path(model_path).stem / "presence"


def image_paths(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise PresenceError(f"No blank folder at {folder}")
    found = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths = readable_image_paths(found)
    if not paths:
        raise PresenceError(f"No readable photos in {folder}")
    return paths


def foreground_parts(path):
    """Blob areas after the belt slots have been averaged away.

    A wide Gaussian takes out a lighting swing. The resize is small
    enough that a moving slot becomes about a pixel and drops into the
    median. What is left, and larger than the deviation threshold, is
    treated as an object. Areas come back largest first.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise PresenceError(f"Could not read {path}")
    gray = gray.astype(np.float32)
    height, width = gray.shape
    target = ILLUMINATION_WIDTH if width > ILLUMINATION_WIDTH else width
    scale = target / float(width)
    resized_height = max(1, int(round(height * scale)))
    if target != width:
        working = cv2.resize(
            gray,
            (target, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = gray
    sigma = max((width / SIGMA_DIVISOR) * scale, 1.5)
    blur = cv2.GaussianBlur(working, (0, 0), sigmaX=sigma, sigmaY=sigma)
    norm = np.clip(working / np.maximum(blur, 1.0) * 255.0, 0, 255)
    out_height = max(1, int(round(norm.shape[0] * DOWNSCALE_WIDTH / norm.shape[1])))
    tiny = cv2.resize(
        norm,
        (DOWNSCALE_WIDTH, out_height),
        interpolation=cv2.INTER_AREA,
    )
    median = float(np.median(tiny))
    mask = (np.abs(tiny - median) >= DEVIATION_THRESHOLD).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if count <= 1:
        return []
    areas = [int(value) for value in stats[1:, cv2.CC_STAT_AREA]]
    areas.sort(reverse=True)
    return areas


def foreground_area(path):
    """Largest blob after the belt slots have been averaged away."""
    parts = foreground_parts(path)
    if not parts:
        return 0
    return parts[0]


def area_for(paths, split_name):
    areas = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        areas.append(foreground_area(path))
        if index == total or index % 100 == 0:
            print(f"  {split_name} area {index}/{total}", flush=True)
    return np.asarray(areas, dtype=np.int32)


def high_outliers(areas):
    """True where area sits above the blank median plus 3 MAD."""
    areas = np.asarray(areas, dtype=np.float64)
    median = float(np.median(areas))
    mad = float(np.median(np.abs(areas - median)))
    fence = median + OUTLIER_MAD * mad
    return areas > fence, fence


def _round(value):
    value = float(value)
    if not np.isfinite(value):
        raise PresenceError("A presence score was not finite.")
    return round(value, 6)


def choose_area_cut(blank_areas, object_areas):
    """Highest blank catch that holds at most 1% of validation.

    When no cut does both, the saved cut is the best overlap
    (catch minus hold) and ``overlap`` is true. A feasible cut that
    still misses most blanks is an overlap too.
    """
    blank_areas = np.asarray(blank_areas, dtype=np.float64)
    object_areas = np.asarray(object_areas, dtype=np.float64)
    if blank_areas.size == 0:
        raise PresenceError("No audited blank frames to fit the area cut.")
    if object_areas.size == 0:
        raise PresenceError("Validation has no photos to fit the area cut.")
    levels = np.unique(np.concatenate([blank_areas, object_areas]))
    n_blank = int(blank_areas.size)
    n_object = int(object_areas.size)
    best_ok = None
    best_overlap = None
    for level in levels:
        # Areas are whole pixels. area < level + 1 catches this level.
        cut = float(level) + 1.0
        caught = int(np.sum(blank_areas < cut))
        held = int(np.sum(object_areas < cut))
        catch = caught / n_blank
        hold = held / n_object
        if hold <= HARD_HOLD_CAP:
            key = (catch, -hold, -cut)
            if best_ok is None or key > best_ok[0]:
                best_ok = (key, cut, catch, hold, caught, held)
        score = catch - hold
        overlap_key = (score, catch, -hold, -cut)
        if best_overlap is None or overlap_key > best_overlap[0]:
            best_overlap = (overlap_key, cut, catch, hold, caught, held)
    if best_ok is not None:
        _key, cut, catch, hold, caught, held = best_ok
        return {
            "cut": cut,
            "catch": catch,
            "hold": hold,
            "caught": caught,
            "held": held,
            "capped": True,
            "overlap": catch <= 0.5,
        }
    _key, cut, catch, hold, caught, held = best_overlap
    return {
        "cut": cut,
        "catch": catch,
        "hold": hold,
        "caught": caught,
        "held": held,
        "capped": False,
        "overlap": True,
    }


def fit_centroid(features):
    vectors = l2_normalize(features)
    if len(vectors) == 0:
        raise PresenceError("No audited blanks to fit the centroid.")
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        raise PresenceError("Blank centroid has no length.")
    return (mean / norm).astype(np.float32)


def blank_distance(features, centroid):
    vectors = l2_normalize(features)
    dots = vectors.astype(np.float64) @ centroid.astype(np.float64)
    return (1.0 - dots).astype(np.float32)


def decide_row(area, distance, area_cut, area_loose, blank_cut):
    if area < area_cut:
        return True, "hard"
    if area < area_loose and distance < blank_cut:
        return True, "soft"
    return False, "object"


def _range(areas):
    areas = np.asarray(areas)
    return {
        "min": int(areas.min()),
        "median": int(round(float(np.median(areas)))),
        "max": int(areas.max()),
    }


def _frame(path, split, true_name, area, distance, outlier, empty, reason):
    return {
        "split": split,
        "path": display_path(path),
        "true": true_name,
        "area": int(area),
        "blank": _round(distance),
        "outlier": bool(outlier),
        "empty": bool(empty),
        "reason": reason,
    }


def overlap_note(choice):
    if not choice["overlap"]:
        return ""
    if choice["capped"]:
        return (
            "The area cut holds at most 1% of validation and still "
            "misses most blank frames."
        )
    return (
        "No area cut catches most blank frames while holding at most "
        "1% of validation. The saved cut is the best overlap."
    )


def write_report(folder, payload):
    (folder / "scores.json").write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    meta = {key: value for key, value in payload.items() if key != "frames"}
    (folder / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    rates = payload["rates"]
    blank_range = payload["area_ranges"]["blank"]
    val_range = payload["area_ranges"]["validation"]
    lines = [
        "# Belt presence",
        "",
        "Lab preview. This does not change the hold in `decide.py`.",
        "Empty means the belt has nothing large enough to sort.",
        "",
        f"Downscale width {payload['downscale_width']}. "
        f"Deviation threshold {payload['deviation_threshold']}. "
        f"Gaussian sigma is 1/{int(payload['sigma_divisor'])} of the frame width. "
        f"Frames wider than {payload['illumination_width']} are resized "
        "to that width before the blur.",
        "",
        f"Audited blank area {blank_range['min']}–{blank_range['max']} "
        f"(median {blank_range['median']}).",
        f"Validation area {val_range['min']}–{val_range['max']} "
        f"(median {val_range['median']}).",
        "",
        f"Fit the blank centroid on {rates['blank_n']} photos in "
        f"`{payload['blanks_dir']}`. "
        f"{rates['outlier_n']} blank "
        f"{'photo was' if rates['outlier_n'] == 1 else 'photos were'} "
        "left out, above the median plus 3 MAD.",
        "",
        f"Area cut {payload['area_cut']:.0f} "
        f"(hard empty). Loose area {payload['area_loose']:.4f} "
        f"is the 5th percentile of validation. "
        f"Blank distance cut {payload['blank_cut']:.6f} "
        "is the 1st percentile of validation distances.",
        "Test did not pick them.",
        "",
    ]
    note = payload.get("note") or ""
    if note:
        lines.extend([note, ""])
    lines.extend(
        [
            f"Blanks caught: {rates['blank_caught']}/{rates['blank_n']}.",
            f"Validation held: {rates['validation_held']}/{rates['validation_n']} "
            f"({rates['validation_hard']} by area, {rates['validation_soft']} by area and blank).",
            f"Test held: {rates['test_held']}/{rates['test_n']}.",
            "",
        ]
    )
    if payload["outliers"]:
        lines.append("Left out of the centroid:")
        lines.append("")
        for row in payload["outliers"]:
            lines.append(f"- {row['path']} area {row['area']}")
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "test_scored.txt").write_text(
        "Scored once after validation picked the cuts.\n"
        "The area cut is the highest blank catch that holds at most\n"
        "1% of validation. Test did not pick it.\n"
        f"area_cut={payload['area_cut']:.0f}\n"
        f"area_loose={payload['area_loose']:.6f}\n"
        f"blank_cut={payload['blank_cut']:.6f}\n",
        encoding="utf-8",
    )


def build(model_path, data_dir, blanks_dir, out, rerun_test):
    out = Path(out)
    marker = out / "test_scored.txt"
    if marker.is_file() and not rerun_test:
        raise PresenceError(
            f"{marker} is already there. Pass --rerun-test to score "
            "the locked test split again."
        )
    model_path = Path(model_path)
    if not model_path.is_file():
        raise PresenceError(f"Checkpoint not found: {model_path}")
    staging = out.parent / f"{out.name}-next"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        blank_paths = image_paths(blanks_dir)
        validation_paths = split_paths(data_dir, "validation")
        print(f"blanks: {len(blank_paths)} photos", flush=True)
        print(f"validation: {len(validation_paths)} photos", flush=True)
        blank_areas = area_for(blank_paths, "blanks")
        validation_areas = area_for(validation_paths, "validation")
        outlier, fence = high_outliers(blank_areas)
        audited = ~outlier
        dropped = int(outlier.sum())
        print(
            f"Left out {dropped} blank "
            f"{'photo' if dropped == 1 else 'photos'} above area {fence:.1f}.",
            flush=True,
        )
        choice = choose_area_cut(blank_areas[audited], validation_areas)
        area_cut = choice["cut"]
        area_loose = float(np.quantile(validation_areas, AREA_LOOSE_QUANTILE))
        print(
            f"Area cut {area_cut:.0f} catches {choice['caught']}/{int(audited.sum())} "
            f"blanks and holds {choice['held']}/{len(validation_paths)} "
            "validation photos by area.",
            flush=True,
        )
        device = pick_device()
        print(f"Device {device}", flush=True)
        model, _class_to_idx, _idx_to_class = load_model(model_path, device)
        blank_features, _blank_logits = embed_and_logits(
            model, blank_paths, device, "blanks"
        )
        validation_features, _validation_logits = embed_and_logits(
            model, validation_paths, device, "validation"
        )
        centroid = fit_centroid(blank_features[audited])
        blank_distances = blank_distance(blank_features, centroid)
        validation_distances = blank_distance(validation_features, centroid)
        blank_cut = float(np.quantile(validation_distances, BLANK_CUT_QUANTILE))
        print(f"Blank distance cut {blank_cut:.6f}.", flush=True)

        test_paths = split_paths(data_dir, "test")
        print(f"test: {len(test_paths)} photos", flush=True)
        test_areas = area_for(test_paths, "test")
        test_features, _test_logits = embed_and_logits(
            model, test_paths, device, "test"
        )
        test_distances = blank_distance(test_features, centroid)

        frames = []
        outliers = []
        missed = []
        held = []
        blank_caught = 0
        for index, path in enumerate(blank_paths):
            distance = blank_distances[index]
            area = int(blank_areas[index])
            if outlier[index]:
                row = _frame(path, "blanks", "empty", area, distance, True, False, "outlier")
                frames.append(row)
                outliers.append(
                    {"path": row["path"], "area": area, "blank": row["blank"]}
                )
                continue
            empty, reason = decide_row(
                area, distance, area_cut, area_loose, blank_cut
            )
            row = _frame(path, "blanks", "empty", area, distance, False, empty, reason)
            frames.append(row)
            if empty:
                blank_caught += 1
            else:
                missed.append(
                    {"path": row["path"], "area": area, "blank": row["blank"]}
                )

        def count_split(paths, areas, distances, split):
            hard = 0
            soft = 0
            rows_held = []
            for index, path in enumerate(paths):
                true_name = true_class_from_path(path)
                if true_name not in ("metal", "plastic"):
                    raise PresenceError(f"No class for {path}")
                area = int(areas[index])
                distance = distances[index]
                empty, reason = decide_row(
                    area, distance, area_cut, area_loose, blank_cut
                )
                row = _frame(
                    path, split, true_name, area, distance, False, empty, reason
                )
                frames.append(row)
                if reason == "hard":
                    hard += 1
                elif reason == "soft":
                    soft += 1
                if empty and split == "validation":
                    rows_held.append(
                        {
                            "path": row["path"],
                            "true": true_name,
                            "area": area,
                            "blank": row["blank"],
                            "reason": reason,
                        }
                    )
            return hard, soft, rows_held

        validation_hard, validation_soft, held = count_split(
            validation_paths, validation_areas, validation_distances, "validation"
        )
        test_hard, test_soft, _test_held = count_split(
            test_paths, test_areas, test_distances, "test"
        )
        stat = model_path.stat()
        blank_n = int(audited.sum())
        payload = {
            "ready": True,
            "model": display_path(model_path),
            "data_dir": str(data_dir),
            "blanks_dir": display_path(blanks_dir),
            "downscale_width": DOWNSCALE_WIDTH,
            "deviation_threshold": DEVIATION_THRESHOLD,
            "sigma_divisor": SIGMA_DIVISOR,
            "illumination_width": ILLUMINATION_WIDTH,
            "area_cut": _round(area_cut),
            "area_loose": _round(area_loose),
            "blank_cut": _round(blank_cut),
            "overlap": bool(choice["overlap"]),
            "note": overlap_note(choice),
            "area_ranges": {
                "blank": _range(blank_areas[audited]),
                "validation": _range(validation_areas),
            },
            "rates": {
                "blank_caught": blank_caught,
                "blank_n": blank_n,
                "outlier_n": int(outlier.sum()),
                "validation_held": validation_hard + validation_soft,
                "validation_hard": validation_hard,
                "validation_soft": validation_soft,
                "validation_n": len(validation_paths),
                "test_held": test_hard + test_soft,
                "test_hard": test_hard,
                "test_soft": test_soft,
                "test_n": len(test_paths),
            },
            "outliers": outliers,
            "missed": missed,
            "held": held,
            "model_mtime_ns": stat.st_mtime_ns,
            "model_size": stat.st_size,
            "frames": frames,
        }
        write_report(staging, payload)
        publish(staging, out)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Wrote {out}", flush=True)


def main():
    args = parse_args()
    try:
        out = args.out or default_out(args.model)
        build(args.model, args.data_dir, args.blanks, out, args.rerun_test)
    except (PresenceError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
