"""Open-set scores from the pooled ResNeXt vector.

Fit on train only: a metal centroid, a plastic centroid, a 10-neighbor
index, and a 32-d Mahalanobis score with one tied covariance. Validation
picks the suggested distance mark. Test is scored after that and is not
used to move it.

This does not write decide.py. ``others`` here is a lab preview for a
photo that sits outside both clouds. It is not the stage 1 OTHERS resin,
and it is not the 0.85 hold.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from analyze import forward_hooks
from checkpoints import lab_class, read_checkpoint, resin_head
from confidence_report import pick_device, split_paths
from decide import scaled_probabilities, scoring_temperature
from preprocess import (
    class_key,
    inference_transform,
    load_rgb,
    mapped_class,
    true_class_from_path,
)
from trashnet import EXPECTED_CLASSES


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
KNN_K = 10
PCA_DIM = 32
BATCH_SIZE = 16
SUGGEST_QUANTILE = 0.99
SCORE_KEYS = ("cloud", "call", "knn", "maha")


class OpenSetError(Exception):
    pass


class Bank:
    def __init__(
        self,
        class_names,
        centroids,
        train_vectors,
        knn_k,
        pca_mean,
        pca_components,
        pca_means,
        pca_precision,
        temperature,
        suggestions,
        suggestion_count,
        model_mtime_ns,
        model_size,
        model_path,
        data_dir,
        counts,
    ):
        self.class_names = list(class_names)
        self.model_path = Path(model_path)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.train_vectors = np.asarray(train_vectors, dtype=np.float32)
        self.knn_k = int(knn_k)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float32)
        self.pca_components = np.asarray(pca_components, dtype=np.float32)
        self.pca_means = np.asarray(pca_means, dtype=np.float32)
        self.pca_precision = np.asarray(pca_precision, dtype=np.float32)
        self.temperature = float(temperature)
        self.suggestions = dict(suggestions)
        self.suggestion_count = int(suggestion_count)
        self.model_mtime_ns = int(model_mtime_ns)
        self.model_size = int(model_size)
        self.data_dir = data_dir
        self.counts = dict(counts)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit open-set distances on train, then score validation and test."
        ),
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
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Folder for the bank and scores. Default: analysis/<stem>/open-set.",
    )
    parser.add_argument(
        "--rerun-test",
        action="store_true",
        help="Score the locked test split again and replace this folder.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=None,
        help="Score these photos with the saved bank and print JSON.",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="Bank folder for --images. Default: the --out folder.",
    )
    return parser.parse_args()


def default_out(model_path):
    return Path("analysis") / Path(model_path).stem / "open-set"


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def l2_normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


# A probe folder named other is the unknown class. It is not the stage 1
# OTHERS resin, which stays metal when the photo sits in metal/ or plastic/.
_PROBE_FOLDERS = {
    "other": "other",
    "others": "other",
    "unknown": "other",
    "novel": "other",
    "metal": "metal",
    "nonplastic": "metal",
    "plastic": "plastic",
    "hdpe": "plastic",
    "pet": "plastic",
    "pp": "plastic",
}


def probe_true_class(path):
    """Label a photo you bring from the nearest class folder.

    ``data/probes/other`` stays unknown. A ``_OTHERS`` file inside
    ``metal/`` stays metal, because that folder is the nearer label.
    """
    for part in reversed(Path(path).parts[:-1]):
        label = _PROBE_FOLDERS.get(class_key(part))
        if label is not None:
            return label
    return true_class_from_path(path)


def labels_for(paths):
    names = []
    for path in paths:
        true_name = true_class_from_path(path)
        folder = mapped_class(path.parent.name)
        if true_name is None or folder is None or true_name != folder:
            raise OpenSetError(f"No class for {path}")
        if true_name not in EXPECTED_CLASSES:
            raise OpenSetError(f"Unexpected class on {path}")
        names.append(true_name)
    return names


def embed_and_logits(model, paths, device, split_name):
    """One forward per photo: logits plus the avgpool vector.

    The hook is the same one analyze.extract_avgpool_features uses.
    Logits come from that same forward so the split is not run twice.
    """
    if not paths:
        width = int(model.fc.in_features)
        empty = np.zeros((0, width), dtype=np.float32)
        return empty, torch.zeros((0, len(EXPECTED_CLASSES)))
    transform = inference_transform()
    features = []
    logits = []
    with forward_hooks({"avgpool": model.avgpool}) as captured:
        for start in range(0, len(paths), BATCH_SIZE):
            chunk = paths[start : start + BATCH_SIZE]
            batch = torch.stack(
                [transform(load_rgb(path)) for path in chunk]
            ).to(device)
            with torch.inference_mode():
                out = model(batch)
            logits.append(out.detach().cpu())
            features.append(captured["avgpool"].flatten(1).cpu().numpy())
            done = min(start + BATCH_SIZE, len(paths))
            print(f"  {split_name} batch {done}/{len(paths)}", flush=True)
    return np.concatenate(features, axis=0), torch.cat(logits, dim=0)


def _invert_cov(cov):
    eye = np.eye(cov.shape[0], dtype=np.float64)
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1):
        precision = np.linalg.pinv(cov + ridge * eye)
        if np.isfinite(precision).all():
            return precision.astype(np.float32)
    raise OpenSetError("Covariance fit failed.")


def fit_bank(features, label_names, temperature, model_path, data_dir, counts):
    vectors = l2_normalize(features)
    names = list(EXPECTED_CLASSES)
    centroids = []
    for name in names:
        mask = np.array([label == name for label in label_names])
        if int(mask.sum()) == 0:
            raise OpenSetError(f"Train has no {name} photos.")
        mean = vectors[mask].mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
        centroids.append(mean.astype(np.float32))
    centroids = np.stack(centroids, axis=0)

    components = min(PCA_DIM, vectors.shape[0] - 1, vectors.shape[1])
    if components < 2:
        raise OpenSetError("Need more than two train photos to fit PCA.")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
    projected = pca.fit_transform(vectors)
    means = []
    pieces = []
    for name in names:
        mask = np.array([label == name for label in label_names])
        mean = projected[mask].mean(axis=0)
        means.append(mean)
        pieces.append(projected[mask] - mean)
    residuals = np.concatenate(pieces, axis=0)
    cov = np.cov(residuals, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    precision = _invert_cov(np.atleast_2d(cov))
    stat = Path(model_path).stat()
    return Bank(
        class_names=names,
        centroids=centroids,
        train_vectors=vectors,
        knn_k=min(KNN_K, vectors.shape[0]),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_means=np.stack(means, axis=0).astype(np.float32),
        pca_precision=precision,
        temperature=temperature,
        suggestions={},
        suggestion_count=0,
        model_mtime_ns=stat.st_mtime_ns,
        model_size=stat.st_size,
        model_path=model_path,
        data_dir=str(data_dir),
        counts=counts,
    )


def save_bank(bank, folder):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / "bank.npz",
        centroids=bank.centroids,
        train_vectors=bank.train_vectors,
        pca_mean=bank.pca_mean,
        pca_components=bank.pca_components,
        pca_means=bank.pca_means,
        pca_precision=bank.pca_precision,
        class_names=np.array(bank.class_names),
    )


def load_bank(folder):
    folder = Path(folder)
    npz_path = folder / "bank.npz"
    meta_path = folder / "meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise OpenSetError(f"No open-set bank in {folder}.")
    blob = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Bank(
        class_names=[str(name) for name in blob["class_names"].tolist()],
        centroids=blob["centroids"],
        train_vectors=blob["train_vectors"],
        knn_k=int(meta["knn_k"]),
        pca_mean=blob["pca_mean"],
        pca_components=blob["pca_components"],
        pca_means=blob["pca_means"],
        pca_precision=blob["pca_precision"],
        temperature=float(meta["temperature"]),
        suggestions={key: meta["suggestions"].get(key) for key in SCORE_KEYS},
        suggestion_count=int(meta.get("suggestion_count") or 0),
        model_mtime_ns=int(meta["model_mtime_ns"]),
        model_size=int(meta["model_size"]),
        model_path=meta.get("model") or "",
        data_dir=meta.get("data_dir") or "",
        counts=meta.get("counts") or {},
    )


def knn_distance(queries, train_vectors, k):
    if len(queries) == 0:
        return np.zeros((0,), dtype=np.float32)
    k = min(int(k), train_vectors.shape[0])
    train64 = np.asarray(train_vectors, dtype=np.float64)
    distances = np.empty(len(queries), dtype=np.float32)
    step = 128
    for start in range(0, len(queries), step):
        chunk = np.asarray(queries[start : start + step], dtype=np.float64)
        sims = chunk @ train64.T
        index = np.argpartition(sims, -k, axis=1)[:, -k:]
        top = np.take_along_axis(sims, index, axis=1)
        distances[start : start + step] = 1.0 - top.mean(axis=1)
    return distances


def mahalanobis(queries, bank):
    if len(queries) == 0:
        return np.zeros((0,), dtype=np.float32)
    centered = queries.astype(np.float64) - bank.pca_mean.astype(np.float64)
    projected = centered @ bank.pca_components.astype(np.float64).T
    precision = bank.pca_precision.astype(np.float64)
    means = bank.pca_means.astype(np.float64)
    columns = []
    for mean in means:
        diff = projected - mean
        quadratic = np.einsum("ij,jk,ik->i", diff, precision, diff)
        columns.append(np.sqrt(np.maximum(quadratic, 0)))
    return np.min(np.stack(columns, axis=0), axis=0).astype(np.float32)


def measure(features, logits, bank, idx_to_class):
    vectors = l2_normalize(features)
    if vectors.shape[1:] != bank.centroids.shape[1:]:
        raise OpenSetError(
            "Feature width does not match this bank. Build it again."
        )
    head_names = set(idx_to_class.values())
    if head_names != set(bank.class_names):
        resin_on_belt = (
            resin_head(idx_to_class)
            and set(bank.class_names) == set(EXPECTED_CLASSES)
        )
        if not resin_on_belt:
            raise OpenSetError(
                "This checkpoint's classes do not match the bank."
            )
    probs = scaled_probabilities(logits, bank.temperature)
    confidence, pred_index = torch.max(probs, dim=-1)
    pred_index = pred_index.cpu().numpy()
    raw = [idx_to_class[int(index)] for index in pred_index]
    names = [lab_class(name) for name in raw]
    resins = raw if resin_head(idx_to_class) else [""] * len(raw)
    class_index = {name: index for index, name in enumerate(bank.class_names)}
    # float64: a float32 dot of these wide vectors warns and can go non-finite.
    dists = 1.0 - (
        vectors.astype(np.float64) @ bank.centroids.astype(np.float64).T
    )
    nearest_index = np.argmin(dists, axis=1)
    rows = np.arange(len(vectors))
    cloud = dists[rows, nearest_index]
    call_index = np.array([class_index[name] for name in names])
    call = dists[rows, call_index]
    return {
        "pred": names,
        "resin": resins,
        "confidence": confidence.detach().cpu().numpy().astype(np.float32),
        "cloud": cloud.astype(np.float32),
        "call": call.astype(np.float32),
        "knn": knn_distance(vectors, bank.train_vectors, bank.knn_k),
        "maha": mahalanobis(vectors, bank),
        "disagree": nearest_index != call_index,
        "nearest": [bank.class_names[int(index)] for index in nearest_index],
    }


def _round(value):
    value = float(value)
    if not np.isfinite(value):
        raise OpenSetError("A distance was not finite.")
    return round(value, 6)


def pack_row(path, split, true_name, measured, index):
    row = {
        "split": split,
        "path": display_path(path),
        "true": true_name,
        "pred": measured["pred"][index],
        "confidence": _round(measured["confidence"][index]),
        "cloud": _round(measured["cloud"][index]),
        "call": _round(measured["call"][index]),
        "knn": _round(measured["knn"][index]),
        "maha": _round(measured["maha"][index]),
        "disagree": bool(measured["disagree"][index]),
        "nearest": measured["nearest"][index],
    }
    resin = measured["resin"][index]
    if resin:
        row["resin"] = resin
    return row


def score_paths(model, paths, device, idx_to_class, bank, split_name):
    features, logits = embed_and_logits(model, paths, device, split_name)
    measured = measure(features, logits, bank, idx_to_class)
    rows = []
    for index, path in enumerate(paths):
        true_name = probe_true_class(path) if split_name == "probe" else None
        rows.append(pack_row(path, split_name, true_name, measured, index))
    if split_name != "probe":
        true_names = labels_for(paths)
        for row, true_name in zip(rows, true_names):
            row["true"] = true_name
    return rows


def _quantile(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.quantile(values, SUGGEST_QUANTILE))


def suggestions_from(rows):
    correct = [
        row
        for row in rows
        if row["true"] is not None and row["pred"] == row["true"]
    ]
    found = {}
    for key in SCORE_KEYS:
        found[key] = _quantile([row[key] for row in correct])
    return found, len(correct)


def _axis_max(values, suggestion):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    peak = float(values.max()) if values.size else 0.0
    mark = float(suggestion) if suggestion is not None else 0.0
    top = max(peak, mark) * 1.08
    if top <= 0:
        return 1.0
    return top


def write_report(folder, bank, frames, data_dir):
    ranges = {}
    for key in SCORE_KEYS:
        ranges[key] = _round(
            _axis_max((row[key] for row in frames), bank.suggestions.get(key))
        )
    payload = {
        "ready": True,
        "model": display_path(bank.model_path),
        "data_dir": str(data_dir),
        "temperature": _round(bank.temperature),
        "suggestions": {
            key: None if bank.suggestions.get(key) is None else _round(bank.suggestions[key])
            for key in SCORE_KEYS
        },
        "suggestion_count": bank.suggestion_count,
        "ranges": ranges,
        "counts": bank.counts,
        "knn_k": bank.knn_k,
        "pca_dim": int(bank.pca_components.shape[0]),
        "model_mtime_ns": bank.model_mtime_ns,
        "model_size": bank.model_size,
        "frames": frames,
    }
    (folder / "scores.json").write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    meta = {
        "model": payload["model"],
        "data_dir": payload["data_dir"],
        "temperature": payload["temperature"],
        "suggestions": payload["suggestions"],
        "suggestion_count": bank.suggestion_count,
        "counts": bank.counts,
        "knn_k": bank.knn_k,
        "pca_dim": payload["pca_dim"],
        "model_mtime_ns": bank.model_mtime_ns,
        "model_size": bank.model_size,
    }
    (folder / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Open set",
        "",
        "Lab preview. This does not change the hold in `decide.py`.",
        "Others here means the pooled vector sits outside the metal and",
        "plastic clouds. It is not the stage 1 OTHERS resin.",
        "",
        f"Fit on train in `{data_dir}` ({bank.counts.get('train', 0)} photos).",
        f"Temperature {bank.temperature:.6f}.",
        f"k-NN uses {bank.knn_k} neighbors. Mahalanobis uses "
        f"{payload['pca_dim']} principal components and one tied covariance.",
        "",
        "The mark is the 99th percentile of each distance on validation",
        "photos the model already got right "
        f"({bank.suggestion_count}). Test did not pick it.",
        "",
    ]
    if any(row.get("resin") for row in frames):
        lines.extend(
            [
                "The head is HDPE, PET, and PP. Each call is scored as",
                "plastic. This file has no metal class. Confidence under",
                "0.85 is their review bucket. Here that cutoff is uncertain.",
                "",
            ]
        )
    for key in SCORE_KEYS:
        value = bank.suggestions.get(key)
        shown = "none" if value is None else f"{value:.6f}"
        lines.append(f"- {key}: {shown}")
    lines.extend(
        [
            "",
            f"Validation: {bank.counts.get('validation', 0)}.",
            f"Test: {bank.counts.get('test', 0)}.",
            "",
        ]
    )
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    cloud = bank.suggestions.get("cloud")
    cloud_text = "none" if cloud is None else f"{cloud:.6f}"
    (folder / "test_scored.txt").write_text(
        "Scored once after the bank was fit on train.\n"
        "The distance mark is the 99th percentile of cloud distance on\n"
        "validation photos the model got right. Test did not pick it.\n"
        f"cloud={cloud_text}\n",
        encoding="utf-8",
    )


def publish(staging, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = out.parent / f"{out.name}-previous"
    if previous.exists():
        shutil.rmtree(previous)
    if out.exists():
        out.rename(previous)
    staging.rename(out)
    if previous.exists():
        shutil.rmtree(previous)


def build(model_path, data_dir, out, rerun_test):
    out = Path(out)
    marker = out / "test_scored.txt"
    if marker.is_file() and not rerun_test:
        raise OpenSetError(
            f"{marker} is already there. Pass --rerun-test to score "
            "the locked test split again."
        )
    staging = out.parent / f"{out.name}-next"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device {device}", flush=True)
    try:
        model, _class_to_idx, idx_to_class = read_checkpoint(model_path, device)
    except (OSError, ValueError) as exc:
        raise OpenSetError(str(exc)) from exc
    temperature = scoring_temperature()
    counts = {}
    frames = []
    bank = None
    try:
        for split in ("train", "validation", "test"):
            paths = split_paths(data_dir, split)
            counts[split] = len(paths)
            print(f"{split}: {len(paths)} photos", flush=True)
            if split == "train":
                features, _logits = embed_and_logits(
                    model, paths, device, split
                )
                label_names = labels_for(paths)
                bank = fit_bank(
                    features,
                    label_names,
                    temperature,
                    model_path,
                    data_dir,
                    counts,
                )
                print(
                    f"Fit centroids on {len(paths)} train photos.",
                    flush=True,
                )
                continue
            rows = score_paths(
                model, paths, device, idx_to_class, bank, split
            )
            if split == "validation":
                suggestions, correct = suggestions_from(rows)
                bank.suggestions = suggestions
                bank.suggestion_count = correct
                cloud = suggestions.get("cloud")
                cloud_text = "none" if cloud is None else f"{cloud:.6f}"
                print(
                    f"Validation 99% cloud distance {cloud_text} "
                    f"from {correct} correct photos.",
                    flush=True,
                )
            frames.extend(rows)
        bank.counts = counts
        save_bank(bank, staging)
        write_report(staging, bank, frames, data_dir)
        publish(staging, out)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Wrote {out}", flush=True)


def checkpoint_matches(bank, model_path):
    stat = Path(model_path).stat()
    return (
        stat.st_mtime_ns == bank.model_mtime_ns
        and stat.st_size == bank.model_size
    )


def load_scorer(model_path, bank_dir):
    """Load the checkpoint and bank. The caller keeps them for later photos."""
    bank = load_bank(bank_dir)
    if not checkpoint_matches(bank, model_path):
        raise OpenSetError(
            "This checkpoint changed after the bank was built. Rerun first."
        )
    device = pick_device()
    try:
        model, _class_to_idx, idx_to_class = read_checkpoint(model_path, device)
    except (OSError, ValueError) as exc:
        raise OpenSetError(str(exc)) from exc
    return model, device, idx_to_class, bank


def score_loaded(model, device, idx_to_class, bank, image_paths):
    paths = [Path(path) for path in image_paths]
    for path in paths:
        if not path.is_file():
            raise OpenSetError(f"Not a photo: {path}")
    return score_paths(model, paths, device, idx_to_class, bank, "probe")


def score_image_list(model_path, bank_dir, image_paths):
    model, device, idx_to_class, bank = load_scorer(model_path, bank_dir)
    return score_loaded(model, device, idx_to_class, bank, image_paths)


def main():
    args = parse_args()
    try:
        if args.images:
            bank_dir = args.bank or args.out or default_out(args.model)
            rows = score_image_list(args.model, bank_dir, args.images)
            print(json.dumps(rows, indent=2))
            return
        out = args.out or default_out(args.model)
        build(args.model, args.data_dir, out, args.rerun_test)
    except (OpenSetError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
