import argparse
import random
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torch.nn import functional as F

from checkpoints import lab_class, read_checkpoint
from preprocess import (
    INPUT_SIZE,
    RESIN_TAGS,
    inference_transform,
    load_rgb,
    readable_image_paths,
    resin_from_path,
    tensor_to_display,
    true_class_from_path,
)
from trashnet import (
    EXPECTED_CLASSES,
    assert_split_layout,
    split_dirs_for,
)


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
LEGACY_MODEL_PATH = Path("resnext50_metal_plastic.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALL_STEPS = ("maps", "cam", "stats", "pca", "filters")
LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")
STATS_PER_CLASS = 16
MAPS_PER_CLASS = 2
CORRECT_CAM_COUNT = 3
CAM_ALL_LIMIT = 32
RANK_CHANNELS = 20
FEATURE_BATCH = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ResNeXt-50 internals: feature maps, "
            "Grad-CAM, layer stats, PCA, and filter ranking."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root with train/validation/test "
            "class folders. Used when --images is omitted."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Photo files or folders to inspect. "
            "Folders are searched recursively. "
            "With 32 or fewer images, Grad-CAM runs on each one, "
            "including a photo that is not in a class folder. "
            "Skips the test-split layout."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=9,
        help=(
            "How many photos to use for maps, filters, "
            "and correct Grad-CAM samples (default: 9)."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every image for sampled steps instead of --count.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis"),
        help="Folder for PNGs and ANALYSIS_REPORT.md (default: analysis).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save/print only; do not open windows.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Defaults to "
            "models/resnext50_metal_plastic.pt, then "
            "resnext50_metal_plastic.pt in the current "
            "directory."
        ),
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=",".join(ALL_STEPS),
        help=(
            "Comma-separated subset of "
            "maps,cam,stats,pca,filters (default: all)."
        ),
    )
    parser.add_argument(
        "--mistakes-only",
        action="store_true",
        help="Limit Grad-CAM figures to misclassified labeled images.",
    )
    return parser.parse_args()


def parse_steps(raw):
    steps = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = steps - set(ALL_STEPS)
    if unknown:
        raise SystemExit(
            "Unknown --steps: " + ", ".join(sorted(unknown))
        )
    if not steps:
        raise SystemExit("Pass at least one step in --steps.")
    return steps


def resolve_model_path(requested):
    if requested is not None:
        return requested
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return DEFAULT_MODEL_PATH


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_image_paths(locations):
    paths = []
    for location in locations:
        if location.is_file():
            if location.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(location)
            continue
        if not location.is_dir():
            raise FileNotFoundError(f"No such file or folder: {location}")
        for path in sorted(location.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
    return paths


def load_classifier(model_path, device):
    return read_checkpoint(model_path, device)


def tensor_from_path(path, preprocess, device):
    return preprocess(load_rgb(path)).to(device)


@contextmanager
def forward_hooks(modules_by_name, detach=True):
    captured = {}
    handles = []

    def make_hook(key):
        def hook(_module, _inputs, output):
            captured[key] = output.detach() if detach else output

        return hook

    for name, module in modules_by_name.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fwd = target_layer.register_forward_hook(self._save_activation)
        self.bwd = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, _module, _inputs, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute(self, image_batch, class_index):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_batch)
        logits[0, class_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam,
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach()
        span = cam.max() - cam.min()
        if span.item() < 1e-8:
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam.min()) / span
        return cam, logits.detach()

    def close(self):
        self.fwd.remove()
        self.bwd.remove()


def save_figure(figure, path, show):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def slug(path):
    return Path(path).stem.replace(" ", "_")[:80]


def choose_sample(paths, count, use_all):
    paths = list(paths)
    if use_all or len(paths) <= count:
        return paths
    groups = {}
    for path in paths:
        key = (
            resin_from_path(path)
            or true_class_from_path(path)
            or "_unlabeled"
        )
        groups.setdefault(key, []).append(path)
    keys = list(groups)
    per_group = max(1, count // len(keys))
    chosen = []
    leftover = []
    for key in keys:
        bucket = list(groups[key])
        random.shuffle(bucket)
        take = min(per_group, len(bucket))
        chosen.extend(bucket[:take])
        leftover.extend(bucket[take:])
    random.shuffle(leftover)
    if len(chosen) < count:
        chosen.extend(leftover[: count - len(chosen)])
    return chosen[:count]


def pick_per_class(paths, per_class):
    groups = {}
    unlabeled = []
    for path in paths:
        label = true_class_from_path(path)
        if label is None:
            unlabeled.append(path)
        else:
            groups.setdefault(label, []).append(path)
    picked = []
    for label in EXPECTED_CLASSES:
        picked.extend(groups.get(label, [])[:per_class])
    for label, bucket in groups.items():
        if label not in EXPECTED_CLASSES:
            picked.extend(bucket[:per_class])
    if unlabeled:
        picked.extend(unlabeled[:per_class])
    return picked


def sample_labeled(paths, per_class):
    groups = {}
    for path in paths:
        label = true_class_from_path(path)
        if label is None:
            continue
        groups.setdefault(label, []).append(path)
    picked = {}
    for label, bucket in groups.items():
        if len(bucket) <= per_class:
            picked[label] = list(bucket)
        else:
            picked[label] = random.sample(bucket, per_class)
    return picked


def score_images(model, paths, preprocess, device, idx_to_class):
    records = []
    with torch.no_grad():
        for path in paths:
            image_tensor = tensor_from_path(path, preprocess, device)
            logits = model(image_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probabilities).item())
            predicted = idx_to_class[pred_idx]
            true_class = true_class_from_path(path)
            resin = resin_from_path(path)
            ranked = sorted(
                (
                    (idx_to_class[index], probabilities[index].item())
                    for index in range(probabilities.numel())
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            records.append(
                {
                    "path": path,
                    "pred_idx": pred_idx,
                    "predicted": predicted,
                    "true": true_class,
                    "resin": resin,
                    "confidence": probabilities[pred_idx].item(),
                    "probs": {
                        idx_to_class[index]: probabilities[index].item()
                        for index in range(probabilities.numel())
                    },
                    "ranked": ranked,
                    "mistake": (
                        true_class is not None
                        and lab_class(predicted) != true_class
                    ),
                }
            )
    return records


def paint_cam(display_image, cam):
    heat = plt.colormaps["jet"](cam.detach().cpu().numpy())[..., :3]
    base = np.asarray(display_image).astype(np.float32) / 255.0
    blended = (0.45 * heat + 0.55 * base).clip(0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def plot_probability_bars(axis, record, class_order):
    names = list(class_order)
    values = [record["probs"][name] for name in names]
    colors = []
    for name in names:
        if name == record["predicted"] and record["mistake"]:
            colors.append("tab:red")
        elif name == record["true"]:
            colors.append("tab:green")
        elif name == record["predicted"]:
            colors.append("tab:blue")
        else:
            colors.append("0.7")
    axis.barh(names, values, color=colors)
    axis.set_xlim(0, 1)
    axis.set_xlabel("confidence")
    for index, value in enumerate(values):
        axis.text(min(value + 0.02, 0.85), index, f"{value:.0%}", va="center")


def run_maps(model, paths, preprocess, device, out_dir, show, figures):
    if not paths:
        print("No images for feature maps.")
        return []
    lines = [
        "Bright cells are filters that fired on this 224×224 input. "
        "Each grid shows the 16 strongest channels, not filters 0–15. "
        "layer1 is edges and color; layer4 is a 7×7 abstract map."
    ]
    layer_modules = {name: getattr(model, name) for name in LAYER_NAMES}
    for path in paths:
        image_tensor = tensor_from_path(path, preprocess, device)
        with forward_hooks(layer_modules) as captured:
            with torch.no_grad():
                model(image_tensor.unsqueeze(0))
        label = true_class_from_path(path) or "unlabeled"
        for layer_name in LAYER_NAMES:
            maps = captured[layer_name][0].cpu()
            channel_mean = maps.mean(dim=(1, 2))
            top = torch.argsort(channel_mean, descending=True)[:16]
            figure, axes = plt.subplots(4, 4, figsize=(10, 10))
            for axis, index in zip(axes.flat, top):
                axis.imshow(maps[int(index)], cmap="viridis")
                axis.set_title(f"filter {int(index)}", fontsize=8)
                axis.set_axis_off()
            figure.suptitle(
                f"Top-activated {layer_name} maps\n{label} · {path.name}",
                fontsize=12,
            )
            save_path = (
                out_dir / f"maps_{label}_{slug(path)}_{layer_name}.png"
            )
            save_figure(figure, save_path, show)
            figures.append(save_path)
            mean_act = float(maps.mean())
            sparsity = float((maps == 0).sum() / maps.numel() * 100)
            lines.append(
                f"- {label} {path.name} {layer_name}: "
                f"{maps.shape[0]} filters, mean={mean_act:.4f}, "
                f"sparsity={sparsity:.1f}%"
            )
    return lines


def run_filters(
    model,
    example_paths,
    labeled_groups,
    preprocess,
    device,
    out_dir,
    show,
    figures,
):
    lines = [
        "Column 0 is the 224×224 tensor the network scored. "
        "The next four panels are the layer4 filters with the "
        "highest mean activation on that photo."
    ]
    if example_paths:
        rows = len(example_paths)
        figure, axes = plt.subplots(
            rows,
            5,
            figsize=(16, 3.2 * rows),
        )
        axes = np.atleast_2d(axes)
        layer = model.layer4
        for row, path in enumerate(example_paths):
            image_tensor = tensor_from_path(path, preprocess, device)
            with forward_hooks({"layer4": layer}) as captured:
                with torch.no_grad():
                    model(image_tensor.unsqueeze(0))
            maps = captured["layer4"][0].cpu()
            channel_mean = maps.mean(dim=(1, 2))
            top = torch.argsort(channel_mean, descending=True)[:4]
            label = true_class_from_path(path) or "unlabeled"
            axes[row, 0].imshow(np.asarray(tensor_to_display(image_tensor.cpu())))
            axes[row, 0].set_title(f"{label}\n{path.name}", fontsize=9)
            axes[row, 0].set_axis_off()
            top_ids = [int(index) for index in top]
            lines.append(f"- {label} {path.name}: top filters {top_ids}")
            for column, index in enumerate(top, start=1):
                axes[row, column].imshow(maps[int(index)], cmap="viridis")
                axes[row, column].set_title(
                    f"filter {int(index)}", fontsize=9
                )
                axes[row, column].set_axis_off()
        figure.suptitle(
            "Layer4 top-activated filters on the scored 224×224 input",
            fontsize=13,
        )
        save_path = out_dir / "filters_side_by_side.png"
        save_figure(figure, save_path, show)
        figures.append(save_path)

    class_means = {}
    for label, paths in labeled_groups.items():
        if not paths:
            continue
        sums = None
        with forward_hooks({"layer4": model.layer4}) as captured:
            for path in paths:
                image_tensor = tensor_from_path(path, preprocess, device)
                with torch.no_grad():
                    model(image_tensor.unsqueeze(0))
                means = captured["layer4"][0].mean(dim=(1, 2)).cpu().numpy()
                sums = means if sums is None else sums + means
        class_means[label] = sums / len(paths)

    if "metal" in class_means and "plastic" in class_means:
        delta = class_means["metal"] - class_means["plastic"]
        order = np.argsort(np.abs(delta))[::-1][:RANK_CHANNELS]
        colors = [
            "tab:orange" if delta[index] > 0 else "tab:blue"
            for index in order
        ]
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(
            [f"filter {int(index)}" for index in order][::-1],
            delta[order][::-1],
            color=colors[::-1],
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("mean layer4 activation: metal minus plastic")
        axis.set_title(
            "Filters that fire more for one class\n"
            "orange = metal, blue = plastic"
        )
        save_path = out_dir / "filters_class_ranking.png"
        save_figure(figure, save_path, show)
        figures.append(save_path)
        lines.append(
            "Positive bars fire more on metal photos; negative bars "
            "fire more on plastic. Head-only training still uses "
            "these ImageNet channels."
        )
        for index in order[:10]:
            lines.append(
                f"- filter {int(index)}: Δ={delta[index]:+.4f} "
                f"(metal {class_means['metal'][index]:.4f}, "
                f"plastic {class_means['plastic'][index]:.4f})"
            )
    elif class_means:
        lines.append(
            "Need both metal and plastic photos to rank class-preferring filters."
        )
    return lines


def run_stats(
    model,
    labeled_groups,
    preprocess,
    device,
    out_dir,
    show,
    figures,
):
    lines = [
        "Mean activation and sparsity (% of zeros) averaged over a "
        "sample of photos per class. Sparsity usually rises in deeper "
        "ReLU stages. If metal and plastic look the same through "
        "layer4, the frozen backbone is not separating them and the "
        "fc head is doing the work."
    ]
    layer_modules = {name: getattr(model, name) for name in LAYER_NAMES}
    summary = {
        label: {name: [] for name in LAYER_NAMES}
        for label in labeled_groups
    }
    for label, paths in labeled_groups.items():
        print(f"Layer stats: {label} × {len(paths)}")
        for path in paths:
            image_tensor = tensor_from_path(path, preprocess, device)
            with forward_hooks(layer_modules) as captured:
                with torch.no_grad():
                    model(image_tensor.unsqueeze(0))
            for name in LAYER_NAMES:
                maps = captured[name]
                summary[label][name].append(
                    {
                        "mean": float(maps.mean()),
                        "std": float(maps.std()),
                        "sparsity": float(
                            (maps == 0).sum() / maps.numel() * 100
                        ),
                    }
                )

    def average(label, name, field):
        rows = summary[label][name]
        return sum(row[field] for row in rows) / len(rows)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(LAYER_NAMES))
    width = 0.35 if len(labeled_groups) > 1 else 0.5
    for offset, label in enumerate(labeled_groups):
        means = [average(label, name, "mean") for name in LAYER_NAMES]
        sparsities = [
            average(label, name, "sparsity") for name in LAYER_NAMES
        ]
        shift = (offset - (len(labeled_groups) - 1) / 2) * width
        axes[0].bar(x + shift, means, width=width, label=label)
        axes[1].bar(x + shift, sparsities, width=width, label=label)
        lines.append(f"**{label}** ({len(labeled_groups[label])} photos):")
        for name in LAYER_NAMES:
            lines.append(
                f"- {name}: mean={average(label, name, 'mean'):.4f}, "
                f"std={average(label, name, 'std'):.4f}, "
                f"sparsity={average(label, name, 'sparsity'):.1f}%"
            )
    axes[0].set_xticks(x, LAYER_NAMES)
    axes[0].set_ylabel("mean activation")
    axes[0].set_title("Mean activation by layer")
    axes[0].legend()
    axes[1].set_xticks(x, LAYER_NAMES)
    axes[1].set_ylabel("sparsity (%)")
    axes[1].set_title("Sparsity by layer")
    axes[1].legend()
    figure.suptitle("Layer-wise activation statistics (sample average)")
    save_path = out_dir / "stats_layers.png"
    save_figure(figure, save_path, show)
    figures.append(save_path)
    return lines


def extract_avgpool_features(model, paths, preprocess, device):
    features = []
    with forward_hooks({"avgpool": model.avgpool}) as captured:
        for start in range(0, len(paths), FEATURE_BATCH):
            chunk = paths[start : start + FEATURE_BATCH]
            batch = torch.stack(
                [
                    tensor_from_path(path, preprocess, device)
                    for path in chunk
                ]
            )
            with torch.no_grad():
                model(batch)
            features.append(
                captured["avgpool"].flatten(1).cpu().numpy()
            )
    return np.concatenate(features, axis=0)


def scatter_classes(axis, points, labels, mistakes, title, xlabel, ylabel):
    palette = {"metal": "tab:orange", "plastic": "tab:blue"}
    for label in sorted(set(labels)):
        mask = np.array([item == label for item in labels])
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            c=palette.get(label, "0.4"),
            label=label,
            s=36,
            alpha=0.7,
            edgecolors="none",
        )
    if any(mistakes):
        error_mask = np.array(mistakes)
        axis.scatter(
            points[error_mask, 0],
            points[error_mask, 1],
            facecolors="none",
            edgecolors="black",
            s=90,
            linewidths=1.4,
            marker="x",
            label="mistake",
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(True, alpha=0.3)


def run_pca(
    model,
    records,
    preprocess,
    device,
    out_dir,
    show,
    figures,
):
    labeled = [row for row in records if row["true"] is not None]
    if len(labeled) < 5:
        return [
            "Need at least 5 labeled photos for PCA."
        ]
    paths = [row["path"] for row in labeled]
    print(f"Extracting avgpool vectors for {len(paths)} labeled photos...")
    vectors = extract_avgpool_features(model, paths, preprocess, device)
    labels = [row["true"] for row in labeled]
    mistakes = [row["mistake"] for row in labeled]
    scaled = StandardScaler().fit_transform(vectors)
    n_comp = min(10, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    reduced = pca.fit_transform(scaled)
    ratio = pca.explained_variance_ratio_
    lines = [
        "Each point is one photo after global average pooling (2048 numbers). "
        "Separate blobs mean the vector already clusters by class. "
        "A mixed cloud means overlap; black X marks are mistakes. "
        "Two PCs are a cartoon of 2048-D space — read the variance bars."
    ]
    lines.append(f"- samples: {len(paths)}")
    lines.append(f"- feature dim: {vectors.shape[1]}")
    lines.append(
        f"- PC1 {ratio[0]*100:.1f}%, PC2 {ratio[1]*100:.1f}%, "
        f"first two {sum(ratio[:2])*100:.1f}%"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_classes(
        axes[0],
        reduced[:, :2],
        labels,
        mistakes,
        "Feature space (PCA)",
        f"PC1 ({ratio[0]*100:.1f}% var)",
        f"PC2 ({ratio[1]*100:.1f}% var)",
    )
    axes[1].bar(
        [f"PC{i+1}" for i in range(len(ratio))],
        ratio * 100,
        color="0.45",
    )
    axes[1].set_ylabel("% variance")
    axes[1].set_title("Variance explained by the first PCs")
    axes[1].tick_params(axis="x", rotation=45)
    figure.suptitle("Avgpool feature space")
    save_path = out_dir / "pca_feature_space.png"
    save_figure(figure, save_path, show)
    figures.append(save_path)

    if len(paths) >= 10:
        perplexity = min(30, len(paths) - 1)
        pca_width = min(50, n_comp, len(paths) - 1)
        tsne_input = reduced[:, :pca_width]
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            max_iter=500,
            random_state=0,
        )
        embedded = tsne.fit_transform(tsne_input)
        figure, axis = plt.subplots(figsize=(7, 6))
        scatter_classes(
            axis,
            embedded,
            labels,
            mistakes,
            "Feature space (t-SNE on PCA)",
            "t-SNE 1",
            "t-SNE 2",
        )
        save_path = out_dir / "tsne_feature_space.png"
        save_figure(figure, save_path, show)
        figures.append(save_path)
        lines.append(
            f"t-SNE used PCA-{pca_width} then 2-D embedding "
            f"(perplexity={perplexity}). Local neighborhoods, not global axes."
        )
    return lines


def choose_cam_rows(records, sample_paths, mistakes_only, cover_all):
    """Which scored photos get a Grad-CAM figure.

    cover_all is the explicit short list: every photo, labeled or not.
    A long folder still keeps a few correct examples, every mistake,
    and any unlabeled photo that landed in the sample.
    """
    if cover_all and not mistakes_only:
        return list(records)
    mistakes = [row for row in records if row["mistake"]]
    correct = [
        row
        for row in records
        if row["true"] is not None and not row["mistake"]
    ]
    unlabeled = [row for row in records if row["true"] is None]
    chosen = []
    seen = set()

    def take(rows):
        for row in rows:
            if row["path"] in seen:
                continue
            seen.add(row["path"])
            chosen.append(row)

    if not mistakes_only:
        sample_set = set(sample_paths)
        sampled_correct = [
            row for row in correct if row["path"] in sample_set
        ]
        if not sampled_correct:
            sampled_correct = correct
        take(sampled_correct[:CORRECT_CAM_COUNT])
        take(row for row in unlabeled if row["path"] in sample_set)
    take(mistakes)
    return chosen


def run_cam(
    model,
    records,
    class_to_idx,
    preprocess,
    device,
    out_dir,
    show,
    figures,
    mistakes_only,
    sample_paths,
    cover_all,
):
    lines = [
        "Grad-CAM is not a feature map. It weights layer4 by how much "
        "each channel contributed to one class score, then upsamples "
        "the 7×7 result to 224×224. Overlay is on the exact tensor "
        "the model scored. On a mistake the middle panel is “why it "
        "predicted that” and the next panel is the true class. "
        "A photo with no metal or plastic folder still gets a figure "
        "for the predicted class."
    ]
    chosen = choose_cam_rows(
        records, sample_paths, mistakes_only, cover_all
    )
    if not chosen:
        return ["No Grad-CAM images."]

    engine = GradCAM(model, model.layer4[-1])
    class_order = [
        name
        for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    try:
        for row in chosen:
            image_tensor = tensor_from_path(row["path"], preprocess, device)
            image_batch = image_tensor.unsqueeze(0).detach().requires_grad_(True)
            display = tensor_to_display(image_tensor.cpu())
            with torch.enable_grad():
                cam_pred, _logits = engine.compute(
                    image_batch, row["pred_idx"]
                )
            figure, axes = plt.subplots(1, 4, figsize=(16.8, 4.2))
            axes[0].imshow(np.asarray(display))
            true_note = row["true"] if row["true"] is not None else "unlabeled"
            title_true = true_note
            if row.get("resin"):
                title_true = f"{true_note} ({row['resin']})"
                true_note = f"{true_note}_{row['resin']}"
            axes[0].set_title(f"224×224 input\ntrue {title_true}")
            axes[0].set_axis_off()
            if row["mistake"] and row["true"] in class_to_idx:
                image_batch = (
                    image_tensor.unsqueeze(0).detach().requires_grad_(True)
                )
                with torch.enable_grad():
                    cam_true, _ = engine.compute(
                        image_batch, class_to_idx[row["true"]]
                    )
                axes[1].imshow(np.asarray(paint_cam(display, cam_pred)))
                axes[1].set_title(
                    f"why predicted {row['predicted']}\n"
                    f"{row['confidence']:.0%}"
                )
                axes[1].set_axis_off()
                axes[2].imshow(np.asarray(paint_cam(display, cam_true)))
                axes[2].set_title(f"evidence for {row['true']}")
                axes[2].set_axis_off()
                plot_probability_bars(axes[3], row, class_order)
                status = "WRONG"
            else:
                axes[1].imshow(cam_pred.cpu().numpy(), cmap="jet")
                axes[1].set_title("attention heatmap")
                axes[1].set_axis_off()
                axes[2].imshow(np.asarray(paint_cam(display, cam_pred)))
                if row["true"] is None:
                    status = "UNLABELED"
                elif row["mistake"]:
                    status = "WRONG"
                else:
                    status = "CORRECT"
                axes[2].set_title(
                    f"{status}: {row['predicted']} {row['confidence']:.0%}"
                )
                axes[2].set_axis_off()
                plot_probability_bars(axes[3], row, class_order)
            figure.suptitle(row["path"].name, fontsize=11)
            save_path = (
                out_dir
                / (
                    f"cam_{status}_{true_note}_{row['predicted']}_"
                    f"{slug(row['path'])}.png"
                )
            )
            save_figure(figure, save_path, show)
            figures.append(save_path)
            if row["mistake"]:
                resin_note = (
                    f" ({row['resin']})" if row.get("resin") else ""
                )
                lines.append(
                    f"- {row['path']}: true {row['true']}{resin_note} → "
                    f"{row['predicted']} ({row['confidence']:.1%})"
                )
    finally:
        engine.close()

    labeled = [row for row in records if row["true"] is not None]
    if labeled:
        n_wrong = sum(1 for row in labeled if row["mistake"])
        lines.insert(
            1,
            f"Labeled accuracy on this set: "
            f"{len(labeled) - n_wrong}/{len(labeled)} "
            f"({(len(labeled) - n_wrong) / len(labeled):.1%}). "
            f"{n_wrong} mistakes have dual CAMs.",
        )
    return lines


def write_report(
    path,
    model_path,
    source_label,
    device,
    steps,
    sections,
    figures,
    records,
):
    labeled = [row for row in records if row["true"] is not None]
    mistakes = [row for row in labeled if row["mistake"]]
    lines = [
        "# Model inspection report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Model: `{model_path}`",
        f"- Images: {source_label}",
        f"- Device: {device}",
        f"- Steps: {', '.join(sorted(steps))}",
        "",
        "How to read this: feature maps show what fired; Grad-CAM shows "
        "which of those firings a class score used; PCA is a 2-D cartoon "
        "of the 2048-D pooled vector.",
        "",
    ]
    if labeled:
        correct = len(labeled) - len(mistakes)
        lines += [
            "## Scoring",
            "",
            f"- Labeled images: {len(labeled)}",
            f"- Correct: {correct} ({correct / len(labeled):.1%})",
            f"- Mistakes: {len(mistakes)}",
            "",
        ]
        totals = Counter(row.get("resin") or "unknown" for row in labeled)
        wrongs = Counter(
            row.get("resin") or "unknown" for row in mistakes
        )
        resin_keys = [
            key
            for key in (*RESIN_TAGS, "unknown")
            if totals.get(key)
        ]
        if resin_keys:
            lines.append("By resin:")
            lines.append("")
            for key in resin_keys:
                total = totals[key]
                wrong = wrongs.get(key, 0)
                lines.append(
                    f"- {key}: {wrong}/{total} mistakes "
                    f"({wrong / total:.1%})"
                )
            lines.append("")
        for row in mistakes:
            resin_note = f" ({row['resin']})" if row.get("resin") else ""
            lines.append(
                f"- `{row['path']}`: true {row['true']}{resin_note} → "
                f"{row['predicted']} ({row['confidence']:.1%})"
            )
        if mistakes:
            lines.append("")
    for title, body in sections:
        lines += [f"## {title}", ""]
        lines.extend(body)
        lines.append("")
    lines += ["## Figures", ""]
    for figure_path in figures:
        lines.append(f"- `{figure_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main():
    args = parse_args()
    steps = parse_steps(args.steps)
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    device = pick_device()
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    model, class_to_idx, idx_to_class = load_classifier(model_path, device)
    preprocess = inference_transform()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        image_paths = collect_image_paths(args.images)
        source_label = "selected photos"
    else:
        assert_split_layout(args.data_dir)
        test_dir = split_dirs_for(args.data_dir)["test"]
        image_paths = [
            path
            for class_name in EXPECTED_CLASSES
            for path in sorted((test_dir / class_name).iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        source_label = f"{test_dir} (test split)"

    image_paths = readable_image_paths(image_paths)
    if not image_paths:
        raise FileNotFoundError(
            "No images found. Pass --images or a dataset with a test split."
        )

    sample_paths = choose_sample(image_paths, args.count, args.all)
    print(f"{len(image_paths)} images, sample {len(sample_paths)}")

    need_all = bool(steps & {"cam", "pca"}) or args.mistakes_only
    score_paths = image_paths if need_all else sample_paths
    print(f"Scoring {len(score_paths)} images...")
    records = score_images(
        model, score_paths, preprocess, device, idx_to_class
    )
    labeled_groups = sample_labeled(image_paths, STATS_PER_CLASS)
    map_paths = pick_per_class(sample_paths, MAPS_PER_CLASS)
    figures = []
    sections = []
    show = not args.no_show

    if "maps" in steps:
        print("Step: feature maps")
        sections.append(
            (
                "Feature maps",
                run_maps(
                    model,
                    map_paths,
                    preprocess,
                    device,
                    out_dir,
                    show,
                    figures,
                ),
            )
        )
    if "filters" in steps:
        print("Step: filter activation")
        sections.append(
            (
                "Filter activation",
                run_filters(
                    model,
                    map_paths,
                    labeled_groups,
                    preprocess,
                    device,
                    out_dir,
                    show,
                    figures,
                ),
            )
        )
    if "stats" in steps:
        print("Step: layer stats")
        if labeled_groups:
            sections.append(
                (
                    "Layer activation statistics",
                    run_stats(
                        model,
                        labeled_groups,
                        preprocess,
                        device,
                        out_dir,
                        show,
                        figures,
                    ),
                )
            )
        else:
            sections.append(
                (
                    "Layer activation statistics",
                    ["No labeled metal/plastic folders in the path."],
                )
            )
    if "cam" in steps:
        print("Step: Grad-CAM")
        cover_all = (
            args.images is not None and len(image_paths) <= CAM_ALL_LIMIT
        )
        if args.images is not None and not cover_all:
            print(
                f"{len(image_paths)} images. Grad-CAM samples this set. "
                f"Pass {CAM_ALL_LIMIT} or fewer files to run it on each one."
            )
        sections.append(
            (
                "Grad-CAM",
                run_cam(
                    model,
                    records,
                    class_to_idx,
                    preprocess,
                    device,
                    out_dir,
                    show,
                    figures,
                    args.mistakes_only,
                    sample_paths,
                    cover_all,
                ),
            )
        )
    if "pca" in steps:
        print("Step: PCA / t-SNE")
        sections.append(
            (
                "Feature space PCA",
                run_pca(
                    model,
                    records,
                    preprocess,
                    device,
                    out_dir,
                    show,
                    figures,
                ),
            )
        )

    write_report(
        out_dir / "ANALYSIS_REPORT.md",
        model_path,
        source_label,
        device,
        steps,
        sections,
        figures,
        records,
    )
    print(f"Figures: {len(figures)}")


if __name__ == "__main__":
    main()
