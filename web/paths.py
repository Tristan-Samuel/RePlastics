"""Repo paths and the checks that keep zip and folder reveal inside them."""

import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"
BELT_ZIP = DOWNLOADS / "belt_stage1_224.zip"
JOB_LOGS = ROOT / "web" / "job-logs"

DATA_ROOTS = ("data", "data_real", "data_sep16", "data_belt_stage1", "data_belt_oct23")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
RESINS = ("HDPE", "PET", "PP", "OTHERS")


def is_relative_inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def reveal_allowed(path):
    """The file manager may open a repo file, the belt zip, or a zip this app wrote."""
    resolved = Path(path).resolve()
    if is_relative_inside(resolved, ROOT):
        return True
    if resolved == BELT_ZIP.resolve():
        return True
    downloads = DOWNLOADS.resolve()
    return (
        resolved.parent == downloads
        and resolved.name.startswith("replastic-")
        and resolved.suffix == ".zip"
    )


def analysis_dir_allowed(path):
    resolved = Path(path).resolve()
    return resolved.is_dir() and is_relative_inside(resolved, ROOT / "analysis")


def image_allowed(path):
    resolved = Path(path).resolve()
    return (
        resolved.is_file()
        and resolved.suffix.lower() in IMAGE_SUFFIXES
        and is_relative_inside(resolved, ROOT)
        and not is_relative_inside(resolved, ROOT / "models")
    )


_index_lock = threading.Lock()
_index = {"built": {}, "by_root": {}}
_INDEX_TTL = 30.0


def resolve_image(rel):
    """The photo at this repo path, or the same filename after the file moved.

    Pipeline scores keep the path from when the photo was scored. A capture
    that later lands in a Garbage subfolder, or moves between Garbage and
    probes, still has the same filename under a data folder.
    """
    rel = str(rel or "").replace("\\", "/").lstrip("/")
    requested = Path(rel)
    if requested.is_absolute() or ".." in requested.parts:
        return None
    if requested.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    if not requested.parts or requested.parts[0] not in DATA_ROOTS:
        return None
    direct = ROOT / requested
    if image_allowed(direct):
        return direct.resolve()
    hits = _hits_named(requested.name, requested.parts[0])
    chosen = _prefer(requested, hits)
    if chosen is None or not image_allowed(chosen):
        return None
    return chosen.resolve()


def _hits_named(name, top):
    found = list(_index_root(top).get(name, ()))
    if found:
        return found
    for folder_name in DATA_ROOTS:
        if folder_name == top:
            continue
        found.extend(_index_root(folder_name).get(name, ()))
    return found


def _index_root(folder_name):
    now = time.monotonic()
    with _index_lock:
        cached = _index["by_root"].get(folder_name)
        built_at = _index["built"].get(folder_name, 0.0)
        if cached is not None and now - built_at < _INDEX_TTL:
            return cached
        mapping = {}
        folder = ROOT / folder_name
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    mapping.setdefault(path.name, []).append(path)
        _index["by_root"][folder_name] = mapping
        _index["built"][folder_name] = now
        return mapping


def _prefer(requested, hits):
    def rank(path):
        try:
            rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        parts = Path(rel).parts
        shared = 0
        for left, right in zip(requested.parts, parts):
            if left != right:
                break
            shared += 1
        in_folder = 1 if requested.parent.name in parts else 0
        return (-shared, -in_folder, len(parts), rel)

    ordered = sorted(hits, key=rank)
    return ordered[0] if ordered else None
