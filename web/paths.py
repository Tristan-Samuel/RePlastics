"""Repo paths and the checks that keep zip and Finder inside them."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"
BELT_ZIP = DOWNLOADS / "belt_stage1_224.zip"
JOB_LOGS = ROOT / "web" / "job-logs"

DATA_ROOTS = ("data", "data_real", "data_sep16", "data_belt_stage1")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
RESINS = ("HDPE", "PET", "PP", "OTHERS")


def is_relative_inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def reveal_allowed(path):
    """Finder may open a repo file, the belt zip, or a zip this app wrote."""
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
