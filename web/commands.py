"""Build argv lists for the existing scripts. No shell."""

import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from web.catalog import model_by_rel
from web.paths import BELT_ZIP, DATA_ROOTS, IMAGE_SUFFIXES, ROOT

UNFREEZE = ("fc", "layer4", "layer3", "all")
GPUS = ("T4", "L4", "A100")
STEPS = ("maps", "cam", "stats", "pca", "filters")
SESSION = "trainer"


class CommandError(Exception):
    pass


class Prepared:
    def __init__(
        self,
        argv,
        title,
        display,
        needs_confirm,
        confirm_message,
        record=None,
    ):
        self.argv = argv
        self.title = title
        self.display = display
        self.needs_confirm = needs_confirm
        self.confirm_message = confirm_message
        self.record = record


def _is_bash(arg):
    return Path(arg).name.lower() in {"bash", "bash.exe"}


def _display_interpreter(executable):
    """A command a Windows cmd prompt can run, and a Mac terminal too.

    The lab process is often ``.../MacOS/Python``. That path only exists
    on this Mac. ``Python`` does not start with a lowercase ``python``,
    so the old check left the whole path in the preview.
    """
    path = Path(executable)
    if os.name == "nt":
        try:
            relative = path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            relative = None
        if relative is not None:
            return str(relative)
        venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
        if venv_python.is_file():
            return str(venv_python.relative_to(ROOT))
    if path.name.lower().startswith("python"):
        return "python"
    return str(path)


def display_argv(argv):
    args = list(argv)
    if (
        args
        and _is_bash(args[0])
        and len(args) > 1
        and args[1].endswith("colab_train.sh")
    ):
        args = ["./colab_train.sh", *args[2:]]
    elif args and Path(args[0]).name.lower().startswith("python"):
        args = [_display_interpreter(args[0]), *args[1:]]
    if os.name == "nt":
        return subprocess.list2cmdline([str(arg) for arg in args])
    shown = []
    for index, arg in enumerate(args):
        if index == 0:
            shown.append(arg)
        else:
            shown.append(shlex.quote(str(arg)))
    return " ".join(shown)


def _flag(form, name):
    return form.get(name) in ("1", "on", "yes", "true", "True")


def _model_path(value, must_exist):
    text = (value or "").strip()
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise CommandError("Model path has to stay inside models/.")
    if path.suffix != ".pt" or len(path.parts) < 2 or path.parts[0] != "models":
        raise CommandError("Model path should look like models/name.pt.")
    if any(part.startswith(".") for part in path.parts):
        raise CommandError("Model path should look like models/name.pt.")
    resolved = (ROOT / path).resolve()
    models = (ROOT / "models").resolve()
    try:
        resolved.relative_to(models)
    except ValueError:
        raise CommandError("Model path has to stay inside models.")
    if must_exist and not resolved.is_file():
        raise CommandError(f"Checkpoint not found: {text}")
    return path.as_posix()


def _data_dir(value):
    text = (value or "").strip()
    if text not in DATA_ROOTS:
        raise CommandError("Pick one of the dataset folders.")
    if not (ROOT / text).is_dir():
        raise CommandError(f"{text} is not on disk.")
    return text


def _float_in(value, name, low, high, allow_blank=False):
    text = (value or "").strip()
    if allow_blank and text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        raise CommandError(f"{name} has to be a number.")
    if number < low or number > high:
        raise CommandError(f"{name} has to be between {low} and {high}.")
    return number


def _epochs(value):
    try:
        number = int(str(value).strip())
    except ValueError:
        raise CommandError("Epochs has to be a whole number.")
    if number < 1 or number > 100:
        raise CommandError("Epochs has to be from 1 to 100.")
    return number


def _steps(form):
    picked = []
    values = set(form.getlist("steps")) if hasattr(form, "getlist") else set(form.get("steps") or [])
    for name in STEPS:
        if name in values:
            picked.append(name)
    if not picked:
        raise CommandError("Pick at least one analytics step.")
    return ",".join(picked)


def _listed_images(form, limit=8):
    text = (form.get("images") or "").strip()
    if not text:
        return []
    paths = text.split()
    if len(paths) > limit:
        raise CommandError(f"Pass at most {limit} image paths.")
    cleaned = []
    for raw in paths:
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            raise CommandError(f"Image path has to stay in the repo: {raw}")
        resolved = (ROOT / path).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            raise CommandError(f"Image path has to stay in the repo: {raw}")
        if not resolved.exists() or resolved.suffix == ".pt":
            raise CommandError(f"Not a usable image path: {raw}")
        cleaned.append(path.as_posix())
    return cleaned


def _extra_images(form, data_dir):
    cleaned = _listed_images(form)
    if cleaned:
        return cleaned
    if _flag(form, "both_splits"):
        folders = []
        for split in ("validation", "test"):
            folder = ROOT / data_dir / split
            if not folder.is_dir():
                raise CommandError(f"{data_dir}/{split} is not on disk.")
            folders.append(f"{data_dir}/{split}")
        return folders
    return []


def _stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_dir(model_text):
    stem = Path(model_text).name
    if stem.endswith(".pt"):
        stem = stem[:-3]
    folder = Path("analysis") / stem / _stamp()
    return folder.as_posix()


def _train_args(form, model_text):
    unfreeze = (form.get("unfreeze") or "").strip()
    if unfreeze not in UNFREEZE:
        raise CommandError("Unfreeze has to be fc, layer4, layer3, or all.")
    epochs = _epochs(form.get("epochs"))
    lr = _float_in(form.get("lr"), "Learning rate", 1e-8, 1)
    weight = _float_in(form.get("weight_power"), "Weight power", 0, 2)
    backbone = _float_in(
        form.get("lr_backbone"),
        "Backbone learning rate",
        1e-8,
        1,
        allow_blank=True,
    )
    args = []
    if _flag(form, "resume"):
        args.append("--resume")
    args += [
        "--unfreeze",
        unfreeze,
        "--epochs",
        str(epochs),
        "--lr",
        _trim(lr),
        "--weight-power",
        _trim(weight),
        "--model",
        model_text,
    ]
    if backbone is not None:
        args += ["--lr-backbone", _trim(backbone)]
    return args


def _trim(number):
    text = format(float(number), ".8f").rstrip("0").rstrip(".")
    return text or "0"


def _confirmed(form, needed):
    if needed and form.get("confirm") != "yes":
        raise CommandError("Confirm this action in the browser before starting it.")


def prepare(form, require_confirm=True):
    kind = (form.get("kind") or "train").strip()
    if kind == "train":
        return _prepare_train(form, require_confirm)
    if kind == "onnx":
        return _prepare_onnx(form)
    if kind == "onnx-check":
        return _prepare_onnx_check(form)
    if kind == "eval":
        return _prepare_eval(form)
    if kind == "analyze":
        return _prepare_analyze(form)
    if kind == "gradcam":
        return _prepare_gradcam(form)
    if kind == "mistakes":
        return _prepare_mistakes(form)
    if kind == "confidence":
        return _prepare_confidence(form, require_confirm)
    if kind == "open-set":
        return _prepare_open_set(form, require_confirm)
    if kind == "presence":
        return _prepare_presence(form, require_confirm)
    if kind == "pipeline":
        return _prepare_pipeline(form, require_confirm)
    if kind == "split":
        return _prepare_split(form, require_confirm)
    if kind == "belt-zip":
        return _prepare_belt_zip(form, require_confirm)
    if kind == "stop-t4":
        return _prepare_stop(form, require_confirm)
    raise CommandError("Unknown action.")


def _onnx_argv(model, data_dir, check_only=False):
    model_text = _model_path(
        model or "models/resnext50_metal_plastic.pt",
        must_exist=True,
    )
    folder = _data_dir(data_dir or "data_belt_stage1")
    out = Path(model_text).with_suffix(".onnx").as_posix()
    return [
        _python(),
        "export_onnx.py",
        "--model",
        model_text,
        "--out",
        out,
        "--check-only" if check_only else "--check",
        "--data-dir",
        folder,
    ]


def _prepare_onnx(form):
    argv = _onnx_argv(form.get("model"), form.get("data_dir"))
    return Prepared(argv, "Export ONNX", display_argv(argv), False, "")


def _prepare_onnx_check(form):
    argv = _onnx_argv(form.get("model"), form.get("data_dir"), check_only=True)
    out = argv[argv.index("--out") + 1]
    if not (ROOT / out).is_file():
        raise CommandError(f"No ONNX file at {out}. Make ONNX first.")
    return Prepared(argv, "Check ONNX", display_argv(argv), False, "")


def onnx_display(model="models/resnext50_metal_plastic.pt", data_dir="data_belt_stage1"):
    """The export line, quoted for this machine's shell."""
    return display_argv(_onnx_argv(model, data_dir))


def _pct(value):
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _photo_note():
    if not BELT_ZIP.is_file():
        return (
            f"This computer has no photo zip at {BELT_ZIP}. "
            "The T4 run will stop. Drive's copy is not used."
        )
    size = BELT_ZIP.stat().st_size
    return (
        f"Photos: {BELT_ZIP} ({size:,} bytes) on this computer. "
        "That file is uploaded. Drive's zip is not read."
    )


def _checkpoint_note(model_text, where, drive):
    model = model_by_rel(model_text)
    if model is None:
        return f"Starting file: {model_text}. It has to be a checkpoint on this computer."
    metrics = model["metrics"]
    lines = []
    if model["warning"]:
        lines.append(model["warning"])
    lines.append(f"Starting weights: {model['title']} ({model_text}).")
    lines.append(
        "Scores stored in that file: "
        f"validation {_pct(metrics.get('validation'))}, "
        f"balanced {_pct(metrics.get('balanced'))}, "
        f"test {_pct(metrics.get('test'))}."
    )
    lines.append(model["role"])
    lines.append(
        "Start makes one copy and trains that copy. "
        "The file you picked stays. A second copy is not written."
    )
    if where == "colab":
        lines.append(
            "Those bytes are uploaded from this computer. "
            "A Drive file with the same checkpoint name is not loaded."
        )
        if drive:
            lines.append(_photo_note())
    return " ".join(lines)


def _copy_for_run(model_text):
    source = ROOT / model_text
    if not source.is_file():
        raise CommandError(f"Checkpoint not found: {model_text}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = source.with_name(f"{source.stem}_{stamp}{source.suffix}")
    if dest.exists():
        raise CommandError(f"Already exists: {dest.name}")
    try:
        subprocess.run(["cp", "-c", str(source), str(dest)], check=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.copy2(source, dest)
    return dest.relative_to(ROOT).as_posix()


def _prepare_train(form, require_confirm):
    where = (form.get("where") or "local").strip()
    resume = _flag(form, "resume")
    model_text = _model_path(form.get("model"), must_exist=resume)
    will_copy = (ROOT / model_text).is_file()
    note = _checkpoint_note(model_text, where, _flag(form, "drive")) if will_copy else ""
    copied_from = ""
    if where == "colab":
        _require_t4()
    if require_confirm:
        _confirmed(form, True if where == "colab" else will_copy)
        if will_copy:
            copied_from = model_text
            model_text = _copy_for_run(model_text)
    train_args = _train_args(form, model_text)
    if where == "local":
        data_dir = _data_dir(form.get("data_dir"))
        argv = [
            _python(),
            "train.py",
            *train_args,
            "--data-dir",
            data_dir,
        ]
        display = display_argv(argv)
        if copied_from:
            display = f"Copied {copied_from} to {model_text}\n{display}"
        return Prepared(
            argv,
            "Train on this computer",
            display,
            will_copy,
            note,
        )
    if where != "colab":
        raise CommandError("Pick the T4 or this computer.")
    gpu = (form.get("gpu") or "T4").strip()
    if gpu not in GPUS:
        raise CommandError("GPU has to be T4, L4, or A100.")
    argv = [_real_bash(), str(ROOT / "colab_train.sh"), "--gpu", gpu]
    if _flag(form, "keep"):
        argv.append("--keep")
    if _flag(form, "drive"):
        argv.append("--drive")
    else:
        data_dir = _data_dir(form.get("data_dir"))
        argv += ["--data-dir", data_dir]
    if copied_from:
        argv += ["--out", model_text]
    argv += train_args
    display = display_argv(argv)
    if copied_from:
        display = f"Copied {copied_from} to {model_text}\n{display}"
    return Prepared(argv, "Train on T4", display, True, note or "This starts a Colab GPU session and runs training. Continue?")


def _prepare_eval(form):
    model_text = _model_path(form.get("model"), must_exist=True)
    data_dir = _data_dir(form.get("data_dir"))
    stem = Path(model_text).name
    if stem.endswith(".pt"):
        stem = stem[:-3]
    record = f"analysis/{stem}/eval-{_stamp()}.txt"
    argv = [
        _python(),
        "train.py",
        "--eval",
        "--model",
        model_text,
        "--data-dir",
        data_dir,
    ]
    return Prepared(argv, "Eval", display_argv(argv), False, "", record)


def _prepare_analyze(form):
    model_text = _model_path(form.get("model"), must_exist=True)
    data_dir = _data_dir(form.get("data_dir"))
    steps = _steps(form)
    out = _run_dir(model_text)
    argv = [
        _python(),
        "analyze.py",
        "--no-show",
        "--model",
        model_text,
        "--out",
        out,
        "--steps",
        steps,
    ]
    images = _extra_images(form, data_dir)
    if images:
        argv += ["--images", *images]
    else:
        argv += ["--data-dir", data_dir]
    return Prepared(argv, "Analytics", display_argv(argv), False, "")


def _prepare_gradcam(form):
    model_text = _model_path(form.get("model"), must_exist=True)
    images = _listed_images(form, limit=32)
    if not images:
        raise CommandError("Choose an image, or paste a path inside this repo.")
    for raw in images:
        path = ROOT / raw
        if path.is_dir():
            raise CommandError(
                "Grad-CAM takes image files. Use full analytics for a folder."
            )
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise CommandError(f"Not an image: {raw}")
    out = _run_dir(model_text)
    argv = [
        _python(),
        "analyze.py",
        "--no-show",
        "--model",
        model_text,
        "--out",
        out,
        "--steps",
        "cam",
        "--images",
        *images,
    ]
    return Prepared(argv, "Grad-CAM", display_argv(argv), False, "")


def _prepare_mistakes(form):
    model_text = _model_path(form.get("model"), must_exist=True)
    data_dir = _data_dir(form.get("data_dir"))
    out_dir = _run_dir(model_text)
    png = out_dir + "/all_labeled_mistakes.png"
    argv = [
        _python(),
        "predict.py",
        "--no-show",
        "--model",
        model_text,
        "--mistakes-out",
        png,
    ]
    images = _extra_images(form, data_dir)
    if images:
        argv += ["--images", *images]
    else:
        argv += ["--data-dir", data_dir]
    threshold = _float_in(
        form.get("threshold"),
        "Threshold",
        0.01,
        1,
        allow_blank=True,
    )
    if threshold is not None:
        argv += ["--threshold", _trim(threshold)]
    return Prepared(argv, "Mistakes", display_argv(argv), False, "")


def _prepare_confidence(form, require_confirm):
    model_text = _model_path(form.get("model"), must_exist=True)
    data_dir = _data_dir(form.get("data_dir") or "data_belt_stage1")
    rerun = _flag(form, "rerun")
    if rerun:
        out = (form.get("out") or "").strip()
        path = Path(out)
        if path.is_absolute() or ".." in path.parts:
            raise CommandError("Confidence folder has to stay inside analysis/.")
        resolved = (ROOT / path).resolve()
        analysis = (ROOT / "analysis").resolve()
        try:
            resolved.relative_to(analysis)
        except ValueError:
            raise CommandError("Confidence folder has to stay inside analysis/.")
        if not resolved.is_dir():
            raise CommandError("That confidence folder is not on disk.")
        name = resolved.name
        if name != "confidence" and not name.startswith("confidence-"):
            raise CommandError("Rescore only a confidence folder.")
        out = path.as_posix()
    else:
        stem = Path(model_text).name
        if stem.endswith(".pt"):
            stem = stem[:-3]
        out = f"analysis/{stem}/confidence-{_stamp()}"
    argv = [
        _python(),
        "confidence_report.py",
        "--model",
        model_text,
        "--data-dir",
        data_dir,
        "--out",
        out,
    ]
    message = ""
    if rerun:
        argv.append("--rerun-test")
        message = "This scores the locked test split again. Continue?"
        if require_confirm:
            _confirmed(form, True)
    return Prepared(
        argv,
        "Confidence report",
        display_argv(argv),
        rerun,
        message,
    )


def _prepare_open_set(form, require_confirm):
    model_text = _model_path(form.get("model"), must_exist=True)
    data_dir = _data_dir(form.get("data_dir") or "data_belt_stage1")
    stem = Path(model_text).stem
    out = f"analysis/{stem}/open-set"
    marker = ROOT / out / "test_scored.txt"
    argv = [
        _python(),
        "open_set.py",
        "--model",
        model_text,
        "--data-dir",
        data_dir,
        "--out",
        out,
    ]
    needs = marker.is_file()
    message = ""
    if needs:
        argv.append("--rerun-test")
        message = "This scores the locked test split again. Continue?"
        if require_confirm:
            _confirmed(form, True)
    return Prepared(argv, "Open set", display_argv(argv), needs, message)


def _prepare_presence(form, require_confirm):
    model_text = _model_path(form.get("model"), must_exist=True)
    data_dir = _data_dir(form.get("data_dir") or "data_belt_stage1")
    stem = Path(model_text).stem
    out = f"analysis/{stem}/presence"
    marker = ROOT / out / "test_scored.txt"
    argv = [
        _python(),
        "presence.py",
        "--model",
        model_text,
        "--data-dir",
        data_dir,
        "--blanks",
        "data/probes/blanks",
        "--out",
        out,
    ]
    needs = marker.is_file()
    message = ""
    if needs:
        argv.append("--rerun-test")
        message = "This scores the locked test split again. Continue?"
        if require_confirm:
            _confirmed(form, True)
    return Prepared(argv, "Belt presence", display_argv(argv), needs, message)


def _form_list(form, name):
    getter = getattr(form, "getlist", None)
    if getter is None:
        value = (form.get(name) or "").strip()
        return [value] if value else []
    return [item for item in getter(name) if (item or "").strip()]


def _prepare_pipeline(form, require_confirm):
    data_dir = _data_dir(form.get("data_dir") or "data_belt_stage1")
    models = []
    for item in _form_list(form, "model"):
        models.append(_model_path(item, must_exist=True))
    models = list(dict.fromkeys(models))
    belt = ROOT / "analysis" / "pipeline-cache" / data_dir / "belt.json"
    model_paths = [
        ROOT / "analysis" / "pipeline-cache" / data_dir / "models" / f"{Path(rel).stem}.json"
        for rel in models
    ]
    needs = belt.is_file() and all(path.is_file() for path in model_paths)
    argv = [_python(), "pipeline.py", "--data-dir", data_dir]
    for rel in models:
        argv.extend(["--model", rel])
    if not models:
        argv.append("--belt-only")
    message = ""
    if needs:
        argv.append("--rerun")
        message = "This scores these photos again and replaces the cache. Continue?"
        if require_confirm:
            _confirmed(form, True)
    title = "Pipeline scores" if models else "Pipeline belt checks"
    return Prepared(argv, title, display_argv(argv), needs, message)


def _prepare_split(form, require_confirm):
    data_dir = _data_dir(form.get("data_dir"))
    argv = [_python(), "split_data.py", "--dest", data_dir]
    message = (
        "Split copies metal and plastic into train, validation, and test, "
        "then deletes those two source folders. Continue?"
    )
    if require_confirm:
        _confirmed(form, True)
    return Prepared(argv, f"Split {data_dir}", display_argv(argv), True, message)


def _prepare_belt_zip(form, require_confirm):
    argv = [_python(), "build_belt_stage1.py", "--zip-only"]
    message = (
        "This zips the photos now in data_belt_stage1 into "
        "belt_stage1_224.zip in Downloads. "
        "removed_stage1 stays out. Continue?"
    )
    if require_confirm:
        _confirmed(form, True)
    return Prepared(argv, "Zip current photos", display_argv(argv), True, message)


def _prepare_stop(form, require_confirm):
    if shutil.which("colab") is None:
        raise CommandError(
            "The colab command is not installed, so there is no T4 session to stop."
        )
    argv = ["colab", "stop", "-s", SESSION]
    message = "Stop the Colab session named trainer?"
    if require_confirm:
        _confirmed(form, True)
    return Prepared(argv, "Stop T4", display_argv(argv), True, message)


def _real_bash():
    """A shell that can run colab_train.sh. Skip the Windows Store stub."""
    if os.name == "nt":
        candidates = []
        found = shutil.which("bash")
        if found:
            candidates.append(found)
        candidates.extend(
            [
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files\Git\usr\bin\bash.exe",
            ]
        )
        for candidate in candidates:
            path = Path(candidate)
            if path.parent.name.lower() == "system32":
                continue
            if path.is_file():
                return str(path)
        return None
    return shutil.which("bash")


def _require_t4():
    if _real_bash() is None:
        raise CommandError(
            "T4 training needs bash. This machine does not have it, "
            "so train on this computer instead."
        )
    if shutil.which("colab") is None:
        raise CommandError(
            "T4 training needs the colab command, which is not installed. "
            "Train on this computer instead."
        )


def _python():
    import sys

    return sys.executable
