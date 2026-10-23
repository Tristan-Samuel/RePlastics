"""Build argv lists for the existing scripts. No shell."""

import os
import shlex
import shutil
from datetime import datetime
from pathlib import Path

from web.paths import DATA_ROOTS, ROOT

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


def display_argv(argv):
    args = list(argv)
    if (
        args
        and _is_bash(args[0])
        and len(args) > 1
        and args[1].endswith("colab_train.sh")
    ):
        args = ["./colab_train.sh", *args[2:]]
    elif args and Path(args[0]).name.startswith("python"):
        args = ["python", *args[1:]]
    shown = []
    for index, arg in enumerate(args):
        if index == 0:
            shown.append(arg)
        else:
            shown.append(shlex.quote(arg))
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


def _extra_images(form, data_dir):
    text = (form.get("images") or "").strip()
    if text:
        paths = text.split()
        if len(paths) > 8:
            raise CommandError("Pass at most 8 image paths.")
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
    return format(number, "g")


def _confirmed(form, needed):
    if needed and form.get("confirm") != "yes":
        raise CommandError("Confirm this action in the browser before starting it.")


def prepare(form, require_confirm=True):
    kind = (form.get("kind") or "train").strip()
    if kind == "train":
        return _prepare_train(form, require_confirm)
    if kind == "eval":
        return _prepare_eval(form)
    if kind == "analyze":
        return _prepare_analyze(form)
    if kind == "mistakes":
        return _prepare_mistakes(form)
    if kind == "confidence":
        return _prepare_confidence(form, require_confirm)
    if kind == "split":
        return _prepare_split(form, require_confirm)
    if kind == "belt-zip":
        return _prepare_belt_zip(form, require_confirm)
    if kind == "stop-t4":
        return _prepare_stop(form, require_confirm)
    raise CommandError("Unknown action.")


def _prepare_train(form, require_confirm):
    where = (form.get("where") or "local").strip()
    model_text = _model_path(form.get("model"), must_exist=False)
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
        return Prepared(argv, "Train on this computer", display_argv(argv), False, "")
    if where != "colab":
        raise CommandError("Pick the T4 or this computer.")
    _require_t4()
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
    argv += train_args
    message = "This starts a Colab GPU session and runs training. Continue?"
    if require_confirm:
        _confirmed(form, True)
    return Prepared(argv, "Train on T4", display_argv(argv), True, message)


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
    argv = [_python(), "build_belt_stage1.py"]
    message = (
        "This rebuilds data_belt_stage1 and "
        "belt_stage1_224.zip in Downloads. Continue?"
    )
    if require_confirm:
        _confirmed(form, True)
    return Prepared(argv, "Build 224 zip", display_argv(argv), True, message)


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
