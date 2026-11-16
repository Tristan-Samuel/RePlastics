"""HTTP routes for the lab app."""

import json
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from web import catalog
from web.commands import GPUS, STEPS, UNFREEZE, CommandError, onnx_display, prepare
from web.jobs import BusyError, latest_onnx_check, runner
from web.paths import (
    BELT_ZIP,
    DATA_ROOTS,
    DOWNLOADS,
    IMAGE_SUFFIXES,
    ROOT,
    analysis_dir_allowed,
    image_allowed,
    is_relative_inside,
    resolve_image,
    reveal_allowed,
)
from web.presets import PRESETS

_UPLOAD_LIMIT = 25 * 1024 * 1024
_PROBE_LIMIT = 1500
_session = {"up": False, "label": "T4 …"}
_probe_scorer = None
_probe_lock = threading.Lock()
_session_lock = threading.Lock()
_watch_started = False


def create_app():
    global _watch_started
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    if not _watch_started:
        threading.Thread(target=_watch_session, daemon=True).start()
        _watch_started = True

    @app.context_processor
    def inject():
        with _session_lock:
            session = dict(_session)
        return {
            "session_state": session,
            "job_state": runner.public_state(),
        }

    @app.get("/")
    def jobs_page():
        return render_template(
            "jobs.html",
            presets=PRESETS,
            models=catalog.list_models(),
            belt_zip=catalog.belt_zip_info(),
            data_roots=DATA_ROOTS,
            unfreeze=UNFREEZE,
            gpus=GPUS,
            defaults=PRESETS[0],
        )

    @app.get("/data")
    def data_page():
        return render_template(
            "data.html",
            datasets=catalog.dataset_counts(),
            belt_zip=str(BELT_ZIP),
            belt_zip_exists=BELT_ZIP.exists(),
        )

    @app.get("/jetson")
    def jetson_page():
        models = catalog.list_models()
        data_roots = catalog.present_data_roots()
        selected = _chosen_model(request.args.get("model"), models)
        data_name = _chosen_data(request.args.get("data"))
        onnx_rel = ""
        json_rel = ""
        export_command = ""
        if selected and data_name in data_roots:
            onnx_rel = str(Path(selected).with_suffix(".onnx"))
            json_rel = onnx_rel + ".json"
            try:
                export_command = onnx_display(selected, data_name)
            except CommandError:
                export_command = ""
        engine_onnx = Path(onnx_rel).name if onnx_rel else "model.onnx"
        return render_template(
            "jetson.html",
            models=models,
            data_roots=data_roots,
            selected_model=selected,
            selected_data=data_name,
            export_command=export_command,
            engine_onnx=engine_onnx,
            engine_stem=Path(engine_onnx).stem,
            onnx_rel=onnx_rel,
            json_rel=json_rel,
            onnx_ready=bool(onnx_rel) and (ROOT / onnx_rel).is_file(),
            json_ready=bool(json_rel) and (ROOT / json_rel).is_file(),
            onnx_files=_onnx_files(),
            check=latest_onnx_check(onnx_rel),
        )

    @app.get("/open-set")
    def open_set_page():
        models = catalog.list_models()
        data_roots = catalog.present_data_roots()
        selected = _chosen_model(request.args.get("model"), models)
        data_name = _chosen_data(request.args.get("data"))
        model = next((item for item in models if item["rel"] == selected), None)
        brief = catalog.open_set_brief(model) if model else None
        return render_template(
            "open_set.html",
            models=models,
            data_roots=data_roots,
            selected_model=selected,
            selected_data=data_name,
            model=model,
            brief=brief,
        )

    @app.get("/open-set/scores.json")
    def open_set_scores():
        models = catalog.list_models()
        selected = _chosen_model(request.args.get("model"), models)
        model = next((item for item in models if item["rel"] == selected), None)
        if model is None:
            abort(404)
        path = catalog.open_set_scores_path(model)
        if path is None:
            return {"ready": False, "frames": []}
        return send_file(path, mimetype="application/json")

    @app.post("/open-set/probe")
    def open_set_probe():
        from PIL import UnidentifiedImageError

        from open_set import OpenSetError

        try:
            models = catalog.list_models()
            selected = _chosen_model(request.form.get("model"), models)
            model = next(
                (item for item in models if item["rel"] == selected),
                None,
            )
            if model is None:
                raise CommandError("Pick a checkpoint.")
            brief = catalog.open_set_brief(model)
            if brief is None:
                raise CommandError("Build a bank before scoring photos.")
            if brief["stale"]:
                raise CommandError(
                    "This checkpoint changed after the bank was built. Rerun first."
                )
            paths, skipped = _probe_paths()
            frames = _scorer().score(
                ROOT / model["rel"],
                ROOT / brief["rel"],
                paths,
            )
        except CommandError as error:
            return {"ok": False, "error": str(error)}, 400
        except OpenSetError as error:
            return {"ok": False, "error": str(error)}, 400
        except (OSError, UnidentifiedImageError) as error:
            return {"ok": False, "error": f"Could not read that photo: {error}"}, 400
        return {"ok": True, "frames": frames, "skipped": skipped}

    @app.get("/belt")
    def belt_page():
        models = catalog.list_models()
        data_roots = catalog.present_data_roots()
        selected = _chosen_model(request.args.get("model"), models)
        data_name = _chosen_data(request.args.get("data"))
        model = next((item for item in models if item["rel"] == selected), None)
        brief = catalog.presence_brief(model) if model else None
        return render_template(
            "belt.html",
            models=models,
            data_roots=data_roots,
            selected_model=selected,
            selected_data=data_name,
            model=model,
            brief=brief,
        )

    @app.get("/pipeline")
    def pipeline_page():
        from belt_checks import CHECKS

        models = catalog.list_models()
        data_roots = catalog.present_data_roots()
        selected = _chosen_model(request.args.get("model"), models)
        data_name = _chosen_data(request.args.get("data"))
        return render_template(
            "pipeline.html",
            models=models,
            model_choices=[
                {
                    "rel": item["rel"],
                    "title": item["title"],
                    "label": item["option_label"],
                    "classes": list(item.get("classes") or []),
                    "head": item.get("head") or "",
                }
                for item in models
            ],
            checks=list(CHECKS),
            data_roots=data_roots,
            selected_model=selected,
            selected_data=data_name,
            belt_ready=_belt_ready(data_name),
            scored_models=_scored_model_rels(data_name, models),
            graphs=_pipeline_graph_names(),
        )

    @app.get("/pipeline/belt.json")
    def pipeline_belt():
        data_name = _pipeline_data()
        path = _belt_cache(data_name)
        if not path.is_file():
            return _no_store(jsonify({"ready": False, "frames": [], "defaults": []}))
        return _no_store(send_file(path, mimetype="application/json"))

    @app.get("/pipeline/model.json")
    def pipeline_model():
        from web.commands import _model_path

        data_name = _pipeline_data()
        try:
            rel = _model_path(request.args.get("model"), must_exist=True)
        except CommandError:
            abort(404)
        path = _model_cache(data_name, rel)
        if not path.is_file():
            return _no_store(jsonify({"ready": False, "model": rel, "frames": []}))
        return _no_store(send_file(path, mimetype="application/json"))

    @app.get("/pipeline/graphs/<name>")
    def pipeline_graph(name):
        path = _pipeline_graph_path(name)
        if path is None or not path.is_file():
            abort(404)
        return send_file(path, mimetype="application/json")

    @app.post("/pipeline/graphs")
    def save_pipeline_graph():
        payload = request.get_json(silent=True) or {}
        name = payload.get("name") or ""
        graph = payload.get("graph")
        if _pipeline_graph_path(name) is None:
            return {"ok": False, "error": "Use letters, numbers, and dashes in the name."}, 400
        if not isinstance(graph, dict):
            return {"ok": False, "error": "Missing the graph."}, 400
        text = json.dumps({"name": name, "graph": graph}, indent=2)
        if len(text) > 500_000:
            return {"ok": False, "error": "That graph is too large."}, 400
        path = _pipeline_graph_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        return {"ok": True, "name": name}

    @app.get("/models")
    def models_page():
        return render_template(
            "models.html",
            models=catalog.list_models(),
            others=catalog.other_analyses(),
        )

    @app.post("/models/<stem>/delete")
    def delete_model(stem):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            abort(404)
        if request.form.get("confirm") != "yes":
            return render_template(
                "error.html",
                message="Confirm this delete in the browser before removing the checkpoint.",
            ), 400
        try:
            catalog.delete_model(stem)
        except FileNotFoundError:
            abort(404)
        return redirect(url_for("models_page"))

    @app.post("/models/<stem>/rename")
    def rename_model(stem):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            abort(404)
        back_list = request.form.get("next") == "models"
        back_url = url_for("models_page") if back_list else url_for("model_page", stem=stem)
        back_label = "Back to models" if back_list else "Back to the model"
        try:
            model = catalog.rename_model(
                stem,
                request.form.get("filename") or "",
                request.form.get("title") or "",
            )
        except FileNotFoundError:
            abort(404)
        except ValueError as exc:
            return render_template(
                "error.html",
                message=str(exc),
                back_url=back_url,
                back_label=back_label,
            ), 400
        if back_list:
            args = {}
            query = (request.form.get("q") or "").strip()[:200]
            sort = (request.form.get("sort") or "").strip()
            if query:
                args["q"] = query
            if sort in {
                "oldest",
                "name",
                "name-desc",
                "balanced",
                "validation",
                "test",
                "size",
            }:
                args["sort"] = sort
            return redirect(url_for("models_page", **args))
        return redirect(url_for("model_page", stem=model["stem"]))

    @app.get("/models/<stem>")
    def model_page(stem):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            abort(404)
        model = catalog.model_by_stem(stem)
        if model is None:
            abort(404)
        page = catalog.page_for_model(
            model,
            request.args.get("run"),
            request.args.get("confidence"),
        )
        data_name = _chosen_data(request.args.get("data"))
        catalog.link_run_sources(page["run"], data_name)
        return render_template(
            "model.html",
            model=model,
            page=page,
            data_roots=DATA_ROOTS,
            steps=STEPS,
            default_data=data_name,
            composition=catalog.composition(data_name),
        )

    @app.get("/datasets/<name>/images")
    def dataset_images(name):
        if name not in DATA_ROOTS:
            abort(404)
        return catalog.search_images(
            name,
            request.args.get("q") or "",
            request.args.get("split") or "",
            request.args.get("class") or "",
            request.args.get("resin") or "",
            request.args.get("origin") or "",
        )

    @app.get("/models/<stem>/scores.json")
    def model_scores(stem):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            abort(404)
        model = catalog.model_by_stem(stem)
        if model is None:
            abort(404)
        return catalog.scores_for_model(
            model,
            request.args.get("run"),
            request.args.get("confidence"),
        )

    @app.get("/saved/<slug>")
    def saved_page(slug):
        item = catalog.other_by_slug(slug)
        if item is None:
            abort(404)
        return render_template(
            "saved.html",
            item=item,
            run=catalog.page_for_saved(item["rel"]),
        )

    @app.get("/files/<path:rel>")
    def files(rel):
        path = resolve_image(rel)
        if path is None:
            abort(404)
        return send_file(path)

    @app.post("/jobs/preview")
    def preview_job():
        try:
            prepared = prepare(request.form, require_confirm=False)
        except CommandError as error:
            return {"ok": False, "error": str(error)}
        return {
            "ok": True,
            "display": prepared.display,
            "needs_confirm": prepared.needs_confirm,
            "message": prepared.confirm_message,
        }

    @app.post("/jobs/start")
    def start_job():
        json_job = "application/json" in (request.headers.get("Accept") or "")
        try:
            form = _job_form()
            prepared = prepare(form, require_confirm=True)
            runner.start(prepared)
        except CommandError as error:
            if json_job:
                return jsonify({"ok": False, "error": str(error)}), 400
            return render_template("error.html", message=str(error)), 400
        except BusyError as error:
            if json_job:
                return jsonify({"ok": False, "error": f"{error} is already running."}), 409
            return (
                render_template(
                    "error.html",
                    message=f"{error} is already running.",
                ),
                409,
            )
        if json_job:
            return jsonify({"ok": True, "title": prepared.title})
        if (form.get("kind") or "").strip() == "gradcam":
            stem = Path(form.get("model") or "").stem
            argv = prepared.argv
            out = argv[argv.index("--out") + 1] if "--out" in argv else ""
            return redirect(url_for("model_page", stem=stem, run=out))
        return _back()

    @app.post("/jobs/stop")
    def stop_job():
        runner.stop()
        return _back()

    @app.post("/jobs/stdin")
    def job_stdin():
        text = request.form.get("text") or ""
        if not text.strip():
            return render_template("error.html", message="Paste something first."), 400
        try:
            runner.write_stdin(text)
        except BusyError as error:
            return render_template("error.html", message=str(error)), 409
        return _back()

    @app.get("/jobs/status")
    def job_status():
        state = runner.public_state()
        with _session_lock:
            state["session"] = dict(_session)
        return state

    @app.get("/jobs/log")
    def job_log():
        return Response(runner.text(), mimetype="text/plain")

    @app.get("/jobs/stream")
    def job_stream():
        def generate():
            offset = 0
            while True:
                start = offset
                lines, state = runner.since(offset)
                for index, line in enumerate(lines):
                    payload = {"line": line, "index": start + index}
                    yield "data: " + json.dumps(payload) + "\n\n"
                offset = start + len(lines)
                with _session_lock:
                    state["session"] = dict(_session)
                yield "data: " + json.dumps({"status": state}) + "\n\n"
                time.sleep(0.4)

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.post("/actions/reveal")
    def reveal():
        path = _form_path(request.form.get("path") or "")
        if path is None or not path.exists() or not reveal_allowed(path):
            abort(404)
        _show_in_folder(path)
        return _back()

    @app.post("/actions/zip")
    def zip_folder():
        path = _form_path(request.form.get("path") or "")
        if path is None or not analysis_dir_allowed(path):
            abort(404)
        dest = DOWNLOADS / f"replastic-{path.name}.zip"
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in path.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix == ".pt" or file.name == ".DS_Store":
                    continue
                archive.write(file, file.relative_to(path).as_posix())
        return render_template(
            "done.html",
            title="Zip ready",
            detail=str(dest),
            reveal_path=str(dest),
        )

    @app.errorhandler(404)
    def missing(_error):
        return render_template("error.html", message="That page is not here."), 404

    return app


def _chosen_model(raw, models):
    text = (raw or "").replace("\\", "/").strip()
    rels = [model["rel"] for model in models]
    if text in rels:
        return text
    for model in models:
        if model["default"]:
            return model["rel"]
    if rels:
        return rels[0]
    return ""


def _onnx_files():
    models_dir = ROOT / "models"
    if not models_dir.is_dir():
        return []
    found = []
    for path in models_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".onnx") or path.name.endswith(".onnx.json"):
            found.append(path.relative_to(ROOT).as_posix())
    return found


def _default_data():
    present = catalog.present_data_roots()
    if "data_belt_stage1" in present:
        return "data_belt_stage1"
    if present:
        return present[0]
    return DATA_ROOTS[0]


def _chosen_data(raw):
    if raw in DATA_ROOTS and (ROOT / raw).is_dir():
        return raw
    return _default_data()


def _pipeline_data():
    raw = (request.args.get("data") or "").strip()
    if raw not in DATA_ROOTS or not (ROOT / raw).is_dir():
        abort(404)
    return raw


def _belt_cache(data_name):
    return ROOT / "analysis" / "pipeline-cache" / data_name / "belt.json"


def _model_cache(data_name, rel):
    return (
        ROOT
        / "analysis"
        / "pipeline-cache"
        / data_name
        / "models"
        / f"{Path(rel).stem}.json"
    )


def _belt_ready(data_name):
    return _belt_cache(data_name).is_file()


def _scored_model_rels(data_name, models):
    ready = set()
    folder = ROOT / "analysis" / "pipeline-cache" / data_name / "models"
    if folder.is_dir():
        ready = {path.stem for path in folder.glob("*.json")}
    return [item["rel"] for item in models if item["stem"] in ready]


_GRAPH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,60}")


def _pipeline_graph_path(name):
    if not isinstance(name, str) or not _GRAPH_NAME.fullmatch(name):
        return None
    return ROOT / "analysis" / "pipelines" / f"{name}.json"


def _pipeline_graph_names():
    folder = ROOT / "analysis" / "pipelines"
    if not folder.is_dir():
        return []
    names = []
    for path in sorted(folder.glob("*.json")):
        if _GRAPH_NAME.fullmatch(path.stem):
            names.append(path.stem)
    return names


class _ImageForm:
    """The posted form, with an uploaded photo added to the image list."""

    def __init__(self, form, images):
        self._form = form
        self._images = images

    def get(self, key, default=None):
        if key == "images":
            return self._images
        return self._form.get(key, default)

    def getlist(self, key):
        return self._form.getlist(key)


def _job_form():
    upload = request.files.get("photo")
    if upload is None or not (upload.filename or "").strip():
        return request.form
    saved = _save_upload(upload)
    text = (request.form.get("images") or "").strip()
    images = f"{text} {saved}".strip() if text else saved
    return _ImageForm(request.form, images)


def _scorer():
    global _probe_scorer
    with _probe_lock:
        if _probe_scorer is None:
            from web.probe import ProbeScorer

            _probe_scorer = ProbeScorer()
        return _probe_scorer


def _probe_paths():
    paths = []
    skipped = []
    for upload in request.files.getlist("photo"):
        if upload is None or not (upload.filename or "").strip():
            continue
        paths.append(ROOT / _save_upload(upload))
    folder = (request.form.get("folder") or "").strip()
    if folder:
        folder_paths, folder_skipped = _folder_images(folder)
        paths.extend(folder_paths)
        skipped.extend(folder_skipped)
    if not paths:
        if skipped:
            raise CommandError(_unreadable_message(skipped))
        raise CommandError("Add a photo or a folder.")
    if len(paths) > _PROBE_LIMIT:
        raise CommandError(f"Score at most {_PROBE_LIMIT} photos at a time.")
    return paths, skipped


def _folder_images(text):
    if "\x00" in text:
        raise CommandError("That folder is not on disk.")
    path = Path(text)
    if not path.is_absolute():
        if ".." in path.parts:
            raise CommandError("Folder has to stay a real path.")
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    if is_relative_inside(path, ROOT / "models"):
        raise CommandError("That folder is inside models/.")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise CommandError("Use a JPEG, PNG, or WebP.")
        return [_servable_copy(path)], []
    if not path.is_dir():
        raise CommandError("That folder is not on disk.")
    found = []
    skipped = []
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not _image_readable(item):
            skipped.append(_repo_display(item))
            continue
        found.append(item)
        if len(found) > _PROBE_LIMIT:
            raise CommandError(
                f"That folder has more than {_PROBE_LIMIT} images."
            )
    if not found:
        if skipped:
            raise CommandError(_unreadable_message(skipped))
        raise CommandError("That folder has no photos.")
    found.sort()
    skipped.sort()
    return [_servable_copy(item) for item in found], skipped


def _image_readable(path):
    """False for a zero-byte or corrupt photo a folder score should skip."""
    from PIL import Image
    from PIL import UnidentifiedImageError

    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as image:
            image.load()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def _repo_display(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _unreadable_message(skipped):
    shown = ", ".join(skipped[:3])
    more = len(skipped) - 3
    tail = f", and {more} more" if more > 0 else ""
    noun = "file" if len(skipped) == 1 else "files"
    return (
        f"Could not read {len(skipped)} empty or unreadable {noun}: "
        f"{shown}{tail}."
    )


def _servable_copy(path):
    resolved = Path(path).resolve()
    if image_allowed(resolved):
        return resolved
    return ROOT / _copy_probe(resolved)


def _copy_probe(src):
    import shutil
    from PIL import Image
    from PIL import UnidentifiedImageError

    src = Path(src)
    suffix = src.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise CommandError("Use a JPEG, PNG, or WebP.")
    size = src.stat().st_size
    if size == 0 or size > _UPLOAD_LIMIT:
        raise CommandError("That image is empty or larger than 25 MB.")
    safe = re.sub(r"[^A-Za-z0-9.-]+", "-", src.stem).strip(".-")[:60] or "photo"
    folder = ROOT / "analysis" / "_uploads"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = folder / f"{stamp}-{safe}{suffix}"
    if dest.exists():
        dest = folder / f"{stamp}-{time.time_ns()}-{safe}{suffix}"
    shutil.copyfile(src, dest)
    try:
        with Image.open(dest) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        dest.unlink(missing_ok=True)
        raise CommandError("That file is not a readable image.")
    return dest.relative_to(ROOT).as_posix()


def _save_upload(upload):
    from PIL import Image
    from PIL import UnidentifiedImageError

    name = Path(upload.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise CommandError("Use a JPEG, PNG, or WebP.")
    raw_stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9.-]+", "-", raw_stem).strip(".-")[:60] or "photo"
    folder = ROOT / "analysis" / "_uploads"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = folder / f"{stamp}-{safe}{suffix}"
    if dest.exists():
        raise CommandError("That upload name is already in use. Try again.")
    upload.save(dest)
    size = dest.stat().st_size
    if size == 0 or size > _UPLOAD_LIMIT:
        dest.unlink(missing_ok=True)
        raise CommandError("That image is empty or larger than 25 MB.")
    try:
        with Image.open(dest) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        dest.unlink(missing_ok=True)
        raise CommandError("That file is not a readable image.")
    return dest.relative_to(ROOT).as_posix()


def _form_path(raw):
    text = raw.strip()
    if not text or "\x00" in text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _show_in_folder(path):
    resolved = path.resolve()
    if os.name == "nt":
        subprocess.run(["explorer", f"/select,{resolved}"], check=False)
        return
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(resolved)], check=False)
        return
    folder = resolved if resolved.is_dir() else resolved.parent
    subprocess.run(["xdg-open", str(folder)], check=False)


def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def _back():
    target = request.referrer or ""
    if target.startswith(request.host_url):
        return redirect(target)
    return redirect(url_for("jobs_page"))


def _watch_session():
    while True:
        state = _probe_session()
        with _session_lock:
            _session.update(state)
        time.sleep(15)


def _probe_session():
    try:
        result = subprocess.run(
            ["colab", "status", "-s", "trainer"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"up": False, "label": "T4 unavailable"}
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    up = re.search(r"IDLE|BUSY|READY", text) is not None
    return {"up": up, "label": "T4 up" if up else "T4 down"}
