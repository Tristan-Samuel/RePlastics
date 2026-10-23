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
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from web import catalog
from web.commands import GPUS, STEPS, UNFREEZE, CommandError, onnx_display, prepare
from web.jobs import BusyError, runner
from web.paths import (
    BELT_ZIP,
    DATA_ROOTS,
    DOWNLOADS,
    ROOT,
    analysis_dir_allowed,
    image_allowed,
    reveal_allowed,
)
from web.presets import PRESETS

_session = {"up": False, "label": "T4 …"}
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
        onnx_path = ROOT / "models" / "resnext50_metal_plastic.onnx"
        json_path = ROOT / "models" / "resnext50_metal_plastic.onnx.json"
        return render_template(
            "jetson.html",
            export_command=onnx_display(),
            onnx_rel="models/resnext50_metal_plastic.onnx",
            json_rel="models/resnext50_metal_plastic.onnx.json",
            onnx_ready=onnx_path.is_file(),
            json_ready=json_path.is_file(),
        )

    @app.get("/models")
    def models_page():
        return render_template(
            "models.html",
            models=catalog.list_models(),
            others=catalog.other_analyses(),
        )

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
        path = ROOT / rel
        if not image_allowed(path):
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
        try:
            prepared = prepare(request.form, require_confirm=True)
            runner.start(prepared)
        except CommandError as error:
            return render_template("error.html", message=str(error)), 400
        except BusyError as error:
            return (
                render_template(
                    "error.html",
                    message=f"{error} is already running.",
                ),
                409,
            )
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
