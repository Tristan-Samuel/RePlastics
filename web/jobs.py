"""One subprocess at a time, with a log the browser can follow."""

import json
import os
import re
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from web.paths import JOB_LOGS, ROOT


_SCORE = re.compile(
    r"^(?P<engine>PyTorch|ONNX): accuracy (?P<accuracy>\S+) \| "
    r"balanced (?P<balanced>\S+)(?: \| (?P<detail>.*))?$"
)
_DISAGREE = re.compile(
    r"^Argmax disagreements: (?P<argmax>\d+)\. "
    r"Hold disagreements: (?P<hold>\d+)\. "
    r"Uncertain: PyTorch (?P<torch_uncertain>\d+), "
    r"ONNX (?P<onnx_uncertain>\d+)\.$"
)
_WORST = re.compile(r"^Worst max abs logit diff (?P<diff>\S+)\.$")
_MATCH = re.compile(r"^ONNX matches PyTorch on (?P<count>\d+) test image")


def latest_onnx_check(onnx_rel):
    """The newest finished ONNX check for this file, from the job logs."""
    needle = (onnx_rel or "").replace("\\", "/").strip()
    if not needle or not JOB_LOGS.is_dir():
        return None
    logs = sorted(
        JOB_LOGS.glob("*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in logs[:40]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if needle not in text.replace("\\", "/"):
            continue
        parsed = _parse_onnx_check(text)
        if parsed is None:
            continue
        parsed["when"] = datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%d %b %Y %H:%M"
        )
        return parsed
    return None


def _parse_onnx_check(text):
    scores = {}
    disagree = worst = count = None
    failed = False
    for line in text.splitlines():
        score = _SCORE.match(line.strip())
        if score:
            scores[score.group("engine")] = {
                "accuracy": score.group("accuracy"),
                "balanced": score.group("balanced"),
                "detail": (score.group("detail") or "").replace(" | ", " · "),
            }
            continue
        found = _DISAGREE.match(line.strip())
        if found:
            disagree = found.groupdict()
            continue
        found = _WORST.match(line.strip())
        if found:
            worst = found.group("diff").rstrip(".")
            continue
        found = _MATCH.match(line.strip())
        if found:
            count = found.group("count")
            continue
        if line.startswith("ONNX labels do not match"):
            failed = True
    torch_score = scores.get("PyTorch")
    onnx_score = scores.get("ONNX")
    if torch_score is None and onnx_score is None and not failed:
        return None
    if disagree and (
        disagree["argmax"] != "0" or disagree["hold"] != "0"
    ):
        failed = True
    detail = ""
    if torch_score and onnx_score:
        if torch_score["detail"] == onnx_score["detail"]:
            detail = torch_score["detail"]
        else:
            detail = (
                f"PyTorch {torch_score['detail']}. "
                f"ONNX {onnx_score['detail']}."
            ).strip()
    elif torch_score:
        detail = torch_score["detail"]
    elif onnx_score:
        detail = onnx_score["detail"]
    uncertain = ""
    if disagree:
        if disagree["torch_uncertain"] == disagree["onnx_uncertain"]:
            uncertain = f"Uncertain {disagree['torch_uncertain']}."
        else:
            uncertain = (
                f"Uncertain: PyTorch {disagree['torch_uncertain']}, "
                f"ONNX {disagree['onnx_uncertain']}."
            )
    if failed:
        parts = ["ONNX labels do not match PyTorch."]
        if disagree:
            parts.append(
                f"Argmax disagreements {disagree['argmax']}. "
                f"Hold disagreements {disagree['hold']}."
            )
        parts.append("Leave this file off the Jetson.")
        note = " ".join(parts)
    elif count:
        parts = [
            f"Matches on {count} test images. Same argmax and the same hold."
        ]
        if uncertain:
            parts.append(uncertain)
        if worst:
            parts.append(f"Worst logit diff {worst}.")
        note = " ".join(parts)
    else:
        note = uncertain
    return {
        "torch_accuracy": (torch_score or {}).get("accuracy") or "—",
        "torch_balanced": (torch_score or {}).get("balanced") or "—",
        "onnx_accuracy": (onnx_score or {}).get("accuracy") or "—",
        "onnx_balanced": (onnx_score or {}).get("balanced") or "—",
        "detail": detail,
        "note": note,
        "failed": failed,
    }


def _alive(pid):
    """True when pid is still running. On Windows, signal 0 kills the process."""
    if os.name == "nt":
        import ctypes

        query = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
        handle = ctypes.windll.kernel32.OpenProcess(query, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_tree(pid):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/PID", str(pid)],
            check=False,
            capture_output=True,
        )
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


class Runner:
    def __init__(self):
        self._lock = threading.Lock()
        self.lines = []
        self.proc = None
        self.running = False
        self.title = ""
        self.command = ""
        self.exit_code = None
        self.status_line = ""
        self.started = ""
        self._orphan_pid = None
        self._log_path = None
        self._record = None
        JOB_LOGS.mkdir(parents=True, exist_ok=True)
        self._load_orphan()

    def _state_path(self):
        return JOB_LOGS / "current.json"

    def _load_orphan(self):
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        pid = data.get("pid")
        if not pid or not _alive(pid):
            path.unlink(missing_ok=True)
            return
        self.running = True
        self.title = data.get("title") or "Job"
        self.command = data.get("command") or ""
        self.started = data.get("started") or ""
        self._orphan_pid = pid
        log_path = data.get("log")
        if log_path:
            self._log_path = JOB_LOGS / log_path
            self.lines = self._read_log(self._log_path)
        self.status_line = (
            "Server restarted. That process is still running, "
            "so new lines are not arriving."
        )
        threading.Thread(
            target=self._watch_orphan,
            args=(pid,),
            daemon=True,
        ).start()

    def _read_log(self, path):
        if not path or not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        return text.splitlines()

    def _watch_orphan(self, pid):
        while _alive(pid):
            threading.Event().wait(2)
        with self._lock:
            if self._orphan_pid == pid:
                self.running = False
                self.exit_code = None
                self._orphan_pid = None
                self.status_line = "The earlier process has exited."
                self._state_path().unlink(missing_ok=True)

    def public_state(self):
        with self._lock:
            return {
                "running": self.running,
                "title": self.title,
                "command": self.command,
                "exit_code": self.exit_code,
                "status_line": self.status_line,
                "started": self.started,
                "line_count": len(self.lines),
            }

    def since(self, offset):
        with self._lock:
            offset = max(0, min(offset, len(self.lines)))
            return list(self.lines[offset:]), self.public_state_unlocked()

    def public_state_unlocked(self):
        return {
            "running": self.running,
            "title": self.title,
            "command": self.command,
            "exit_code": self.exit_code,
            "status_line": self.status_line,
            "started": self.started,
            "line_count": len(self.lines),
        }

    def text(self):
        with self._lock:
            return "\n".join(self.lines)

    def start(self, prepared):
        with self._lock:
            if self.running:
                raise BusyError(self.title or "A job")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.lines = []
            self.exit_code = None
            self.title = prepared.title
            self.command = prepared.display
            self.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status_line = "Starting"
            self._orphan_pid = None
            self._record = prepared.record
            self._log_path = JOB_LOGS / f"{stamp}.log"
            self._append_unlocked(prepared.display)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["MPLBACKEND"] = "Agg"
            popen = {
                "cwd": str(ROOT),
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "env": env,
            }
            if os.name == "nt":
                popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen["start_new_session"] = True
            try:
                self.proc = subprocess.Popen(prepared.argv, **popen)
            except OSError as error:
                self.running = False
                self.exit_code = 1
                self.status_line = "Failed to start"
                self._append_unlocked(str(error))
                return
            self.running = True
            self._write_state(self.proc.pid)
        threading.Thread(target=self._read, daemon=True).start()

    def _write_state(self, pid):
        payload = {
            "pid": pid,
            "title": self.title,
            "command": self.command,
            "started": self.started,
            "log": self._log_path.name if self._log_path else "",
        }
        self._state_path().write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def _append_unlocked(self, line):
        self.lines.append(line)
        self._note_status(line)
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _note_status(self, line):
        if "batch " in line and "/" in line:
            self.status_line = line.strip()
        if "balanced " in line:
            self.status_line = line.strip()

    def _read(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            with self._lock:
                self._append_unlocked(line)
        code = proc.wait()
        with self._lock:
            self.exit_code = code
            self.running = False
            self.proc = None
            if code == 0:
                if "accuracy" not in self.status_line:
                    self.status_line = "Finished"
            else:
                self.status_line = f"Exited {code}"
            self._save_record()
            self._state_path().unlink(missing_ok=True)

    def stop(self):
        with self._lock:
            pid = None
            if self.proc is not None and self.proc.poll() is None:
                pid = self.proc.pid
            elif self._orphan_pid and _alive(self._orphan_pid):
                pid = self._orphan_pid
            if pid is None:
                return False
        return _stop_tree(pid)

    def write_stdin(self, text):
        with self._lock:
            proc = self.proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise BusyError("No running job is waiting for input.")
            payload = text
            if not payload.endswith("\n"):
                payload += "\n"
        try:
            proc.stdin.write(payload.encode("utf-8"))
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise BusyError("The job is no longer reading input.")
        with self._lock:
            self._append_unlocked("[sent input]")

    def _save_record(self):
        record = getattr(self, "_record", None)
        if not record:
            return
        path = Path(record)
        if path.is_absolute() or ".." in path.parts:
            return
        dest = ROOT / path
        try:
            dest.resolve().relative_to((ROOT / "analysis").resolve())
        except ValueError:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


class BusyError(Exception):
    pass


runner = Runner()
