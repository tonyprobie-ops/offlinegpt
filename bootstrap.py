import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1234
BASE_URL = "http://%s:%d/v1" % (SERVER_HOST, SERVER_PORT)
SERVER_START_TIMEOUT = 90.0
MODEL_LOAD_TIMEOUT = 900.0
PROBE_TIMEOUT = 240.0
PROBE_ATTEMPTS = 30
PROBE_SLEEP = 4.0
LMS_ENV_VAR = "LM_STUDIO_LMS"
LMS_CANDIDATES = (
    r"AppData\Local\Programs\LM Studio\resources\app.webpack\lms.exe",
    r"AppData\Local\Programs\LM Studio\resources\app\webpack\lms.exe",
    r"AppData\Local\LM Studio\resources\app.webpack\lms.exe",
    r"Program Files\LM Studio\resources\app.webpack\lms.exe",
)


class BootstrapError(RuntimeError):
    pass


def _log(on_step: Optional[Callable[[str], None]], message: str) -> None:
    if on_step is not None:
        on_step(message)


def decode_output(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def lms_candidates() -> List[str]:
    candidates: List[str] = []
    env_value = os.environ.get(LMS_ENV_VAR)
    if env_value:
        candidates.append(env_value)
    which = shutil.which("lms") or shutil.which("lms.exe")
    if which:
        candidates.append(which)
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("USERPROFILE"), os.path.expanduser("~")):
        if not base:
            continue
        for relative in LMS_CANDIDATES:
            candidates.append(os.path.join(base, relative))
    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def find_lms() -> Optional[str]:
    for candidate in lms_candidates():
        if os.path.isfile(candidate):
            return candidate
    for candidate in lms_candidates():
        found = shutil.which(candidate)
        if found:
            return found
    return None


def run_lms(args: List[str], timeout: float = 120.0, lms_path: Optional[str] = None) -> Tuple[int, str]:
    executable = lms_path or find_lms()
    if not executable:
        raise BootstrapError(
            "The LM Studio CLI (lms.exe) was not found. Start LM Studio once, or set the "
            "%s environment variable to the full path of lms.exe." % LMS_ENV_VAR
        )
    creation_flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creation_flags = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            [executable] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            creationflags=creation_flags,
        )
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError("The command 'lms %s' timed out after %d seconds." % (" ".join(args), int(timeout))) from exc
    except OSError as exc:
        raise BootstrapError("Could not run 'lms %s': %s" % (" ".join(args), exc)) from exc
    return completed.returncode, decode_output(completed.stdout).strip()


def first_json(text: str) -> Any:
    indices = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not indices:
        return None
    start = min(indices)
    pairs = [("{", "}"), ("[", "]")]
    if text[start] == "[":
        pairs.reverse()
    for opener, closer in pairs:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
    return None


def models_url() -> str:
    return "%s/models" % BASE_URL.rstrip("/")


def server_is_up(timeout: float = 2.0) -> bool:
    try:
        response = requests.get(models_url(), timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code < 500


def available_models(timeout: float = 5.0) -> List[str]:
    try:
        response = requests.get(models_url(), timeout=timeout)
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def loaded_models(lms_path: Optional[str] = None) -> List[str]:
    try:
        code, output = run_lms(["ps", "--json"], timeout=60.0, lms_path=lms_path)
    except BootstrapError:
        return []
    if code != 0:
        return []
    payload = first_json(output)
    if payload is None:
        return []
    entries: List[Any] = []
    if isinstance(payload, dict):
        for key in ("models", "data", "loadedModels"):
            if isinstance(payload.get(key), list):
                entries = payload[key]
                break
    elif isinstance(payload, list):
        entries = payload
    ids: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            for key in ("identifier", "modelKey", "model", "id", "path"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
        elif isinstance(entry, str):
            ids.append(entry)
    return ids


def start_server(on_step: Optional[Callable[[str], None]] = None, lms_path: Optional[str] = None) -> None:
    if server_is_up():
        _log(on_step, "Local server already running on port %d." % SERVER_PORT)
        return
    _log(on_step, "Starting the LM Studio local server on port %d..." % SERVER_PORT)
    code, output = run_lms(["server", "start", "--port", str(SERVER_PORT)], timeout=180.0, lms_path=lms_path)
    deadline = time.time() + SERVER_START_TIMEOUT
    while time.time() < deadline:
        if server_is_up(timeout=1.5):
            _log(on_step, "Local server is up.")
            return
        time.sleep(1.5)
    detail = output.strip() or "no output"
    raise BootstrapError("The LM Studio server did not start on port %d. CLI said: %s" % (SERVER_PORT, detail[:400]))


def unload_others(keep: str, on_step: Optional[Callable[[str], None]] = None, lms_path: Optional[str] = None) -> None:
    keep_key = (keep or "").lower()
    for identifier in loaded_models(lms_path=lms_path):
        if keep_key and (identifier.lower() == keep_key or keep_key in identifier.lower()):
            continue
        _log(on_step, "Freeing RAM: unloading %s" % identifier)
        try:
            run_lms(["unload", identifier], timeout=120.0, lms_path=lms_path)
        except BootstrapError as exc:
            _log(on_step, "Could not unload %s (%s). Continuing anyway." % (identifier, exc))


def unload_all(on_step: Optional[Callable[[str], None]] = None, lms_path: Optional[str] = None) -> None:
    for identifier in loaded_models(lms_path=lms_path):
        _log(on_step, "Unloading %s" % identifier)
        try:
            run_lms(["unload", identifier], timeout=120.0, lms_path=lms_path)
        except BootstrapError as exc:
            _log(on_step, "Could not unload %s (%s)." % (identifier, exc))


def is_model_loaded(model: str, available: Optional[List[str]] = None, lms_path: Optional[str] = None) -> bool:
    key = (model or "").lower()
    if not key:
        return False
    names = list(available if available is not None else available_models())
    names.extend(loaded_models(lms_path=lms_path))
    return any(item.lower() == key or key in item.lower() for item in names)


def load_model(model: str, on_step: Optional[Callable[[str], None]] = None, lms_path: Optional[str] = None) -> None:
    if is_model_loaded(model, lms_path=lms_path):
        _log(on_step, "%s is already loaded." % model)
        return
    _log(on_step, "Loading %s into RAM... this can take a while on CPU." % model)
    attempts = [["load", model, "--yes"], ["load", model, "-y"], ["load", model]]
    last_error = ""
    for args in attempts:
        try:
            code, output = run_lms(args, timeout=MODEL_LOAD_TIMEOUT, lms_path=lms_path)
        except BootstrapError as exc:
            last_error = str(exc)
            continue
        if code == 0:
            return
        last_error = output.strip()[:400] or "exit code %d" % code
        lowered = last_error.lower()
        if "already" in lowered and "load" in lowered:
            _log(on_step, "%s was already resident in LM Studio." % model)
            return
        if "unknown" not in lowered and "unrecognized" not in lowered and "usage" not in lowered:
            break
    if is_model_loaded(model, lms_path=lms_path):
        _log(on_step, "%s is reported as available, continuing." % model)
        return
    raise BootstrapError("Could not load '%s' in LM Studio. CLI said: %s" % (model, last_error or "no output"))


def start_server_if_needed(on_step=None, lms_path=None) -> None:
    start_server(on_step=on_step, lms_path=lms_path)


def stop_server(on_step=None, lms_path=None) -> None:
    _log(on_step, "Stopping the LM Studio local server...")
    try:
        code, output = run_lms(["server", "stop"], timeout=120.0, lms_path=lms_path)
    except BootstrapError as exc:
        _log(on_step, str(exc))
        return
    if code != 0:
        _log(on_step, (output or "lms server stop failed").strip()[:300])


def probe_model(model: str, on_step: Optional[Callable[[str], None]] = None) -> bool:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    last_reason = ""
    deadline = time.time() + PROBE_TIMEOUT
    attempt = 0
    while attempt < PROBE_ATTEMPTS and time.time() < deadline:
        attempt += 1
        try:
            response = requests.post(
                "%s/chat/completions" % BASE_URL,
                json=payload,
                timeout=60.0,
            )
        except requests.RequestException as exc:
            last_reason = "server not reachable (%s)" % exc
        else:
            if response.status_code == 200:
                return True
            last_reason = "HTTP %d" % response.status_code
            if response.status_code in (404, 400) and attempt < PROBE_ATTEMPTS:
                last_reason = "HTTP %d (model still loading)" % response.status_code
        if on_step is not None and attempt in (1, 5, 10, 20):
            _log(on_step, "Waiting for %s to answer (attempt %d): %s" % (model, attempt, last_reason))
        time.sleep(PROBE_SLEEP)
    raise BootstrapError("The model '%s' did not respond within %d seconds. Last error: %s" % (model, int(PROBE_TIMEOUT), last_reason or "unknown"))


def ensure_ready_with_reason(
    model: str,
    on_step: Optional[Callable[[str], None]] = None,
    lms_path: Optional[str] = None,
) -> Tuple[bool, str]:
    if not model or not str(model).strip():
        return False, "No model was selected."
    model = str(model).strip()
    try:
        _log(on_step, "Checking the LM Studio local server...")
        start_server(on_step=on_step, lms_path=lms_path)
        unload_others(model, on_step=on_step, lms_path=lms_path)
        load_model(model, on_step=on_step, lms_path=lms_path)
        _log(on_step, "Sending a one-token probe to %s..." % model)
        probe_model(model, on_step=on_step)
    except BootstrapError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)
    _log(on_step, "%s is ready." % model)
    return True, ""


def ensure_ready(model: str, on_step: Optional[Callable[[str], None]] = None, lms_path: Optional[str] = None) -> bool:
    ok, _ = ensure_ready_with_reason(model, on_step=on_step, lms_path=lms_path)
    return ok


def server_state() -> Dict[str, Any]:
    return {
        "base_url": BASE_URL,
        "running": server_is_up(),
        "models": available_models(),
        "lms": find_lms() or "",
        "loaded": loaded_models(),
    }


def build_worker_class():
    from PySide6.QtCore import QThread, Signal

    class BootstrapWorker(QThread):
        step = Signal(str)
        done = Signal(bool, str)

        def __init__(self, model: str, lms_path: Optional[str] = None, parent: Any = None) -> None:
            super().__init__(parent)
            self.model = model
            self.lms_path = lms_path

        def run(self) -> None:
            ok, message = ensure_ready_with_reason(self.model, on_step=self.step.emit, lms_path=self.lms_path)
            self.done.emit(ok, message)

    return BootstrapWorker


try:
    BootstrapWorker = build_worker_class()
except Exception:
    BootstrapWorker = None
