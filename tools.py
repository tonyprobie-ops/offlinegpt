import fnmatch
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 400_000
MAX_LIST_ENTRIES = 400
MAX_SEARCH_HITS = 120
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "out",
    "target",
    ".idea",
    ".vscode",
}
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".bin",
    ".pyc",
    ".mp3",
    ".mp4",
    ".wav",
    ".sqlite",
    ".db",
}


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: Optional[str] = None) -> None:
        self._root: Optional[Path] = None
        if root:
            self.set_root(root)

    @property
    def root(self) -> Optional[Path]:
        return self._root

    def set_root(self, root: Optional[str]) -> Optional[Path]:
        if not root:
            self._root = None
            return None
        candidate = Path(os.path.expanduser(str(root))).resolve()
        self._root = candidate
        return candidate

    def is_open(self) -> bool:
        return self._root is not None

    def resolve(self, relative: str = ".", must_exist: bool = False) -> Path:
        if self._root is None:
            raise ToolError("No project folder is open. Use 'Open project folder' in the sidebar first.")
        if relative is None:
            relative = "."
        raw = str(relative).strip().strip('"').strip("'")
        if raw in ("", "."):
            raw = "."
        raw = raw.replace("\\", "/")
        if raw.startswith("/"):
            raise ToolError("Paths must be relative to the open project folder, for example 'src/main.py'.")
        if re.match(r"^[a-zA-Z]:", raw):
            raise ToolError("Paths must be relative to the open project folder, not absolute Windows paths.")
        target = (self._root / raw).resolve()
        if target != self._root and self._root not in target.parents:
            raise ToolError("That path points outside the open project folder, so it was blocked.")
        if must_exist and not target.exists():
            raise ToolError("'%s' does not exist inside the project folder." % raw)
        return target

    def relative(self, path: Path) -> str:
        if self._root is None:
            return str(path)
        try:
            return str(path.relative_to(self._root)) or "."
        except ValueError:
            return str(path)


def _is_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return False
    return b"\x00" in chunk


def _looks_binary_extension(path: Path) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_READ_BYTES:
        data = data[:MAX_READ_BYTES]
    return data.decode("utf-8", errors="replace")


def _skip_dir(path: Path) -> bool:
    return path.name in SKIP_DIR_NAMES


def _walk(root: Path, limit: int = 4000) -> List[Path]:
    results: List[Path] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        for name in sorted(filenames):
            results.append(Path(current) / name)
            if len(results) >= limit:
                return results
    return results


class ToolResult:
    def __init__(self, ok: bool, text: str) -> None:
        self.ok = ok
        self.text = text

    def as_payload(self) -> Dict[str, Any]:
        return {"ok": self.ok, "result": self.text}


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], ToolResult],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: Dict[str, Any]) -> ToolResult:
        try:
            return self.handler(arguments or {})
        except ToolError as exc:
            return ToolResult(False, "Error: %s" % exc)
        except OSError as exc:
            return ToolResult(False, "Error: Windows reported %s" % exc)
        except Exception as exc:
            return ToolResult(False, "Error: %s: %s" % (type(exc).__name__, exc))


def _arg(args: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = args.get(key, default)
    if isinstance(value, str):
        value = value.strip()
    return value


def _require_str(args: Dict[str, Any], key: str) -> str:
    value = _arg(args, key)
    if not isinstance(value, str) or not value:
        raise ToolError("The argument '%s' is required and must be a non-empty string." % key)
    return value


def list_dir_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    path = _arg(args, "path", ".") or "."
    target = workspace.resolve(path, must_exist=True)
    if target.is_file():
        raise ToolError("'%s' is a file, not a folder." % path)
    lines: List[str] = []
    entries = sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    hidden = 0
    for entry in entries:
        if len(lines) >= MAX_LIST_ENTRIES:
            hidden += 1
            continue
        if entry.is_dir():
            lines.append("[dir]  %s/" % entry.name)
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            lines.append("[file] %s  (%d bytes)" % (entry.name, size))
    header = "Contents of %s (%d entries)" % (workspace.relative(target), len(entries))
    if hidden:
        header += ", showing the first %d" % MAX_LIST_ENTRIES
    return ToolResult(True, header + ":\n" + ("\n".join(lines) if lines else "(empty folder)"))


def read_file_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    relative = _require_str(args, "path")
    target = workspace.resolve(relative, must_exist=True)
    if target.is_dir():
        raise ToolError("'%s' is a folder. Use list_dir to inspect folders." % relative)
    if _is_binary(target):
        raise ToolError("'%s' looks like a binary file, so it was not read." % relative)
    max_lines = _arg(args, "max_lines", 0)
    try:
        max_lines = int(max_lines or 0)
    except (TypeError, ValueError):
        max_lines = 0
    text = _read_text(target)
    lines = text.splitlines()
    truncated = False
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    numbered = "\n".join("%4d| %s" % (index + 1, line) for index, line in enumerate(lines))
    header = "File %s (%d bytes, %d lines shown)" % (workspace.relative(target), target.stat().st_size, len(lines))
    if truncated:
        header += " - truncated at max_lines"
    return ToolResult(True, header + "\n" + numbered)


def write_file_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    relative = _require_str(args, "path")
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("The argument 'content' is required and must be a string.")
    target = workspace.resolve(relative)
    if target.exists() and target.is_dir():
        raise ToolError("'%s' is a folder, not a file." % relative)
    payload = content.encode("utf-8")
    if len(payload) > MAX_WRITE_BYTES:
        raise ToolError("That file is larger than the %d byte limit for a single write." % MAX_WRITE_BYTES)
    existed = target.exists()
    previous = len(target.read_bytes()) if existed else 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    action = "Updated" if existed else "Created"
    return ToolResult(
        True,
        "%s %s (%d bytes written, %d lines)." % (
            action,
            workspace.relative(target),
            len(payload),
            content.count("\n") + 1 if content else 0,
        )
        + (" Previous size was %d bytes." % previous if existed else ""),
    )


def search_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    needle = _require_str(args, "query")
    pattern = "*%s*" % _arg(args, "glob", "*") if _arg(args, "glob") else "*"
    scope = _arg(args, "path", ".") or "."
    target = workspace.resolve(scope, must_exist=True)
    files = [target] if target.is_file() else _walk(target)
    matches: List[str] = []
    scanned = 0
    for path in files:
        if len(matches) >= MAX_SEARCH_HITS:
            break
        if _looks_binary_extension(path) or not path.is_file():
            continue
        if pattern and pattern != "*" and not fnmatch.fnmatch(path.name, pattern):
            continue
        scanned += 1
        try:
            text = _read_text(path)
        except OSError:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if needle.lower() in line.lower():
                matches.append("%s:%d: %s" % (workspace.relative(path), index, line.strip()[:200]))
                if len(matches) >= MAX_SEARCH_HITS:
                    break
    header = "Search for %r in %s across %d files" % (needle, workspace.relative(target), scanned)
    if not matches:
        return ToolResult(True, header + " found no matches.")
    return ToolResult(True, header + " found %d matches:\n" % len(matches) + "\n".join(matches))


def make_directory_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    relative = _require_str(args, "path")
    target = workspace.resolve(relative)
    existed = target.exists()
    target.mkdir(parents=True, exist_ok=True)
    return ToolResult(
        True,
        "%s folder %s." % ("Folder already existed:" if existed else "Created", workspace.relative(target)),
    )


def delete_file_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    if not bool(args.get("confirm")):
        raise ToolError("Set 'confirm' to true to delete a file. This action cannot be undone.")
    relative = _require_str(args, "path")
    target = workspace.resolve(relative, must_exist=True)
    if target.is_dir():
        raise ToolError("delete_file only removes single files, never folders: '%s' is a folder." % relative)
    size = target.stat().st_size
    target.unlink()
    return ToolResult(True, "Deleted %s (%d bytes)." % (workspace.relative(target), size))


def file_info_handler(workspace: Workspace, args: Dict[str, Any]) -> ToolResult:
    relative = _require_str(args, "path")
    target = workspace.resolve(relative, must_exist=True)
    stat = target.stat()
    kind = "folder" if target.is_dir() else "file"
    lines = [
        "Path: %s" % workspace.relative(target),
        "Kind: %s" % kind,
        "Size: %d bytes" % stat.st_size,
        "Modified: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    ]
    if target.is_file():
        lines.append("Binary: %s" % ("yes" if _is_binary(target) else "no"))
    return ToolResult(True, "\n".join(lines))


def _string_param(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}


def build_tools(workspace: Workspace) -> List[Tool]:
    return [
        Tool(
            "list_dir",
            "List the files and folders inside the open project folder. Path is relative to the project root.",
            {
                "type": "object",
                "properties": {"path": _string_param("Folder to list, relative to the project root. Defaults to '.'")},
                "required": [],
            },
            lambda args: list_dir_handler(workspace, args),
        ),
        Tool(
            "read_file",
            "Read a text file from the open project folder and return it with line numbers.",
            {
                "type": "object",
                "properties": {
                    "path": _string_param("File to read, relative to the project root."),
                    "max_lines": {"type": "integer", "description": "Optional cap on the number of lines returned."},
                },
                "required": ["path"],
            },
            lambda args: read_file_handler(workspace, args),
        ),
        Tool(
            "write_file",
            "Create or overwrite a text file inside the open project folder, including parent folders.",
            {
                "type": "object",
                "properties": {
                    "path": _string_param("File to write, relative to the project root."),
                    "content": _string_param("Full content of the file."),
                },
                "required": ["path", "content"],
            },
            lambda args: write_file_handler(workspace, args),
        ),
        Tool(
            "search_text",
            "Case-insensitive text search across the files of the open project folder.",
            {
                "type": "object",
                "properties": {
                    "query": _string_param("Text to search for."),
                    "path": _string_param("Folder or file to search in. Defaults to the project root."),
                    "glob": _string_param("Optional file name filter, for example '*.py'."),
                },
                "required": ["query"],
            },
            lambda args: search_handler(workspace, args),
        ),
        Tool(
            "make_directory",
            "Create a folder (and any missing parents) inside the open project folder.",
            {
                "type": "object",
                "properties": {"path": _string_param("Folder to create, relative to the project root.")},
                "required": ["path"],
            },
            lambda args: make_directory_handler(workspace, args),
        ),
        Tool(
            "file_info",
            "Report size, type and modification time for a file or folder inside the open project folder.",
            {
                "type": "object",
                "properties": {"path": _string_param("Path to inspect, relative to the project root.")},
                "required": ["path"],
            },
            lambda args: file_info_handler(workspace, args),
        ),
        Tool(
            "delete_file",
            "Delete a single file inside the open project folder. Requires confirm=true and never touches folders.",
            {
                "type": "object",
                "properties": {
                    "path": _string_param("File to delete, relative to the project root."),
                    "confirm": {"type": "boolean", "description": "Must be true; asks for explicit confirmation."},
                },
                "required": ["path", "confirm"],
            },
            lambda args: delete_file_handler(workspace, args),
        ),
    ]


def tool_specs(tools: List[Tool]) -> List[Dict[str, Any]]:
    return [tool.spec() for tool in tools]


def tool_names(tools: List[Tool]) -> List[str]:
    return [tool.name for tool in tools]


def find_tool(tools: List[Tool], name: str) -> Optional[Tool]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def describe_arguments(raw: str) -> Tuple[str, Dict[str, Any]]:
    text = (raw or "").strip() or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text, {}
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False), parsed
    return json.dumps(parsed, ensure_ascii=False), {}
