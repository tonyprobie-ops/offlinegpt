import base64
import json
import os
import re
import sys
import uuid
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import bootstrap
from agent import AgentRunner, default_system_prompt
from llm import (
    LLMClient,
    MAX_CHAT_CHARS,
    NO_CONTENT_MESSAGE,
    reply_token_budget,
)
from tools import Workspace, build_tools

APP_NAME = "Offline AI Studio"
APP_SLUG = "OfflineAIStudio"
VERSION = "1.0"
DEFAULT_MODEL = "qwen/qwen3-8b"
DEFAULT_MAX_TOKENS = 2048
MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 12_000
TITLE_MAX = 52
ATTACH_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*.*)"
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

BG = "#0b0d12"
SURFACE = "#12151d"
SURFACE_ALT = "#171b25"
SURFACE_HI = "#1d2231"
BORDER = "#252b3a"
TEXT = "#e7eaf1"
MUTED = "#8d95a8"
ACCENT = "#5b8cff"
ACCENT_DARK = "#3f6ede"
CYAN = "#22d3ee"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f8717a"
CODE_BG = "#0f1218"

MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "qwen/qwen3-8b",
        "name": "Qwen3 8B",
        "tag": "Default - fastest",
        "vision": False,
        "description": (
            "The quick one on this machine: roughly 3 tokens per second on a Ryzen 5 5600GT, about 5 GiB of "
            "RAM. Best choice for everyday chat, quick reasoning and running the project tools. Thinking mode "
            "is switched off so it starts answering immediately instead of burning CPU tokens."
        ),
    },
    {
        "id": "qwen2.5-coder-7b-instruct",
        "name": "Qwen2.5 Coder 7B Instruct",
        "tag": "Coding",
        "vision": False,
        "description": (
            "Tuned for code. Reads, explains, writes and refactors Python and other languages, and it pairs "
            "well with the file tools when you open a project folder. Slightly slower than the 8B but much "
            "more reliable on syntax and structure."
        ),
    },
    {
        "id": "google/gemma-4-12b",
        "name": "Gemma 4 12B",
        "tag": "Writing - vision",
        "vision": True,
        "description": (
            "The best writer here: reports, documentation, emails and long explanations. Vision-capable, so it "
            "can look at screenshots and photos you attach. Uses about 7.6 GiB of RAM, so close other apps "
            "first - on 16 GB it only fits when nothing else is resident."
        ),
    },
    {
        "id": "google/gemma-3-12b",
        "name": "Gemma 3 12B",
        "tag": "General chat - vision",
        "vision": True,
        "description": (
            "General-purpose chat with image understanding. Good middle ground between the fast 8B and the "
            "writer: same 7.6 GiB RAM footprint, a little lighter on very long documents."
        ),
    },
]

WELCOME_FEATURES = [
    ("Ask anything", "Questions, explanations and step-by-step reasoning stay on this PC."),
    ("Coding help", "Paste code or open a folder and let the tools read, search and edit files."),
    ("Image analysis", "Attach a screenshot or photo - the two Gemma models can see it."),
    ("Teaching partner", "Turn a topic into a lesson, a quiz or a worked example."),
]

QUICK_ACTIONS = [
    ("Open project folder", "project"),
    ("Attach an image", "attach"),
    ("Change model", "model"),
]


def icon_pixmap(size: int = 64) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(SURFACE_HI))
    painter.setPen(QColor(ACCENT))
    painter.drawRoundedRect(2, 2, size - 4, size - 4, size * 0.22, size * 0.22)
    painter.setPen(QColor(CYAN))
    font = QFont("Segoe UI", int(size * 0.42), QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "O")
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    return QIcon(icon_pixmap())


def app_data_dir() -> Path:
    override = os.environ.get("OFFLINE_AI_STUDIO_HOME")
    if override:
        path = Path(override)
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        path = Path(base) / APP_SLUG
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
        path = Path(base) / APP_SLUG
    path.mkdir(parents=True, exist_ok=True)
    return path


def sessions_path() -> Path:
    return app_data_dir() / "sessions.json"


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def load_settings() -> Dict[str, Any]:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: Dict[str, Any]) -> None:
    path = settings_path()
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setWindowIcon(app_icon())
    base_font = QFont("Segoe UI", 10)
    base_font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(base_font)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(SURFACE))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE_ALT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.PlaceholderText, QColor(MUTED))
    palette.setColor(QPalette.Button, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Link, QColor(CYAN))
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE_HI))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#5b6274"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5b6274"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#5b6274"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = """
QWidget { background-color: %(bg)s; color: %(text)s; }
QMainWindow, QDialog { background-color: %(bg)s; }
QFrame#Sidebar { background-color: %(surface)s; border-right: 1px solid %(border)s; }
QFrame#TopBar { background-color: %(surface)s; border-bottom: 1px solid %(border)s; }
QFrame#Composer { background-color: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
QFrame#Card { background-color: %(alt)s; border: 1px solid %(border)s; border-radius: 10px; }
QFrame#Bubble { border: 1px solid %(border)s; border-radius: 12px; }
QFrame#BubbleUser { background-color: #1a2340; border: 1px solid #2c3a63; border-radius: 12px; }
QFrame#BubbleAssistant { background-color: %(alt)s; border: 1px solid %(border)s; border-radius: 12px; }
QFrame#BubbleNote { background-color: #1d1a12; border: 1px solid #4a3a12; border-radius: 12px; }
QFrame#BubbleError { background-color: #2a1418; border: 1px solid #5c2530; border-radius: 12px; }
QFrame#ToolCard { background-color: %(alt)s; border: 1px solid %(border)s; border-radius: 10px; }
QFrame#Reasoning { background-color: #141826; border: 1px solid #2a3350; border-radius: 10px; }
QFrame#AttachChip { background-color: %(hi)s; border: 1px solid %(border)s; border-radius: 8px; }
QLabel { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: 700; color: #ffffff; }
QLabel#Subtitle { color: %(muted)s; font-size: 9pt; }
QLabel#Caption { color: %(muted)s; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Heading { font-size: 12pt; font-weight: 700; color: #ffffff; }
QLabel#Muted { color: %(muted)s; }
QLabel#Chip { background-color: %(hi)s; color: #cdd4e4; border: 1px solid %(border)s;
              border-radius: 8px; padding: 2px 8px; font-size: 9pt; }
QLabel#StatusOk { color: %(green)s; font-size: 9pt; }
QLabel#StatusWarn { color: %(amber)s; font-size: 9pt; }
QLabel#StatusBad { color: %(red)s; font-size: 9pt; }
QLabel#ToolName { font-weight: 700; color: #ffffff; }
QPushButton { background-color: %(hi)s; color: %(text)s; border: 1px solid %(border)s;
              border-radius: 8px; padding: 7px 12px; }
QPushButton:hover { background-color: #232a3c; border-color: #38415a; }
QPushButton:pressed { background-color: #1a2030; }
QPushButton:disabled { background-color: %(surface)s; color: #5b6274; border-color: #1e2431; }
QPushButton[class="accent"] { background-color: %(accent)s; border: 1px solid %(accent)s; color: #ffffff;
                              font-weight: 700; }
QPushButton[class="accent"]:hover { background-color: #6f9bff; border-color: #6f9bff; }
QPushButton[class="accent"]:pressed { background-color: %(accent_dark)s; }
QPushButton[class="accent"]:disabled { background-color: #26304a; border-color: #26304a; color: #7b8499; }
QPushButton[class="ghost"] { background-color: transparent; border: 1px solid %(border)s; color: #cbd2e2; }
QPushButton[class="ghost"]:hover { background-color: %(hi)s; }
QPushButton[class="danger"] { background-color: #35171d; border: 1px solid #5c2530; color: #ffbcc2; }
QPushButton[class="link"] { background: transparent; border: none; color: %(cyan)s; padding: 2px 4px; }
QPushButton[class="link"]:hover { color: #7ce7f7; }
QToolButton { background-color: transparent; border: 1px solid %(border)s; border-radius: 8px;
              padding: 4px 10px; color: #cbd2e2; }
QToolButton:hover { background-color: %(hi)s; }
QToolButton:checked { background-color: #1c2740; border-color: %(accent)s; color: #ffffff; }
QLineEdit, QTextEdit, QPlainTextEdit { background-color: %(code_bg)s; border: 1px solid %(border)s;
              border-radius: 8px; padding: 6px 8px; selection-background-color: %(accent)s; }
QLineEdit:focus, QTextEdit:focus { border-color: %(accent)s; }
QTextEdit#ComposerInput { background: transparent; border: none; padding: 4px 2px; font-size: 10.5pt; }
QTextEdit#LogView { background-color: %(code_bg)s; color: #c8d2e0; font-family: Consolas, "Cascadia Mono", monospace;
                    font-size: 9pt; border: 1px solid %(border)s; }
QListWidget { background-color: transparent; border: none; outline: none; }
QListWidget::item { padding: 8px 10px; border-radius: 8px; color: #cbd2e2; }
QListWidget::item:hover { background-color: %(hi)s; }
QListWidget::item:selected { background-color: #1c2740; color: #ffffff; border: 1px solid #2c3a63; }
QListWidget#ModelList::item { padding: 10px 12px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2b3243; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3a445c; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #2b3243; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QMenu { background-color: %(surface)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 18px; border-radius: 6px; }
QMenu::item:selected { background-color: %(hi)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 8px; }
QToolTip { background-color: %(hi)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px 6px; }
QMessageBox { background-color: %(surface)s; }
QMessageBox QLabel { color: %(text)s; }
QMessageBox QPushButton { min-width: 84px; }
QProgressBar { background-color: %(code_bg)s; border: 1px solid %(border)s; border-radius: 6px;
               height: 10px; text-align: center; color: transparent; }
QProgressBar::chunk { background-color: %(accent)s; border-radius: 5px; }
QCheckBox { color: #cbd2e2; spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px; border: 1px solid %(border)s;
                       background-color: %(code_bg)s; }
QCheckBox::indicator:checked { background-color: %(accent)s; border-color: %(accent)s; }
QSplitter::handle { background-color: %(border)s; }
""" % {
    "bg": BG,
    "surface": SURFACE,
    "alt": SURFACE_ALT,
    "hi": SURFACE_HI,
    "border": BORDER,
    "text": TEXT,
    "muted": MUTED,
    "accent": ACCENT,
    "accent_dark": ACCENT_DARK,
    "cyan": CYAN,
    "green": GREEN,
    "amber": AMBER,
    "red": RED,
    "code_bg": CODE_BG,
}


def model_info(model_id: str) -> Dict[str, Any]:
    for entry in MODEL_CATALOG:
        if entry["id"] == model_id:
            return entry
    return {
        "id": model_id,
        "name": model_id,
        "tag": "Custom model",
        "vision": "gemma" in (model_id or "").lower(),
        "description": (
            "A model you typed in yourself. The app will load it in LM Studio exactly like the built-in ones; "
            "image support depends on the model."
        ),
    }


def model_supports_vision(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return "gemma" in lowered or bool(model_info(model_id).get("vision"))


def short_model_name(model_id: str) -> str:
    info = model_info(model_id)
    return str(info.get("name") or model_id)


def elide(text: str, width: int, widget: Optional[QWidget] = None) -> str:
    metrics = QFontMetrics(widget.font() if widget is not None else QApplication.font())
    return metrics.elidedText(text, Qt.ElideMiddle, max(40, width))


FENCE_PATTERN = re.compile(r"```([a-zA-Z0-9_+\-.#]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
BLOCK_MARK = "\n\u0000CB%d\u0000\n"
INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
STRIKE = re.compile(r"~~([^~]+)~~")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
ORDERED = re.compile(r"^\s*(\d{1,3})[.)]\s+(.*)$")
RULE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]{3,}\|?\s*$")
CODE_STYLE = (
    "background-color:%s; color:#dbe4f0; border:1px solid %s; border-radius:6px;"
    " padding:8px; font-family:Consolas,'Cascadia Mono','Courier New',monospace; font-size:9.5pt;"
) % (CODE_BG, BORDER)
INLINE_STYLE = "background-color:%s; color:#e4b8ff; font-family:Consolas,'Cascadia Mono',monospace;" % CODE_BG


def _inline_html(escaped: str) -> str:
    text = LINK.sub(r"\1 (\2)", escaped)
    text = INLINE_CODE.sub(r'<span style="%s">\1</span>' % INLINE_STYLE, text)
    text = BOLD.sub(r"<b>\1</b>", text)
    text = ITALIC.sub(r"<i>\1</i>", text)
    text = STRIKE.sub(r"<s>\1</s>", text)
    return text


def render_markdown(text: str) -> str:
    if text is None:
        return ""
    source = str(text)
    if not source.strip():
        return ""
    blocks: List[str] = []

    def stash(match: "re.Match[str]") -> str:
        language = (match.group(1) or "").strip()
        code = escape((match.group(2) or "").rstrip("\n"))
        label = (
            '<div style="color:%s; font-size:8pt; font-weight:700; letter-spacing:1px; margin-bottom:4px;">%s</div>'
            % (MUTED, escape(language.upper()))
            if language
            else ""
        )
        blocks.append('<pre style="%s">%s%s</pre>' % (CODE_STYLE, label, code))
        return BLOCK_MARK % (len(blocks) - 1)

    stashed = FENCE_PATTERN.sub(stash, source)
    lines = stashed.split("\n")
    html: List[str] = []
    paragraph: List[str] = []
    bullets: List[str] = []
    numbers: List[str] = []
    quote: List[str] = []
    table: List[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            html.append('<p style="margin:7px 0;">%s</p>' % _inline_html("<br/>".join(paragraph)))
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            items = "".join('<li style="margin:3px 0;">%s</li>' % item for item in bullets)
            html.append('<ul style="margin:6px 0 6px 4px; -qt-list-indent:1;">%s</ul>' % items)
            bullets.clear()

    def flush_numbers() -> None:
        if numbers:
            items = "".join('<li style="margin:3px 0;">%s</li>' % item for item in numbers)
            html.append('<ol style="margin:6px 0 6px 16px;">%s</ol>' % items)
            numbers.clear()

    def flush_quote() -> None:
        if quote:
            body = "<br/>".join(quote)
            html.append(
                '<table width="100%%" cellspacing="0" cellpadding="0" style="margin:6px 0;"><tr>'
                '<td style="border-left:3px solid %s; padding-left:10px; color:%s;">%s</td></tr></table>'
                % (ACCENT, MUTED, body)
            )
            quote.clear()

    def flush_table() -> None:
        if table:
            html.append('<pre style="%s">%s</pre>' % (CODE_STYLE, "\n".join(table)))
            table.clear()

    def flush_all() -> None:
        flush_paragraph()
        flush_bullets()
        flush_numbers()
        flush_quote()
        flush_table()

    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()
        stripped = line.strip()
        if re.fullmatch(r"\u0000CB\d+\u0000", stripped):
            flush_all()
            html.append(stripped)
            index += 1
            continue
        if not stripped:
            flush_all()
            index += 1
            continue
        heading = HEADING.match(stripped)
        if heading:
            flush_all()
            level = len(heading.group(1))
            size = {1: "13pt", 2: "12pt", 3: "11pt"}.get(level, "10.5pt")
            html.append(
                '<p style="margin:12px 0 6px 0; font-size:%s; font-weight:700; color:#ffffff;">%s</p>'
                % (size, _inline_html(escape(heading.group(2))))
            )
            index += 1
            continue
        if RULE.match(stripped):
            flush_all()
            html.append('<hr style="border:none; border-top:1px solid %s; margin:12px 0;"/>' % BORDER)
            index += 1
            continue
        if "|" in stripped and index + 1 < len(lines) and TABLE_SEP.match(lines[index + 1].strip()):
            flush_all()
            table.append(stripped)
            index += 2
            while index < len(lines) and "|" in lines[index]:
                table.append(lines[index].strip())
                index += 1
            flush_table()
            continue
        bullet = BULLET.match(line)
        if bullet:
            flush_paragraph()
            flush_numbers()
            bullets.append(_inline_html(escape(bullet.group(1).strip())))
            index += 1
            continue
        ordered = ORDERED.match(line)
        if ordered:
            flush_paragraph()
            flush_bullets()
            numbers.append(_inline_html(escape(ordered.group(2).strip())))
            index += 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote.append(_inline_html(escape(stripped.lstrip("> ").strip())))
            index += 1
            continue
        flush_bullets()
        flush_numbers()
        flush_quote()
        paragraph.append(_inline_html(escape(stripped)))
        index += 1
    flush_all()
    rendered = "".join(html)
    for position, block in enumerate(blocks):
        rendered = rendered.replace("\u0000CB%d\u0000" % position, block)
    return rendered


class SessionStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else sessions_path()
        self.sessions: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            self.sessions = []
            return self.sessions
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            try:
                self.path.replace(self.path.with_suffix(".corrupt.json"))
            except OSError:
                pass
            self.sessions = []
            return self.sessions
        items = data.get("sessions") if isinstance(data, dict) else data
        self.sessions = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        self.sessions.sort(key=lambda item: item.get("updated", ""), reverse=True)
        return self.sessions

    def save(self) -> None:
        payload = {"version": 1, "sessions": self.sessions[:120]}
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            pass

    def create(self, model: str, workspace: str = "") -> Dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        session = {
            "id": uuid.uuid4().hex[:12],
            "title": "New chat",
            "created": now,
            "updated": now,
            "model": model,
            "workspace": workspace,
            "messages": [],
            "display": [],
        }
        self.sessions.insert(0, session)
        return session

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        for session in self.sessions:
            if session.get("id") == session_id:
                return session
        return None

    def upsert(self, session: Dict[str, Any]) -> None:
        session["updated"] = datetime.now().isoformat(timespec="seconds")
        existing = self.get(session.get("id", ""))
        if existing is None:
            self.sessions.insert(0, session)
            return
        if existing is session:
            self.sessions.sort(key=lambda item: item.get("updated", ""), reverse=True)
            return
        snapshot = dict(session)
        existing.clear()
        existing.update(snapshot)
        self.sessions.sort(key=lambda item: item.get("updated", ""), reverse=True)

    def delete(self, session_id: str) -> None:
        self.sessions = [item for item in self.sessions if item.get("id") != session_id]


def strip_images_for_storage(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stored: List[Dict[str, Any]] = []
    for message in messages:
        clone = dict(message)
        content = clone.get("content")
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and "image_url" in part:
                    parts.append({"type": "text", "text": "[image attached in this chat was not saved to disk]"})
                else:
                    parts.append(part)
            clone["content"] = parts
        stored.append(clone)
    return stored


def conversation_markdown(
    title: str,
    model: str,
    workspace: str,
    display: List[Dict[str, Any]],
    created: str = "",
) -> str:
    lines: List[str] = [
        "# %s" % (title or "Offline AI Studio chat"),
        "",
        "- Model: `%s`" % model,
        "- Project folder: `%s`" % (workspace or "none opened"),
        "- Started: %s" % (created or "unknown"),
        "- Exported: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"),
        "- Source: %s %s, running fully offline" % (APP_NAME, VERSION),
        "",
        "---",
        "",
    ]
    for index, item in enumerate(display, start=1):
        kind = item.get("kind")
        if kind == "user":
            lines.extend(["## %d. You" % index, "", str(item.get("text", "")), ""])
        elif kind == "assistant":
            lines.append("## %d. %s" % (index, APP_NAME))
            lines.append("")
            reasoning = str(item.get("reasoning", "") or "").strip()
            if reasoning:
                lines.append("> Reasoning")
                for chunk in reasoning.splitlines():
                    lines.append("> %s" % chunk)
                lines.append("")
            lines.extend([str(item.get("text", "")), ""])
        elif kind == "tool":
            status = "ok" if item.get("ok") else "failed"
            lines.extend(
                [
                    "### %d. Tool - %s (%s)" % (index, item.get("name", "tool"), status),
                    "",
                    "```text",
                    "arguments: %s" % item.get("arguments", "{}"),
                    str(item.get("result", ""))[:4000],
                    "```",
                    "",
                ]
            )
        elif kind in ("note", "error"):
            lines.extend(["> %s %s" % ("Note:" if kind == "note" else "Error:", item.get("text", "")), ""])
    if len(lines) <= 9:
        lines.append("_This conversation is still empty._")
    return "\n".join(lines).rstrip() + "\n"


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str = "", heading: str = "You", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.role = role
        self.heading = heading
        self.raw = text or ""
        self.streaming = False
        self.setObjectName(
            {
                "user": "BubbleUser",
                "assistant": "BubbleAssistant",
                "note": "BubbleNote",
                "error": "BubbleError",
            }.get(role, "Bubble")
        )
        self.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self.heading_label = QLabel(heading.upper())
        self.heading_label.setObjectName("Caption")
        header.addWidget(self.heading_label)
        header.addStretch(1)
        self.time_label = QLabel(datetime.now().strftime("%H:%M"))
        self.time_label.setObjectName("Caption")
        header.addWidget(self.time_label)
        layout.addLayout(header)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.body)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        footer.addStretch(1)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setProperty("class", "link")
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.copy_button.setVisible(False)
        footer.addWidget(self.copy_button)
        layout.addLayout(footer)
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(110)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._render)
        self.set_markdown(self.raw)

    def copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.raw)

    def set_markdown(self, text: str) -> None:
        self.raw = text or ""
        self._render()

    def append_text(self, piece: str) -> None:
        if not piece:
            return
        self.raw += piece
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def set_streaming(self, streaming: bool) -> None:
        self.streaming = streaming
        if streaming:
            self.copy_button.setVisible(False)
        self._render()

    def finish(self) -> None:
        if self._flush_timer.isActive():
            self._flush_timer.stop()
        self.streaming = False
        self._render()
        self.copy_button.setVisible(bool(self.raw.strip()))

    def _render(self) -> None:
        body = render_markdown(self.raw)
        if self.streaming:
            body += '<span style="color:%s;">|</span>' % ACCENT
        if not body:
            body = '<span style="color:%s;">%s</span>' % (
                MUTED,
                escape("Thinking..." if self.role == "assistant" else ""),
            )
        self.body.setText(body)


class ReasoningPanel(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Reasoning")
        self.raw = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.toggle = QPushButton("Reasoning")
        self.toggle.setProperty("class", "link")
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.clicked.connect(self.toggle_body)
        header.addWidget(self.toggle)
        self.hint = QLabel("")
        self.hint.setObjectName("Caption")
        header.addWidget(self.hint)
        header.addStretch(1)
        layout.addLayout(header)
        self.body = QLabel("")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.body)
        self.setVisible(False)
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(160)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._render)
        self.body_visible = True

    def toggle_body(self) -> None:
        self.body_visible = not self.body_visible
        self.body.setVisible(self.body_visible)
        self.hint.setText("(%d chars)" % len(self.raw))

    def append_text(self, piece: str) -> None:
        if not piece:
            return
        self.raw += piece
        self.setVisible(True)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def finish(self) -> None:
        if self._flush_timer.isActive():
            self._flush_timer.stop()
        self._render()
        self.hint.setText("(%d chars)" % len(self.raw))
        if len(self.raw) > 900:
            self.body_visible = False
            self.body.setVisible(False)

    def _render(self) -> None:
        escaped = escape(self.raw).replace("\n", "<br/>")
        self.body.setText('<span style="color:%s; font-size:9.5pt;">%s</span>' % (MUTED, escaped))


class ToolCard(QFrame):
    def __init__(self, name: str, arguments: str = "{}", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.name = name
        self.arguments = arguments or "{}"
        self.result = ""
        self.ok = False
        self.finished = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self.status_dot = QLabel("...")
        self.status_dot.setObjectName("Caption")
        header.addWidget(self.status_dot)
        self.title = QLabel("Tool \u2022 %s" % name)
        self.title.setObjectName("ToolName")
        header.addWidget(self.title)
        self.args_label = QLabel(arguments or "{}")
        self.args_label.setObjectName("Muted")
        self.args_label.setWordWrap(False)
        header.addWidget(self.args_label, 1)
        self.toggle = QPushButton("Show output")
        self.toggle.setProperty("class", "ghost")
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.clicked.connect(self.toggle_output)
        header.addWidget(self.toggle)
        layout.addLayout(header)
        self.output = QLabel("Running...")
        self.output.setWordWrap(True)
        self.output.setTextFormat(Qt.PlainText)
        self.output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output.setFont(QFont("Consolas", 9))
        self.output.setVisible(False)
        layout.addWidget(self.output)
        self.output_visible = False

    def toggle_output(self) -> None:
        self.output_visible = not self.output_visible
        self.output.setVisible(self.output_visible)
        self.toggle.setText("Hide output" if self.output_visible else "Show output")

    def set_result(self, ok: bool, result: str) -> None:
        self.ok = ok
        self.finished = True
        self.result = result or ""
        self.status_dot.setText("OK" if ok else "FAIL")
        self.status_dot.setObjectName("StatusOk" if ok else "StatusBad")
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)
        text = self.result.strip() or "(no output)"
        if len(text) > 4000:
            text = text[:4000] + "\n... (truncated in the UI)"
        self.output.setText(text)
        if not ok or len(text) < 500:
            self.output_visible = True
            self.output.setVisible(True)
            self.toggle.setText("Hide output")


class EmptyState(QWidget):
    open_project = Signal()
    attach_image = Signal()
    change_model = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(16)
        outer.addStretch(1)
        logo = QLabel()
        logo.setPixmap(icon_pixmap(56))
        logo.setAlignment(Qt.AlignCenter)
        outer.addWidget(logo)
        title = QLabel("Your offline AI workspace")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)
        subtitle = QLabel(
            "Coding help, writing, image analysis and a patient teaching partner - all running on this PC "
            "through LM Studio. No internet, no accounts, no API keys."
        )
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFixedWidth(640)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(subtitle)
        row.addStretch(1)
        outer.addLayout(row)
        grid = QGridLayout()
        grid.setSpacing(10)
        for index, (heading, body) in enumerate(WELCOME_FEATURES):
            card = QFrame()
            card.setObjectName("Card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(4)
            heading_label = QLabel(heading)
            heading_label.setObjectName("Heading")
            card_layout.addWidget(heading_label)
            body_label = QLabel(body)
            body_label.setObjectName("Muted")
            body_label.setWordWrap(True)
            card_layout.addWidget(body_label)
            card_layout.addStretch(1)
            grid.addWidget(card, index // 2, index % 2)
        outer.addLayout(grid)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addStretch(1)
        for label, kind in QUICK_ACTIONS:
            button = QPushButton(label)
            button.setProperty("class", "ghost")
            button.setCursor(Qt.PointingHandCursor)
            if kind == "project":
                button.clicked.connect(self.open_project.emit)
            elif kind == "attach":
                button.clicked.connect(self.attach_image.emit)
            else:
                button.clicked.connect(self.change_model.emit)
            actions.addWidget(button)
        actions.addStretch(1)
        outer.addLayout(actions)
        hint = QLabel("Everything stays on this PC. Export report turns chat into a Markdown file you can keep.")
        hint.setObjectName("Caption")
        hint.setAlignment(Qt.AlignCenter)
        outer.addWidget(hint)
        outer.addStretch(1)


class ModelSelectDialog(QDialog):
    def __init__(self, current: str = DEFAULT_MODEL, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose a local model")
        self.setMinimumWidth(660)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        title = QLabel("Start Offline AI Studio")
        title.setObjectName("Title")
        layout.addWidget(title)
        subtitle = QLabel(
            "Four models are already downloaded in LM Studio. Pick one - the app starts the local server and "
            "loads your choice by itself. Model ids are exact, so the app never picks the wrong file."
        )
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        body = QHBoxLayout()
        body.setSpacing(12)
        self.list = QListWidget()
        self.list.setObjectName("ModelList")
        self.list.setMinimumWidth(300)
        for entry in MODEL_CATALOG:
            item = QListWidgetItem("%s  -  %s" % (entry["name"], entry["tag"]))
            item.setData(Qt.UserRole, entry["id"])
            item.setToolTip(entry["id"])
            self.list.addItem(item)
        body.addWidget(self.list, 3)
        self.details = QLabel("")
        self.details.setObjectName("Muted")
        self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignTop)
        self.details.setMinimumWidth(260)
        body.addWidget(self.details, 2)
        layout.addLayout(body)
        self.id_label = QLabel("")
        self.id_label.setObjectName("Chip")
        layout.addWidget(self.id_label, 0, Qt.AlignLeft)
        custom_row = QVBoxLayout()
        custom_label = QLabel("Or type any other model id you have already downloaded")
        custom_label.setObjectName("Caption")
        custom_row.addWidget(custom_label)
        self.custom = QLineEdit()
        self.custom.setPlaceholderText("for example: lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF")
        self.custom.textChanged.connect(self._on_custom_changed)
        custom_row.addWidget(self.custom)
        layout.addLayout(custom_row)
        note = QLabel(
            "Tip: 8B on CPU is about three tokens per second, the 12B models are roughly twice as slow but "
            "handle writing and images better."
        )
        note.setObjectName("Caption")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton("Load this model", QDialogButtonBox.AcceptRole)
        self.ok_button.setProperty("class", "accent")
        self.cancel_button = buttons.addButton("Quit", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._last_row = 0
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())
        target_row = 0
        for index, entry in enumerate(MODEL_CATALOG):
            if entry["id"] == current:
                target_row = index
                break
        self.list.setCurrentRow(target_row)
        self._on_row_changed(target_row)

    def _on_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.list.item(row)
        if item is None:
            return
        self._last_row = row
        model_id = item.data(Qt.UserRole)
        if self.custom.text().strip():
            self.custom.blockSignals(True)
            self.custom.clear()
            self.custom.blockSignals(False)
        self.id_label.setText(model_id)
        info = model_info(model_id)
        vision = "Vision-capable - you can attach images." if info.get("vision") else "Text only - no image input."
        self.details.setText("%s\n\n%s\n\nRuns locally in LM Studio. %s" % (info["name"], info["description"], vision))

    def _on_custom_changed(self, text: str) -> None:
        if text.strip():
            row = self.list.currentRow()
            if row >= 0:
                self._last_row = row
            self.list.setCurrentRow(-1)
            self.id_label.setText(text.strip())
            self.details.setText(
                "Custom model id. The app loads it with the LM Studio CLI exactly like the built-in ones. If "
                "the id does not match a downloaded model, loading will fail and you will be asked to pick again."
            )
        else:
            self.list.setCurrentRow(self._last_row)

    def selected_model(self) -> str:
        typed = self.custom.text().strip()
        if typed:
            return typed
        item = self.list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")


class LoadingDialog(QDialog):
    def __init__(self, model: str, lms_path: Optional[str] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.lms_path = lms_path
        self.ok = False
        self.message = ""
        self.worker: Optional[Any] = None
        self.setWindowTitle("Preparing %s" % short_model_name(model))
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        title = QLabel("Waking up your offline studio")
        title.setObjectName("Title")
        layout.addWidget(title)
        self.step_label = QLabel("Checking the LM Studio local server...")
        self.step_label.setObjectName("Muted")
        self.step_label.setWordWrap(True)
        layout.addWidget(self.step_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setObjectName("LogView")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(190)
        layout.addWidget(self.log)
        note = QLabel(
            "The app: 1) starts the local server, 2) unloads any other model to free RAM, 3) loads %s, "
            "4) sends a one-token probe until it answers. On CPU this can take a couple of minutes."
            % model
        )
        note.setObjectName("Caption")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        row.addStretch(1)
        self.quit_button = QPushButton("Cancel")
        self.quit_button.setProperty("class", "ghost")
        self.quit_button.clicked.connect(self.reject)
        row.addWidget(self.quit_button)
        layout.addLayout(row)

    def append_step(self, text: str) -> None:
        self.step_label.setText(text)
        self.log.append(text)

    def run_and_wait(self) -> Tuple[bool, str]:
        worker_class = bootstrap.BootstrapWorker
        if worker_class is None:
            ok, message = bootstrap.ensure_ready_with_reason(self.model, on_step=self.append_step, lms_path=self.lms_path)
            self.ok, self.message = ok, message
            return self.ok, self.message
        self.worker = worker_class(self.model, self.lms_path)
        self.worker.step.connect(self.append_step)
        self.worker.done.connect(self._on_worker_done)
        self.worker.start()
        self.exec()
        return self.ok, self.message

    def _on_worker_done(self, ok: bool, message: str) -> None:
        self.ok = ok
        self.message = message
        if ok:
            self.append_step("%s is ready." % short_model_name(self.model))
            QTimer.singleShot(250, self.accept)
        else:
            self.reject()

    def closeEvent(self, event: QEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(1500)
        super().closeEvent(event)


class GenerationWorker(QThread):
    event = Signal(dict)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        client: LLMClient,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        workspace: Workspace,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        max_steps: int = MAX_STEPS,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.messages = messages
        self.tools = tools
        self.workspace = workspace
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_steps = max_steps
        self.runner: Optional[AgentRunner] = None

    def run(self) -> None:
        try:
            self.runner = AgentRunner(
                self.client,
                tools=self.tools,
                workspace=self.workspace,
                system_prompt=self.system_prompt,
                max_steps=self.max_steps,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            result = self.runner.run(self.messages, on_event=self.event.emit)
        except Exception as exc:
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))
            return
        self.done.emit(
            {
                "messages": result.messages,
                "content": result.content,
                "reasoning": result.reasoning,
                "tool_calls": len(result.tool_calls),
                "steps": result.steps,
                "cancelled": result.cancelled,
                "finish_reason": result.finish_reason,
                "hit_step_limit": result.hit_step_limit,
            }
        )

    def cancel(self) -> None:
        try:
            self.client.cancel()
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        lms_path: Optional[str] = None,
        store: Optional[SessionStore] = None,
        settings: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.model = model or DEFAULT_MODEL
        self.lms_path = lms_path
        self.workspace = Workspace()
        self.tools = build_tools(self.workspace)
        self.store = store or SessionStore()
        self.store.load()
        self.settings = dict(settings or {})
        self.messages: List[Dict[str, Any]] = []
        self.display: List[Dict[str, Any]] = []
        self.session_id: str = ""
        self.session_created: str = ""
        self.conversation_title: str = "New chat"
        self.generating = False
        self.worker: Optional[GenerationWorker] = None
        self.stream_bubble: Optional[MessageBubble] = None
        self.stream_reasoning: Optional[ReasoningPanel] = None
        self.stream_assistant_item: Optional[Dict[str, Any]] = None
        self.tool_cards: List[ToolCard] = []
        self.attachment: Optional[Dict[str, str]] = None
        self.tools_enabled = bool(self.settings.get("tools_enabled", True))
        self.bubbles: List[MessageBubble] = []
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1220, 820)
        self.setMinimumSize(940, 620)
        self._build_ui()
        self._restore_workspace(self.settings.get("workspace") or "")
        self._refresh_session_list()
        self.new_chat()
        self._refresh_actions()

    def _build_ui(self) -> None:
        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._build_sidebar())
        row.addWidget(self._build_chat(), 1)
        self.setCentralWidget(central)
        copy_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_shortcut.activated.connect(self._copy_last_reply)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(268)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)
        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(icon_pixmap(34))
        brand.addWidget(logo)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        name = QLabel(APP_NAME)
        name.setStyleSheet("font-weight: 700; font-size: 12pt; color: #ffffff;")
        brand_text.addWidget(name)
        tagline = QLabel("100% local - no cloud, no keys")
        tagline.setObjectName("Subtitle")
        brand_text.addWidget(tagline)
        brand.addLayout(brand_text)
        brand.addStretch(1)
        layout.addLayout(brand)
        layout.addSpacing(4)
        self.new_chat_button = QPushButton("New chat")
        self.new_chat_button.setProperty("class", "accent")
        self.new_chat_button.setCursor(Qt.PointingHandCursor)
        self.new_chat_button.clicked.connect(self.new_chat)
        layout.addWidget(self.new_chat_button)
        self.open_project_button = QPushButton("Open project folder")
        self.open_project_button.setCursor(Qt.PointingHandCursor)
        self.open_project_button.clicked.connect(self.open_project_folder)
        layout.addWidget(self.open_project_button)
        self.export_button = QPushButton("Export report (.md)")
        self.export_button.setCursor(Qt.PointingHandCursor)
        self.export_button.clicked.connect(self.export_report)
        layout.addWidget(self.export_button)
        layout.addSpacing(6)
        workspace_caption = QLabel("PROJECT FOLDER")
        workspace_caption.setObjectName("Caption")
        layout.addWidget(workspace_caption)
        self.workspace_label = QLabel("No folder open")
        self.workspace_label.setObjectName("Muted")
        self.workspace_label.setWordWrap(True)
        self.workspace_label.setToolTip("Tools stay inside this folder.")
        layout.addWidget(self.workspace_label)
        workspace_row = QHBoxLayout()
        workspace_row.setSpacing(6)
        self.clear_workspace_button = QPushButton("Clear")
        self.clear_workspace_button.setProperty("class", "ghost")
        self.clear_workspace_button.setCursor(Qt.PointingHandCursor)
        self.clear_workspace_button.clicked.connect(lambda: self.set_project_folder(""))
        workspace_row.addWidget(self.clear_workspace_button)
        workspace_row.addStretch(1)
        layout.addLayout(workspace_row)
        layout.addSpacing(6)
        conversations_caption = QLabel("SAVED CONVERSATIONS")
        conversations_caption.setObjectName("Caption")
        layout.addWidget(conversations_caption)
        self.session_list = QListWidget()
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._session_menu)
        self.session_list.itemDoubleClicked.connect(self._open_session_item)
        self.session_list.setToolTip("Double-click a conversation to load it. Right-click to rename or delete.")
        layout.addWidget(self.session_list, 1)
        self.delete_session_button = QPushButton("Delete selected conversation")
        self.delete_session_button.setProperty("class", "ghost")
        self.delete_session_button.clicked.connect(self._delete_selected_session)
        layout.addWidget(self.delete_session_button)
        layout.addSpacing(4)
        footer = QFrame()
        footer.setObjectName("Card")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(12, 10, 12, 10)
        footer_layout.setSpacing(4)
        self.model_label = QLabel(short_model_name(self.model))
        self.model_label.setObjectName("ToolName")
        footer_layout.addWidget(self.model_label)
        self.model_id_label = QLabel(self.model)
        self.model_id_label.setObjectName("Caption")
        self.model_id_label.setWordWrap(True)
        footer_layout.addWidget(self.model_id_label)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusOk")
        self.status_label.setWordWrap(True)
        footer_layout.addWidget(self.status_label)
        footer_row = QHBoxLayout()
        self.change_model_button = QPushButton("Change model")
        self.change_model_button.setProperty("class", "ghost")
        self.change_model_button.clicked.connect(self.change_model)
        footer_row.addWidget(self.change_model_button)
        self.help_button = QPushButton("?")
        self.help_button.setProperty("class", "ghost")
        self.help_button.setFixedWidth(34)
        self.help_button.clicked.connect(self.show_help)
        footer_row.addWidget(self.help_button)
        footer_layout.addLayout(footer_row)
        layout.addWidget(footer)
        return sidebar

    def _build_chat(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(18, 10, 18, 10)
        top_layout.setSpacing(10)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        self.title_label = QLabel("New chat")
        self.title_label.setStyleSheet("font-weight: 700; font-size: 11.5pt; color: #ffffff;")
        title_box.addWidget(self.title_label)
        self.subtitle_label = QLabel("Local model - everything you type stays on this PC")
        self.subtitle_label.setObjectName("Subtitle")
        title_box.addWidget(self.subtitle_label)
        top_layout.addLayout(title_box)
        top_layout.addStretch(1)
        self.model_chip = QLabel(short_model_name(self.model))
        self.model_chip.setObjectName("Chip")
        self.model_chip.setToolTip(self.model)
        top_layout.addWidget(self.model_chip)
        self.tools_toggle = QToolButton()
        self.tools_toggle.setText("Tools: on")
        self.tools_toggle.setCheckable(True)
        self.tools_toggle.setChecked(self.tools_enabled)
        self.tools_toggle.setCursor(Qt.PointingHandCursor)
        self.tools_toggle.setToolTip(
            "Let the model read, search and write files inside the open project folder."
        )
        self.tools_toggle.toggled.connect(self._on_tools_toggled)
        top_layout.addWidget(self.tools_toggle)
        layout.addWidget(top_bar)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(22, 18, 22, 18)
        self.chat_layout.setSpacing(12)
        self.empty_state = EmptyState()
        self.empty_state.open_project.connect(self.open_project_folder)
        self.empty_state.attach_image.connect(self.attach_image)
        self.empty_state.change_model.connect(self.change_model)
        self.chat_layout.addWidget(self.empty_state)
        self.chat_layout.addStretch(1)
        self.scroll.setWidget(self.chat_container)
        layout.addWidget(self.scroll, 1)
        composer_wrap = QWidget()
        composer_layout = QVBoxLayout(composer_wrap)
        composer_layout.setContentsMargins(22, 0, 22, 18)
        composer_layout.setSpacing(6)
        self.attach_hint = QLabel("")
        self.attach_hint.setObjectName("StatusWarn")
        self.attach_hint.setVisible(False)
        composer_layout.addWidget(self.attach_hint)
        composer = QFrame()
        composer.setObjectName("Composer")
        inner = QVBoxLayout(composer)
        inner.setContentsMargins(12, 10, 12, 10)
        inner.setSpacing(8)
        self.attach_row = QWidget()
        attach_layout = QHBoxLayout(self.attach_row)
        attach_layout.setContentsMargins(0, 0, 0, 0)
        attach_layout.setSpacing(8)
        self.attach_chip = QFrame()
        self.attach_chip.setObjectName("AttachChip")
        chip_layout = QHBoxLayout(self.attach_chip)
        chip_layout.setContentsMargins(6, 4, 6, 4)
        chip_layout.setSpacing(8)
        self.attach_thumb = QLabel()
        self.attach_thumb.setFixedSize(38, 38)
        chip_layout.addWidget(self.attach_thumb)
        self.attach_name = QLabel("")
        self.attach_name.setObjectName("Muted")
        chip_layout.addWidget(self.attach_name)
        self.attach_clear_button = QPushButton("\u2715")
        self.attach_clear_button.setProperty("class", "ghost")
        self.attach_clear_button.setFixedWidth(30)
        self.attach_clear_button.setCursor(Qt.PointingHandCursor)
        self.attach_clear_button.setToolTip("Remove the attached image")
        self.attach_clear_button.clicked.connect(lambda: self.set_attachment(None))
        chip_layout.addWidget(self.attach_clear_button)
        attach_layout.addWidget(self.attach_chip)
        attach_layout.addStretch(1)
        self.attach_row.setVisible(False)
        inner.addWidget(self.attach_row)
        self.input = QTextEdit()
        self.input.setObjectName("ComposerInput")
        self.input.setPlaceholderText(
            "Ask about code, ask for a summary, attach a screenshot...  (Enter to send, Shift+Enter for a new line)"
        )
        self.input.setAcceptRichText(False)
        self.input.setFixedHeight(78)
        self.input.installEventFilter(self)
        self.input.textChanged.connect(self._sync_status)
        inner.addWidget(self.input)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.attach_button = QPushButton("Attach image")
        self.attach_button.setProperty("class", "ghost")
        self.attach_button.setCursor(Qt.PointingHandCursor)
        self.attach_button.clicked.connect(self.attach_image)
        controls.addWidget(self.attach_button)
        self.clear_chat_button = QPushButton("Clear chat")
        self.clear_chat_button.setProperty("class", "ghost")
        self.clear_chat_button.setCursor(Qt.PointingHandCursor)
        self.clear_chat_button.clicked.connect(self.new_chat)
        controls.addWidget(self.clear_chat_button)
        controls.addStretch(1)
        self.context_label = QLabel("")
        self.context_label.setObjectName("Caption")
        controls.addWidget(self.context_label)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("class", "danger")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setVisible(False)
        controls.addWidget(self.stop_button)
        self.send_button = QPushButton("Send")
        self.send_button.setProperty("class", "accent")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.clicked.connect(self.send_message)
        controls.addWidget(self.send_button)
        inner.addLayout(controls)
        composer_layout.addWidget(composer)
        self.footer_label = QLabel(
            "Running on LM Studio on this machine. Nothing is uploaded anywhere; unplug the network and it still works."
        )
        self.footer_label.setObjectName("Caption")
        self.footer_label.setAlignment(Qt.AlignCenter)
        composer_layout.addWidget(self.footer_label)
        layout.addWidget(composer_wrap)
        return container

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if source is self.input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
                self.send_message()
                return True
        return super().eventFilter(source, event)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._apply_bubble_width()

    def _apply_bubble_width(self) -> None:
        width = max(320, int(self.scroll.viewport().width() * 0.82))
        for bubble in self.bubbles:
            bubble.setMaximumWidth(width)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._apply_bubble_width()

    def _copy_last_reply(self) -> None:
        for bubble in reversed(self.bubbles):
            if bubble.role == "assistant" and bubble.raw.strip():
                bubble.copy_to_clipboard()
                self.set_status("Last reply copied to the clipboard.")
                return

    def _restore_workspace(self, path: str) -> None:
        if not path:
            return
        candidate = Path(path)
        if candidate.is_dir():
            self.workspace.set_root(str(candidate))
            self._refresh_workspace_label()

    def _refresh_workspace_label(self) -> None:
        root = self.workspace.root
        if root is None:
            self.workspace_label.setText("No folder open - file tools are disabled until you open one.")
            return
        self.workspace_label.setText(elide(str(root), 220, self.workspace_label))
        self.workspace_label.setToolTip(str(root))

    def set_project_folder(self, folder: str) -> None:
        if self.generating:
            return
        if folder:
            try:
                resolved = self.workspace.set_root(folder)
            except Exception as exc:
                self.set_status("Could not open that folder: %s" % exc, "bad")
                return
            self.settings["workspace"] = str(resolved)
            self.set_status("Folder open: %s" % resolved.name)
        else:
            self.workspace.set_root(None)
            self.settings["workspace"] = ""
            self.set_status("Project folder cleared - file tools are disabled.")
        save_settings(self.settings)
        self._refresh_workspace_label()

    def open_project_folder(self) -> None:
        if self.generating:
            return
        start = self.settings.get("workspace") or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose the folder the tools may work in", start)
        if folder:
            self.set_project_folder(folder)
            self._append_note(
                "Project folder opened: %s\nFile tools now work only inside this folder." % folder
            )

    def attach_image(self) -> None:
        if self.generating:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Attach an image", str(Path.home()), ATTACH_FILTER)
        if not path:
            return
        self.load_attachment(path)

    def load_attachment(self, path: str) -> bool:
        source = Path(path)
        try:
            raw = source.read_bytes()
        except OSError as exc:
            self.set_status("Could not read that image: %s" % exc, "bad")
            return False
        if len(raw) > 12_000_000:
            self.set_status("That image is too large (over 12 MB). Try a smaller screenshot.", "warn")
            return False
        mime = MIME_BY_SUFFIX.get(source.suffix.lower(), "image/png")
        data_uri = "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))
        self.set_attachment(
            {"name": source.name, "path": str(source), "mime": mime, "data_uri": data_uri, "size": str(len(raw))}
        )
        if not model_supports_vision(self.model):
            self.attach_hint.setText(
                "Image analysis needs a vision model. %s cannot see images - switch to Gemma 4 12B or Gemma 3 12B."
                % short_model_name(self.model)
            )
            self.attach_hint.setVisible(True)
        return True

    def set_attachment(self, attachment: Optional[Dict[str, str]]) -> None:
        self.attachment = attachment
        if not attachment:
            self.attach_row.setVisible(False)
            self.attach_hint.setVisible(False)
            self.attach_thumb.clear()
            self.attach_name.setText("")
            return
        pixmap = QPixmap(attachment.get("path", ""))
        if not pixmap.isNull():
            self.attach_thumb.setPixmap(
                pixmap.scaled(QSize(38, 38), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        else:
            self.attach_thumb.clear()
        self.attach_name.setText(attachment.get("name", "image"))
        self.attach_row.setVisible(True)

    def _on_tools_toggled(self, checked: bool) -> None:
        self.tools_enabled = bool(checked)
        self.tools_toggle.setText("Tools: on" if checked else "Tools: off")
        self.settings["tools_enabled"] = checked
        save_settings(self.settings)
        self.set_status("Project tools are %s." % ("enabled" if checked else "disabled"))

    def change_model(self) -> None:
        if self.generating:
            return
        dialog = ModelSelectDialog(self.model, self)
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = dialog.selected_model()
        if not chosen or chosen == self.model:
            return
        loading = LoadingDialog(chosen, self.lms_path, self)
        ok, message = loading.run_and_wait()
        if not ok:
            QMessageBox.critical(self, "Could not load that model", message)
            return
        self.set_model(chosen)
        self._append_note(
            "Model switched to %s (%s). The conversation history is kept." % (short_model_name(chosen), chosen)
        )

    def set_model(self, model: str) -> None:
        self.model = model
        self.model_label.setText(short_model_name(model))
        self.model_id_label.setText(model)
        self.model_chip.setText(short_model_name(model))
        self.model_chip.setToolTip(model)
        self.settings["model"] = model
        save_settings(self.settings)
        if self.attachment and not model_supports_vision(model):
            self.attach_hint.setText(
                "%s cannot see images. Switch to a Gemma model to analyse the attachment."
                % short_model_name(model)
            )
            self.attach_hint.setVisible(True)
        elif self.attachment:
            self.attach_hint.setVisible(False)
        self.set_status("Model: %s" % model)

    def show_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("About %s" % APP_NAME)
        box.setIcon(QMessageBox.Information)
        box.setText("A fully offline AI workspace running on LM Studio.")
        box.setInformativeText(
            "Four jobs, one window:\n"
            "  - coding help, with tools that read and write files in your project folder\n"
            "  - content writing and information, best with the Gemma 12B models\n"
            "  - image analysis - attach a screenshot to a Gemma model\n"
            "  - a teaching partner that explains, quizzes and works through examples\n\n"
            "Keyboard: Enter sends, Shift+Enter adds a line, Ctrl+Shift+C copies the last reply.\n\n"
            "Nothing leaves this PC. No accounts, no API keys, no internet needed."
        )
        box.exec()

    def set_status(self, text: str, kind: str = "ok") -> None:
        self.status_label.setText(text)
        self.status_label.setObjectName({"ok": "StatusOk", "warn": "StatusWarn", "bad": "StatusBad"}.get(kind, "StatusOk"))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _sync_status(self) -> None:
        blocks = self.input.document().blockCount()
        chars = len(self.input.toPlainText())
        self.context_label.setText(
            "%d line%s - %d chars - 8192-token window, history capped at %d chars"
            % (blocks, "" if blocks == 1 else "s", chars, MAX_CHAT_CHARS)
        )

    def new_chat(self) -> None:
        if self.generating:
            return
        self.messages = []
        self.display = []
        self.session_id = ""
        self.session_created = datetime.now().isoformat(timespec="seconds")
        self.conversation_title = "New chat"
        self.title_label.setText(self.conversation_title)
        self.subtitle_label.setText("Local model - everything you type stays on this PC")
        for bubble in self.bubbles:
            self.chat_layout.removeWidget(bubble)
            bubble.deleteLater()
        self.bubbles = []
        for card in self.tool_cards:
            self.chat_layout.removeWidget(card)
            card.deleteLater()
        self.tool_cards = []
        self.input.clear()
        self.attach_hint.setVisible(False)
        self.set_attachment(None)
        self._refresh_actions()
        self.set_status("Ready - nothing leaves this PC.")

    def _append_bubble(self, role: str, text: str, heading: str) -> MessageBubble:
        bubble = MessageBubble(role, text, heading)
        bubble.setMaximumWidth(max(320, int(self.scroll.viewport().width() * 0.82)))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        if role == "user":
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch(1)
        wrapper = QWidget()
        wrapper.setLayout(row)
        wrapper.setObjectName("BubbleRow")
        bubble.wrapper = wrapper
        position = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertWidget(position, wrapper)
        self.bubbles.append(bubble)
        self._scroll_to_bottom()
        return bubble

    def _remove_bubble(self, bubble: MessageBubble) -> None:
        wrapper = getattr(bubble, "wrapper", None)
        if wrapper is not None:
            self.chat_layout.removeWidget(wrapper)
            wrapper.deleteLater()
        if bubble in self.bubbles:
            self.bubbles.remove(bubble)

    def _append_tool_card(self, name: str, arguments: str) -> ToolCard:
        card = ToolCard(name, arguments)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(card, 1)
        wrapper = QWidget()
        wrapper.setLayout(row)
        card.wrapper = wrapper
        position = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertWidget(position, wrapper)
        self.tool_cards.append(card)
        self._scroll_to_bottom()
        return card

    def _append_note(self, text: str) -> None:
        self.display.append({"kind": "note", "text": text})
        self._append_bubble("note", text, "Note")
        self._refresh_actions()
        self._persist_session()

    def _refresh_empty_state(self) -> None:
        self.empty_state.setVisible(not self.messages)

    def _refresh_actions(self) -> None:
        has_messages = bool(self.messages)
        busy = self.generating
        self.export_button.setEnabled(has_messages and not busy)
        self.new_chat_button.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.clear_workspace_button.setEnabled(not busy)
        self.change_model_button.setEnabled(not busy)
        self.delete_session_button.setEnabled(not busy and self.session_list.count() > 0)
        self.session_list.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.attach_button.setEnabled(not busy)
        self.clear_chat_button.setEnabled(not busy)
        self.tools_toggle.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.send_button.setVisible(not busy)
        self.stop_button.setVisible(busy)
        self._refresh_empty_state()

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        if bar.value() >= bar.maximum() - 140 or not self.generating:
            QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def _make_title(self, text: str) -> str:
        clean = " ".join((text or "").split())
        if len(clean) > TITLE_MAX:
            clean = clean[: TITLE_MAX - 1].rstrip() + "\u2026"
        return clean or "New chat"

    def _build_user_content(self, text: str) -> Any:
        if not self.attachment:
            return text
        return [
            {"type": "text", "text": text or "Describe this image."},
            {"type": "image_url", "image_url": {"url": self.attachment["data_uri"]}},
        ]

    def send_message(self) -> None:
        if self.generating:
            return
        text = self.input.toPlainText().strip()
        if not text and not self.attachment:
            self.set_status("Type something, or attach an image first.", "warn")
            return
        attachment = self.attachment
        user_message = {"role": "user", "content": self._build_user_content(text)}
        self.messages.append(user_message)
        heading = "You"
        self._append_bubble("user", text or "(image attached)", heading)
        self.display.append({"kind": "user", "text": text or "(image only)"})
        if attachment:
            self.display.append({"kind": "note", "text": "Attached image: %s" % attachment.get("name", "image")})
        if self.conversation_title == "New chat":
            self.conversation_title = self._make_title(text or "Image analysis")
            self.title_label.setText(self.conversation_title)
        if attachment:
            self.set_attachment(None)
        self.input.clear()
        self._refresh_actions()
        self._persist_session()
        self._start_generation()

    def _system_prompt(self) -> str:
        return default_system_prompt(self.workspace, bool(self.tools_enabled and self.tools))

    def _start_stream_segment(self) -> None:
        if self.stream_bubble is not None:
            return
        self.stream_bubble = self._append_bubble("assistant", "", short_model_name(self.model))
        self.stream_bubble.set_streaming(True)
        self.stream_reasoning = ReasoningPanel()
        position = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertWidget(position, self.stream_reasoning)
        self.stream_assistant_item = {"kind": "assistant", "text": "", "reasoning": "", "model": self.model}

    def _close_stream_segment(self) -> None:
        if self.stream_bubble is not None:
            self.stream_bubble.finish()
            if self.stream_assistant_item is not None:
                self.stream_assistant_item["text"] = self.stream_bubble.raw
        if self.stream_reasoning is not None:
            self.stream_reasoning.finish()
            if self.stream_assistant_item is not None:
                self.stream_assistant_item["reasoning"] = self.stream_reasoning.raw
        item = self.stream_assistant_item
        if item is not None and (str(item.get("text", "")).strip() or str(item.get("reasoning", "")).strip()):
            self.display.append(item)
        self.stream_bubble = None
        self.stream_reasoning = None
        self.stream_assistant_item = None

    def _drop_empty_segment_placeholder(self) -> None:
        if self.stream_bubble is not None and not self.stream_bubble.raw.strip():
            self._remove_bubble(self.stream_bubble)
            self.stream_bubble = None
        if self.stream_reasoning is not None and not self.stream_reasoning.raw.strip():
            self.chat_layout.removeWidget(self.stream_reasoning)
            self.stream_reasoning.deleteLater()
            self.stream_reasoning = None

    def _start_generation(self) -> None:
        tools = self.tools if self.tools_enabled else []
        prompt = self._system_prompt()
        budget = reply_token_budget([{"role": "system", "content": prompt}] + self.messages, ceiling=DEFAULT_MAX_TOKENS)
        client = LLMClient(self.model, bootstrap.BASE_URL, thinking=False)
        self.tool_cards = []
        self.generating = True
        self._start_stream_segment()
        self._refresh_actions()
        self.set_status("Generating with %s - about 3 tokens per second on this CPU." % short_model_name(self.model))
        payload = [{"role": "system", "content": prompt}] + [dict(item) for item in self.messages]
        self.worker = GenerationWorker(
            client, payload, tools, self.workspace, prompt, budget, 0.7, MAX_STEPS, self
        )
        self.worker.event.connect(self._on_agent_event)
        self.worker.done.connect(self._on_generation_done)
        self.worker.failed.connect(self._on_generation_failed)
        self.worker.start()

    def stop_generation(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.set_status("Stopping...", "warn")

    def _on_agent_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "content":
            self._start_stream_segment()
            if self.stream_bubble is not None:
                self.stream_bubble.append_text(event.get("text", ""))
                if self.stream_assistant_item is not None:
                    self.stream_assistant_item["text"] = self.stream_bubble.raw
            self._scroll_to_bottom()
        elif kind == "reasoning":
            self._start_stream_segment()
            if self.stream_reasoning is not None:
                self.stream_reasoning.append_text(event.get("text", ""))
                if self.stream_assistant_item is not None:
                    self.stream_assistant_item["reasoning"] = self.stream_reasoning.raw
        elif kind == "tool_start":
            self._close_stream_segment()
            card = self._append_tool_card(event.get("name", "tool"), event.get("arguments", "{}"))
            card.setProperty("call_id", event.get("call_id", ""))
            entry = {
                "kind": "tool",
                "name": event.get("name", "tool"),
                "arguments": event.get("arguments", "{}"),
                "result": "",
                "ok": False,
                "call_id": event.get("call_id", ""),
            }
            self.display.append(entry)
            self.set_status("Tool call: %s" % event.get("name", "tool"))
        elif kind == "tool_result":
            for card in reversed(self.tool_cards):
                if card.property("call_id") == event.get("call_id") and not card.finished:
                    card.set_result(bool(event.get("ok")), event.get("result", ""))
                    break
            else:
                card = self._append_tool_card(event.get("name", "tool"), event.get("arguments", "{}"))
                card.set_result(bool(event.get("ok")), event.get("result", ""))
            for entry in reversed(self.display):
                if entry.get("kind") == "tool" and entry.get("call_id") == event.get("call_id"):
                    entry["result"] = event.get("result", "")
                    entry["ok"] = bool(event.get("ok"))
                    break
        elif kind == "notice":
            self._append_bubble("note", event.get("message", ""), "Note")
            self.display.append({"kind": "note", "text": event.get("message", "")})
        elif kind == "error":
            self._append_error(event.get("message", NO_CONTENT_MESSAGE))
        elif kind == "final":
            if self.stream_bubble is not None and event.get("content"):
                self.stream_bubble.set_markdown(event["content"])
                if self.stream_assistant_item is not None:
                    self.stream_assistant_item["text"] = event["content"]
        elif kind == "assistant":
            message = event.get("message") or {}
            if message.get("tool_calls") and not message.get("content"):
                self._drop_empty_segment_placeholder()

    def _append_error(self, message: str) -> None:
        text = message or NO_CONTENT_MESSAGE
        self.display.append({"kind": "error", "text": text})
        self._append_bubble("error", text, "Error")
        self.set_status("Generation failed - see the message in the chat.", "bad")

    def _on_generation_failed(self, message: str) -> None:
        self._append_error(message)
        self._finish_generation()
        self._persist_session()

    def _on_generation_done(self, summary: Dict[str, Any]) -> None:
        for message in summary.get("messages") or []:
            self.messages.append(message)
        self._close_stream_segment()
        cancelled = bool(summary.get("cancelled"))
        steps = int(summary.get("steps") or 0)
        if cancelled:
            self.set_status("Stopped. You can keep typing - the partial answer is kept.", "warn")
        elif summary.get("hit_step_limit"):
            self.set_status("Stopped after %d tool steps. Ask for the next step when ready." % MAX_STEPS, "warn")
        else:
            calls = int(summary.get("tool_calls") or 0)
            detail = ""
            if calls:
                detail = " with %d tool call%s" % (calls, "" if calls == 1 else "s")
            self.set_status("Done in %d step%s%s." % (steps, "" if steps == 1 else "s", detail))
        self._finish_generation()
        self._persist_session()

    def _finish_generation(self) -> None:
        self.generating = False
        self.stream_bubble = None
        self.stream_reasoning = None
        self.stream_assistant_item = None
        self.tool_cards = []
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.wait(2000)
        self._refresh_actions()

    def _persist_session(self) -> None:
        if not self.messages:
            return
        if not self.session_id:
            session = self.store.create(self.model, self.settings.get("workspace", ""))
            self.session_id = session["id"]
        session = self.store.get(self.session_id)
        if session is None:
            session = self.store.create(self.model, self.settings.get("workspace", ""))
            self.session_id = session["id"]
        session["title"] = self.conversation_title
        session["model"] = self.model
        session["workspace"] = self.settings.get("workspace", "")
        session["created"] = self.session_created or session.get("created", "")
        session["messages"] = strip_images_for_storage(self.messages)
        session["display"] = json.loads(json.dumps(self.display, ensure_ascii=False))
        self.store.upsert(session)
        self.store.save()
        self._refresh_session_list()
        self._refresh_actions()

    def _refresh_session_list(self) -> None:
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session in self.store.sessions:
            title = session.get("title") or "New chat"
            item = QListWidgetItem("%s\n%s" % (title, session.get("updated", "")[:16].replace("T", " ")))
            item.setData(Qt.UserRole, session.get("id", ""))
            item.setToolTip("%s\nModel: %s\nFolder: %s" % (title, session.get("model", ""), session.get("workspace") or "none"))
            self.session_list.addItem(item)
            if session.get("id") == self.session_id:
                self.session_list.setCurrentItem(item)
        self.session_list.blockSignals(False)
        self.delete_session_button.setEnabled(self.session_list.count() > 0 and not self.generating)

    def _open_session_item(self, item: QListWidgetItem) -> None:
        self.load_session(str(item.data(Qt.UserRole) or ""))

    def _session_menu(self, position: Any) -> None:
        item = self.session_list.itemAt(position)
        if item is None:
            return
        session_id = str(item.data(Qt.UserRole) or "")
        menu = QMenu(self)
        open_action = menu.addAction("Open conversation")
        rename_action = menu.addAction("Rename...")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(self.session_list.mapToGlobal(position))
        if chosen is open_action:
            self.load_session(session_id)
        elif chosen is rename_action:
            self._rename_session(session_id)
        elif chosen is delete_action:
            self._delete_session(session_id)

    def _rename_session(self, session_id: str) -> None:
        session = self.store.get(session_id)
        if session is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Rename conversation")
        layout = QVBoxLayout(dialog)
        field = QLineEdit(session.get("title", ""))
        layout.addWidget(field)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        session["title"] = field.text().strip() or session.get("title", "New chat")
        self.store.save()
        if session_id == self.session_id:
            self.conversation_title = session["title"]
            self.title_label.setText(self.conversation_title)
        self._refresh_session_list()

    def _delete_session(self, session_id: str) -> None:
        self.store.delete(session_id)
        self.store.save()
        if session_id == self.session_id:
            self.new_chat()
        self._refresh_session_list()
        self.set_status("Conversation deleted.")

    def _delete_selected_session(self) -> None:
        item = self.session_list.currentItem()
        if item is None:
            return
        self._delete_session(str(item.data(Qt.UserRole) or ""))

    def load_session(self, session_id: str) -> None:
        if self.generating:
            return
        session = self.store.get(session_id)
        if session is None:
            return
        self.messages = [dict(message) for message in session.get("messages") or []]
        self.display = [dict(item) for item in session.get("display") or []]
        self.session_id = session_id
        self.session_created = session.get("created", "")
        self.conversation_title = session.get("title") or "New chat"
        self.title_label.setText(self.conversation_title)
        self.model = session.get("model") or self.model
        self.set_model(self.model)
        for bubble in self.bubbles:
            wrapper = getattr(bubble, "wrapper", None)
            if wrapper is not None:
                self.chat_layout.removeWidget(wrapper)
                wrapper.deleteLater()
        self.bubbles = []
        for card in self.tool_cards:
            wrapper = getattr(card, "wrapper", None)
            if wrapper is not None:
                self.chat_layout.removeWidget(wrapper)
                wrapper.deleteLater()
        self.tool_cards = []
        for item in self.display:
            kind = item.get("kind")
            if kind == "user":
                self._append_bubble("user", item.get("text", ""), "You")
            elif kind == "assistant":
                bubble = self._append_bubble("assistant", item.get("text", ""), short_model_name(self.model))
                bubble.finish()
                reasoning = str(item.get("reasoning") or "").strip()
                if reasoning:
                    panel = ReasoningPanel()
                    panel.append_text(reasoning)
                    panel.finish()
                    position = max(0, self.chat_layout.count() - 1)
                    self.chat_layout.insertWidget(position, panel)
            elif kind == "tool":
                card = self._append_tool_card(item.get("name", "tool"), item.get("arguments", "{}"))
                card.setProperty("call_id", item.get("call_id", ""))
                card.set_result(bool(item.get("ok")), item.get("result", ""))
            elif kind == "note":
                self._append_bubble("note", item.get("text", ""), "Note")
            elif kind == "error":
                self._append_bubble("error", item.get("text", ""), "Error")
        self.subtitle_label.setText(
            "Loaded from sessions.json - model %s, %d messages" % (self.model, len(self.messages))
        )
        self._refresh_session_list()
        self._refresh_actions()
        self.set_status("Conversation loaded.")
        self._scroll_to_bottom()

    def export_report(self) -> None:
        if not self.messages:
            self.set_status("Nothing to export yet - send a message first.", "warn")
            return
        default_name = re.sub(r"[^A-Za-z0-9 _-]+", "", self.conversation_title or "report").strip() or "report"
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export this conversation as a Markdown report",
            str(Path(self.settings.get("workspace") or Path.home()) / ("%s.md" % default_name)),
            "Markdown (*.md)",
        )
        if not target:
            return
        path = Path(target)
        if path.suffix.lower() != ".md":
            path = path.with_suffix(".md")
        body = conversation_markdown(
            self.conversation_title,
            self.model,
            self.settings.get("workspace", ""),
            self.display,
            self.session_created,
        )
        try:
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Could not save the report", str(exc))
            return
        self.set_status("Report exported to %s" % path)
        self._append_note("Report exported to %s (%d chars)." % (path, len(body)))

    def closeEvent(self, event: QEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self._persist_session()
        save_settings(self.settings)
        super().closeEvent(event)


def prompt_for_model(initial: str = DEFAULT_MODEL) -> Optional[str]:
    dialog = ModelSelectDialog(initial)
    if dialog.exec() != QDialog.Accepted:
        return None
    return dialog.selected_model() or None


def bootstrap_model(model: str, lms_path: Optional[str] = None) -> Tuple[bool, str]:
    dialog = LoadingDialog(model, lms_path)
    return dialog.run_and_wait()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    apply_theme(app)
    settings = load_settings()
    lms_path = bootstrap.find_lms()
    model = str(settings.get("model") or DEFAULT_MODEL)
    while True:
        if not model:
            model = prompt_for_model(DEFAULT_MODEL)
            if not model:
                return 0
        ok, message = bootstrap_model(model, lms_path)
        if ok:
            break
        QMessageBox.critical(
            None,
            "Could not start the local model",
            "%s\n\nPick another model, or start LM Studio manually and try again." % message,
        )
        model = prompt_for_model(model if model in {entry["id"] for entry in MODEL_CATALOG} else DEFAULT_MODEL)
        if not model:
            return 1
    settings["model"] = model
    save_settings(settings)
    window = MainWindow(model, lms_path=lms_path, settings=settings)
    window.show()
    window.set_status("%s is loaded and answering locally." % short_model_name(model))
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
