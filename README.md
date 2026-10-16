# Offline AI Studio

A Windows desktop app that gives you one offline workspace for four jobs:

1. **Coding help** - chat about code, or let the model read, search and write files in a folder you pick.
2. **Content writing and information** - reports, documentation, emails, explanations.
3. **Image analysis** - attach a screenshot or photo and have a vision model describe or debug it.
4. **A teaching partner** - lessons, worked examples, quizzes and patient explanations.

Everything runs on your PC through **LM Studio** on CPU. No internet, no accounts, no API keys, and no
telemetry. Once a model is loaded you can unplug the network and keep working.

---

## Quick start

```bat
:: double-click run.bat in this folder, or from a terminal:
run.bat
```

`run.bat` creates `.venv`, installs the two requirements and starts the app. On launch the app shows the
model picker, and after you choose a model it **starts the LM Studio server and loads that model by
itself** - no manual steps in the LM Studio window.

Exactly what happens when the app starts:

1. `lms server start --port 1234` if the local server is not already answering on `http://127.0.0.1:1234`.
2. Any other loaded model is unloaded, so the 12B models have enough RAM.
3. The chosen model is loaded with `lms load <model> --yes`, falling back to the plain form on older CLIs
   if `--yes` is rejected.
4. A one-token chat request is sent repeatedly until the model answers, then the chat window opens.

The whole sequence is shown live in the loading dialog. If it fails, the app tells you why and lets you
pick a different model instead of dumping a stack trace.

---

## The four models

| Model id | Best for | Notes |
| --- | --- | --- |
| `qwen/qwen3-8b` | default, fastest | ~5 GiB RAM, about 3 tokens/second on a Ryzen 5 5600GT. Thinking mode is disabled so it answers immediately. |
| `qwen2.5-coder-7b-instruct` | code | Reads, explains and writes code; pairs well with the file tools. |
| `google/gemma-4-12b` | writing, reports, **vision** | ~7.6 GiB RAM - close other apps first. Best prose of the four. |
| `google/gemma-3-12b` | general chat, **vision** | Same RAM footprint as Gemma 4, slightly lighter on long documents. |

The picker also has a free-text field: type any other model id you have already downloaded in LM Studio
and the app loads it the same way. Models are always addressed by their exact identifier, never by index.

**Image analysis** only works with the two Gemma models. Attach a PNG/JPG/WebP/BMP/GIF with *Attach image*;
if the current model cannot see images the app says so instead of silently ignoring the picture.

---

## Using the app

**Sidebar**

- **New chat** - starts a fresh conversation (the previous one stays in the saved list).
- **Open project folder** - picks the folder the file tools may work in. Nothing outside it is reachable.
- **Export report (.md)** - writes the conversation to Markdown, including reasoning and every tool call
  with its output. Disabled while the conversation is empty and while a reply is being generated.
- **Saved conversations** - every chat is stored in `sessions.json` in the app data folder
  (`%LOCALAPPDATA%\OfflineAIStudio`). Double-click to load, right-click to rename or delete.
- Model footer - shows the loaded model and a short status line, with *Change model* and `?` help.

**Chat area**

- Streaming replies with Markdown rendering: headings, lists, tables, quotes and fenced code blocks.
- A collapsible **Reasoning** panel whenever the model emits `reasoning_content`.
- **Tool • name** cards for each tool call, with the arguments in the header and the output underneath.
- Errors appear as a red card in the chat, written for humans ("No content was returned. The model may
  still be loading - please try again in a moment.") rather than raw JSON.

**Composer**

- Enter sends, Shift+Enter makes a new line, Ctrl+Shift+C copies the last reply.
- **Attach image** shows a chip with a thumbnail and a small ✕ to clear the attachment.
- **Stop** cancels the running stream; a partial answer is kept.
- While generating, the input, the attach button, the sidebar actions and the model switcher are disabled.
- **Tools: on/off** in the top bar disables the file tools for a chat, for example when you want pure
  writing help with no chance of the model touching your files.

---

## The tools the model can call

All of them are sandboxed to the folder you opened. Absolute paths, `..` escapes, Windows drive paths and
symlinks that point outside the folder are refused, and every result is size-limited before it goes back
into the model's context.

| Tool | What it does |
| --- | --- |
| `list_dir` | Lists files and folders with sizes. |
| `read_file` | Reads a text file with line numbers (binary files are refused). |
| `write_file` | Creates or overwrites a text file, creating parent folders as needed. |
| `search_text` | Case-insensitive search across the folder, with an optional `*.py` style filter. |
| `make_directory` | Creates a folder. |
| `file_info` | Size, type and modified time. |
| `delete_file` | Deletes one file, only with `confirm: true`, and never a folder. |

The agent loop runs up to 6 tool steps per message, feeds each result back into the conversation, and
stops with a short note if it hits that limit so one request cannot run for ten minutes on a CPU.

---

## Performance on this machine

The app is tuned for a Ryzen 5 5600GT (6C/12T, no GPU) with 15.3 GB of RAM:

- Context window is pinned to **8192 tokens** (all three presets, plus the request itself).
- Chat history is capped at **24 000 characters** (~6 700 tokens at ~3.6 chars/token). That leaves room
  for the agent system prompt, the tool schemas and the reply; the oldest whole turns are dropped from
  the top when the cap is passed, and an attached image is charged ~4 096 characters of that budget.
- The reply budget is sized per request: `8192 - prompt - tool overhead`, clamped to 256-2048 tokens.
- Tool output sent back to the model is truncated at 12 000 characters.

If everything feels slow: use the 8B model, keep one chat short, switch **Tools: off** when you do not
need files, and avoid the 12B models while other apps are open.

---

## Presets for LM Studio

`presets\*.json` are ready-made LM Studio config presets. Copy them into
`C:\Users\<you>\.lmstudio\config-presets\` and they appear in LM Studio's preset list:

- `teacher.preset.json` - `local/teacher-partner`, temperature 0.4, step-by-step teaching style.
- `writer.preset.json` - `local/content-writer`, temperature 0.75, structured prose.
- `coder.preset.json` - `local/coding-partner`, temperature 0.2, minimal diffs and complexity notes.

All three set `llm.load.contextLength=8192`, `llm.load.llama.cpuThreadPoolSize=6`,
`llm.prediction.reasoning.enableThinking=false` and `llm.prediction.maxPredictedTokens=-1` so they match
what the app requests.

---

## Files in this project

| File | Purpose |
| --- | --- |
| `main.py` | PySide6 UI: model picker, loading dialog, chat window, composer, sessions, export, dark theme. |
| `llm.py` | OpenAI-compatible streaming client: SSE parsing, reasoning, tool-call deltas, `/no_think`, history trimming, friendly errors. |
| `tools.py` | The sandboxed file tools described above. |
| `agent.py` | Multi-step tool-calling loop and the system prompt. |
| `bootstrap.py` | Starts the LM Studio server, unloads other models, loads the chosen model, probes until it answers. |
| `run.bat` | Creates the venv, installs requirements, launches the app. |
| `requirements.txt` | `PySide6`, `requests`. |
| `presets\` | The three LM Studio presets. |
| `scripts\verify_live.py` | Live checks against your real LM Studio install. |
| `tests\test_offline_ai_studio.py` | The full offline test suite, including a fake local server. |

---

## Verifying it on your machine

```bat
:: 1. compile check
.venv\Scripts\python -m py_compile llm.py tools.py agent.py bootstrap.py main.py

:: 2. offline test suite (no LM Studio needed - it spins up a local fake server)
.venv\Scripts\python -m unittest discover -s tests -t .

:: 3. live checks against your LM Studio
.venv\Scripts\python scripts\verify_live.py

:: 4. the full cold start: stop the server, then prove the app brings it back
.venv\Scripts\python scripts\verify_live.py --cold --cancel
```

The suite covers: SSE streaming and finish reasons, `reasoning_content` handling, tool-call delta
assembly, the plain-text JSON fallback for tool calls, unknown-tool rejection, cancellation, HTTP and
connection error messages, history trimming and the character/token budget, every sandbox rule of the
file tools, the agent loop (tool round trip, step limit, error and cancel paths), the bootstrap cold-start
sequence, the Markdown renderer, and the headless Qt checks: the picker defaults to `qwen/qwen3-8b`, the
empty state shows and hides correctly, the attachment chip and its ✕ toggle, export enables only with
messages, new-chat resets the empty state, busy states disable the whole sidebar, and streaming events
render bubbles, reasoning and tool cards.

The suite also drives the real client, agent and window against a fake OpenAI-compatible server on
`127.0.0.1` (started by the tests themselves), so streaming, tool round trips, cancellation and the
bootstrap worker thread are all verified without LM Studio running.

`scripts\verify_live.py` prints `PASS`/`FAIL` per check; the important one is

```
Reply with exactly: READY
```

which must come back as content `READY`, empty reasoning and `finish_reason=stop`.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "The LM Studio CLI (lms.exe) was not found" | Start LM Studio once so the CLI is installed, or set `LM_STUDIO_LMS` to the full path of `lms.exe`. |
| "Could not load ... in LM Studio" | Check the id is downloaded in LM Studio; the picker's free-text field takes alternative ids. |
| "The model did not respond within ... seconds" | The model is too big for free RAM. Close other apps or switch to the 8B model. |
| "Could not reach the local model server" | The server was stopped. Restart the app; it will start it again. |
| Replies are cut off mid-sentence | The reply hit its token budget. Ask for the continuation. |
| Very slow first reply | CPU-only inference warms up slowly; the first tokens can take 30-60 seconds. |

---

## Notes

- No cloud calls anywhere in the code. The only endpoints used are `127.0.0.1:1234` and the local
  `lms.exe` CLI.
- `lms.exe` writes non-UTF-8 bytes on Windows consoles, so every subprocess capture decodes bytes with
  `errors="replace"`; the console codec (cp1252) is never used and would crash.
- Conversations live in `%LOCALAPPDATA%\OfflineAIStudio\sessions.json`. Attached images are shown in the
  chat but replaced with a placeholder when saved to disk, so screenshots do not accumulate in the file.
- Set `OFFLINE_AI_STUDIO_HOME` to move that app-data folder somewhere else.
