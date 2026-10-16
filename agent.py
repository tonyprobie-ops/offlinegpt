import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional

from llm import LLMClient, LLMError
from tools import Tool, ToolError, Workspace, describe_arguments, find_tool, tool_names, tool_specs

MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 12_000
STEP_LIMIT_NOTE = (
    "Stopped after %d tool steps to keep the CPU response time sane. Ask for the next step and I will continue."
)

BASE_SYSTEM_PROMPT = (
    "You are Offline AI Studio, a local assistant running fully offline on the user's Windows machine "
    "through LM Studio. There is no internet access, no cloud service and no telemetry: everything you "
    "produce stays on this PC. The CPU is a Ryzen 5 5600GT with no GPU, so be efficient - short, concrete "
    "answers beat long essays. You can use the provided tools to inspect and edit files inside the project "
    "folder the user opened. Always use relative paths, list a folder before assuming what is inside it, and "
    "never write outside the project folder. When a task needs several files, work step by step, check the "
    "result of each tool call, and finish with a short summary of what changed."
)


def default_system_prompt(workspace: Workspace, has_tools: bool) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    if has_tools and workspace.is_open():
        parts.append(
            "The open project folder is '%s'. All tool paths are relative to it. Use list_dir to see the "
            "root before creating new files." % workspace.root
        )
    elif has_tools:
        parts.append(
            "No project folder is open yet, so file tools will refuse to run. Tell the user to click "
            "'Open project folder' in the sidebar when you need file access."
        )
    else:
        parts.append("Tools are disabled for this chat, so answer from your own knowledge.")
    return "\n\n".join(parts)


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 60] + "\n[... tool result truncated to fit the context window ...]"


class AgentRun:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.content = ""
        self.reasoning = ""
        self.steps = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self.cancelled = False
        self.finish_reason = "stop"
        self.hit_step_limit = False

    def append(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)


class AgentRunner:
    def __init__(
        self,
        client: LLMClient,
        tools: Optional[List[Tool]] = None,
        workspace: Optional[Workspace] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.client = client
        self.tools = list(tools or [])
        self.workspace = workspace
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.executor = executor
        self.system_prompt = system_prompt or default_system_prompt(
            workspace or Workspace(), bool(self.tools)
        )

    def cancel(self) -> None:
        self.client.cancel()

    @property
    def spec(self) -> List[Dict[str, Any]]:
        return tool_specs(self.tools)

    @property
    def names(self) -> List[str]:
        return tool_names(self.tools)

    def _execute(self, tool: Tool, arguments: Dict[str, Any]) -> Any:
        if self.executor is not None:
            return self.executor.submit(tool.run, arguments).result()
        return tool.run(arguments)

    def run(
        self,
        messages: Iterable[Dict[str, Any]],
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AgentRun:
        def emit(event: Dict[str, Any]) -> None:
            if on_event is not None:
                on_event(event)

        run = AgentRun()
        working: List[Dict[str, Any]] = [dict(message) for message in messages]
        if not working or working[0].get("role") != "system" or self.system_prompt not in str(working[0].get("content")):
            working = [{"role": "system", "content": self.system_prompt}] + working
        specs = self.spec
        allowed = self.names
        tokens = int(max_tokens or self.max_tokens)
        heat = float(self.temperature if temperature is None else temperature)

        for step in range(self.max_steps):
            run.steps = step + 1
            try:
                result = self.client.stream_chat(
                    working,
                    tools=specs or None,
                    on_content=lambda piece: emit({"type": "content", "text": piece}),
                    on_reasoning=lambda piece: emit({"type": "reasoning", "text": piece}),
                    max_tokens=tokens,
                    temperature=heat,
                    valid_tool_names=allowed or None,
                )
            except LLMError as exc:
                emit({"type": "error", "message": str(exc)})
                run.finish_reason = "error"
                return run
            if result.get("cancelled"):
                run.cancelled = True
                run.finish_reason = "cancelled"
                if result.get("content"):
                    run.content = result["content"] or run.content
                if result.get("reasoning"):
                    run.reasoning = result["reasoning"] or run.reasoning
                return run
            for rejected in result.get("rejected_tool_calls") or []:
                emit({"type": "notice", "message": "Ignored an unknown tool call: %s" % rejected})
            content = result.get("content") or ""
            reasoning = result.get("reasoning") or ""
            calls = result.get("tool_calls") or []
            if content:
                run.content = content
            if reasoning:
                run.reasoning = reasoning
            run.finish_reason = result.get("finish_reason") or "stop"
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                assistant_message["tool_calls"] = calls
            run.append(assistant_message)
            working.append(assistant_message)
            emit({"type": "assistant", "message": assistant_message})
            if not calls:
                emit(
                    {
                        "type": "final",
                        "content": content,
                        "reasoning": reasoning,
                        "finish_reason": run.finish_reason,
                    }
                )
                return run
            run.tool_calls.extend(calls)
            for index, call in enumerate(calls):
                function = call.get("function") or {}
                name = function.get("name") or "unknown_tool"
                arguments_text, arguments = describe_arguments(function.get("arguments") or "{}")
                emit(
                    {
                        "type": "tool_start",
                        "name": name,
                        "arguments": arguments_text,
                        "call_id": call.get("id") or "call_%d_%d" % (step, index),
                    }
                )
                tool = find_tool(self.tools, name)
                if tool is None:
                    output = "Error: unknown tool '%s'. Available tools: %s." % (name, ", ".join(allowed) or "none")
                    ok = False
                else:
                    try:
                        outcome = self._execute(tool, arguments)
                        ok = bool(getattr(outcome, "ok", False))
                        output = str(getattr(outcome, "text", outcome))
                    except ToolError as exc:
                        ok = False
                        output = "Error: %s" % exc
                    except Exception as exc:
                        ok = False
                        output = "Error: %s: %s" % (type(exc).__name__, exc)
                output = _truncate(output)
                emit(
                    {
                        "type": "tool_result",
                        "name": name,
                        "ok": ok,
                        "result": output,
                        "arguments": arguments_text,
                        "call_id": call.get("id") or "call_%d_%d" % (step, index),
                    }
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "call_%d_%d" % (step, index),
                    "name": name,
                    "content": output,
                }
                run.append(tool_message)
                working.append(tool_message)
        run.hit_step_limit = True
        emit({"type": "notice", "message": STEP_LIMIT_NOTE % self.max_steps})
        emit({"type": "final", "content": run.content, "reasoning": run.reasoning, "finish_reason": "step_limit"})
        return run


def tool_event_summary(event: Dict[str, Any]) -> str:
    if event.get("type") == "tool_start":
        return json.dumps(
            {"tool": event.get("name"), "arguments": event.get("arguments")}, ensure_ascii=False
        )
    return json.dumps(event, ensure_ascii=False, default=str)
