"""Low-level Pi JSON/RPC process integration.

This module owns protocol framing, event collection, provider-failure detection,
and usage accounting. Portfolio policy and worker filesystem concerns live in
separate modules.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import RunnerError

ORCHESTRATOR_SYSTEM = """You are a strategy orchestrator, not an implementer.
You manage a breadth-first portfolio of independent Python implementation avenues.
Never write implementation code. Never collapse the search onto an acceptable
local idea while a materially different feasible family remains unexplored.
Return only the requested JSON object, with no Markdown fences or commentary.
"""


@dataclass
class PiUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    dollars: float = 0.0
    turns: int = 0

    def add_message(self, message: dict) -> None:
        if message.get("role") != "assistant":
            return
        usage = message.get("usage") or {}
        self.turns += 1
        self.input_tokens += int(usage.get("input") or 0)
        self.output_tokens += int(usage.get("output") or 0)
        self.cache_read_tokens += int(usage.get("cacheRead") or 0)
        self.cache_write_tokens += int(usage.get("cacheWrite") or 0)
        cost = usage.get("cost") or {}
        self.dollars += float(cost.get("total") or 0.0)


@dataclass
class PiResult:
    text: str
    usage: PiUsage = field(default_factory=PiUsage)
    messages: list[dict] = field(default_factory=list)
    stderr: str = ""
    returncode: int = 0


def assistant_failure(messages: list[dict]) -> str | None:
    """Return the terminal provider failure Pi reports with exit code zero."""
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if not assistants:
        return "Pi completed without an assistant response."
    final = assistants[-1]
    stop = str(final.get("stopReason") or "")
    error = str(final.get("errorMessage") or "").strip()
    if error or stop in ("error", "aborted"):
        detail = error or f"assistant stopped with reason {stop!r}"
        return f"Pi model call failed: {detail}"
    return None


class PiRpcClient:
    """Small synchronous client for Pi's LF-delimited RPC protocol."""

    def __init__(
        self,
        command: tuple[str, ...] = ("pi",),
        *,
        cwd: str | Path | None = None,
        model: str | None = None,
        system_prompt: str = ORCHESTRATOR_SYSTEM,
        timeout: float = 900.0,
        extensions: tuple[str | Path, ...] = (),
        tools: tuple[str, ...] = (),
    ):
        if shutil.which(command[0]) is None and not Path(command[0]).exists():
            raise RunnerError(
                f"Pi executable {command[0]!r} was not found. Install pi or pass "
                "an explicit PiOrchestratorBackend(command=(...))."
            )
        args = [
            *command,
            "--mode", "rpc", "--no-session", "--no-builtin-tools",
            "--no-extensions", "--no-skills", "--no-prompt-templates",
            "--no-themes", "--no-context-files", "--no-approve",
            "--system-prompt", system_prompt,
        ]
        for extension in extensions:
            args.extend(("--extension", str(extension)))
        if tools:
            args.extend(("--tools", ",".join(tools)))
        if model:
            args.extend(("--model", model))
        self.timeout = timeout
        self._proc = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._events: queue.Queue[dict | None] = queue.Queue()
        self._stderr: list[str] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._err_reader.start()

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw[:-1] if raw.endswith("\n") else raw
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                continue
            try:
                self._events.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._events.put(None)

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for text in self._proc.stderr:
            self._stderr.append(text)

    def prompt(self, message: str) -> PiResult:
        if self._proc.poll() is not None:
            raise RunnerError(
                f"Pi RPC process exited before prompting (code {self._proc.returncode}): "
                f"{''.join(self._stderr)[-2000:]}"
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(
            json.dumps({"id": "prompt", "type": "prompt", "message": message}) + "\n"
        )
        self._proc.stdin.flush()

        deadline = time.monotonic() + self.timeout
        messages: list[dict] = []
        usage = PiUsage()
        final_text = ""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.abort()
                raise RunnerError(f"Pi orchestrator timed out after {self.timeout:g}s.")
            try:
                event = self._events.get(timeout=remaining)
            except queue.Empty:
                self.abort()
                raise RunnerError(
                    f"Pi orchestrator timed out after {self.timeout:g}s."
                ) from None
            if event is None:
                raise RunnerError(
                    f"Pi RPC process exited during a run (code {self._proc.poll()}): "
                    f"{''.join(self._stderr)[-2000:]}"
                )
            if (
                event.get("type") == "response"
                and event.get("command") == "prompt"
                and event.get("success") is False
            ):
                raise RunnerError(
                    f"Pi rejected the orchestrator prompt: "
                    f"{event.get('error') or 'unknown RPC error'}"
                )
            if event.get("type") == "message_end" and isinstance(
                event.get("message"), dict
            ):
                msg = event["message"]
                messages.append(msg)
                usage.add_message(msg)
                if msg.get("role") == "assistant":
                    texts = [
                        part.get("text", "")
                        for part in msg.get("content", [])
                        if part.get("type") == "text"
                    ]
                    if texts:
                        final_text = "".join(texts)
            if event.get("type") == "agent_settled":
                break
        failure = assistant_failure(messages)
        if failure:
            raise RunnerError(f"{failure}\n{''.join(self._stderr)[-2000:]}")
        return PiResult(
            text=final_text,
            usage=usage,
            messages=messages,
            stderr="".join(self._stderr),
            returncode=self._proc.poll() or 0,
        )

    def abort(self) -> None:
        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(json.dumps({"type": "abort"}) + "\n")
                self._proc.stdin.flush()
            except OSError:
                pass
            self._proc.terminate()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def json_object(text: str) -> dict:
    """Extract the JSON object requested from a Pi text response."""
    text = text.strip()
    candidates = [text]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        candidates.insert(0, match.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RunnerError(
        f"Pi did not return the required JSON object. Last output:\n{text[-2000:]}"
    )
