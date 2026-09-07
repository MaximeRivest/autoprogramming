"""How the coding agent gets launched against a workspace (or how you attach yourself)."""

from __future__ import annotations

import importlib.resources
import json
import shutil
import string
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .errors import AutoProgrammingError, RunnerError

if TYPE_CHECKING:
    from .harness import AgentHarness

#: Where the canonical optimizer expertise lives, relative to this package.
#: The same file is copied verbatim into every generated workspace (under
#: .agents/skills/ and .claude/skills/), and its body is embedded below in
#: OPTIMIZER_PROMPT — one source of truth, edited in exactly one place.
SKILL_RESOURCE = ("skills", "candidate-optimizer", "SKILL.md")


def _load_skill_body() -> str:
    """Read the packaged candidate-optimizer SKILL.md, frontmatter stripped.

    The frontmatter (name/description) is skill-discovery metadata; only the
    Markdown body is expertise, so only the body goes into the prompt.
    """
    resource = importlib.resources.files(__package__)
    for part in SKILL_RESOURCE:
        resource = resource / part
    text = resource.read_text(encoding="utf-8")
    marker = "\n---\n"
    if not text.startswith("---\n") or marker not in text[4:]:
        raise AutoProgrammingError(
            "The packaged candidate-optimizer SKILL.md is malformed: it must "
            "start with a '---' frontmatter block (Agent Skills spec) so the "
            "prompt builder can strip the metadata and embed the body. "
            "Restore src/autoprogramming/skills/candidate-optimizer/SKILL.md."
        )
    return text[text.index(marker, 4) + len(marker):].lstrip("\n")


#: Dynamic header: the per-run facts build_prompt() substitutes. Everything
#: static — the expertise — comes from the shared skill body appended below.
_PROMPT_HEADER = """\
You are the optimizer agent for the autoprogramming workspace at:
  $workspace

Work from that directory. Everything after this header is the
candidate-optimizer skill, byte-for-byte the same one embedded in the
workspace under .agents/skills/candidate-optimizer/; where it says to read a
per-workspace fact from the workspace, the resolved values here already
answer it.

Program schema (immutable — you satisfy it; you never edit schema.py):
$schema

Data: $data_sizes.
$bootstrap_note
Budget remaining: $budget
Run context: $context

Attach with:

    import autoprogramming as ap
    prg = ap.attach("$workspace")

"""

OPTIMIZER_PROMPT = _PROMPT_HEADER + _load_skill_body()


def build_prompt(harness: "AgentHarness", context: dict) -> str:
    """Fill OPTIMIZER_PROMPT with one workspace's facts (sizes, budget, mode)."""
    ws = harness.workspace
    root = getattr(ws, "root", ws)
    try:
        budget = json.dumps(harness.budget)
    except (AutoProgrammingError, TypeError, ValueError):
        budget = "not recorded yet"
    sizes = "unknown (data/split.json not found)"
    bootstrap = False
    split_json = getattr(ws, "split_json", Path(root) / "data" / "split.json")
    try:
        info = json.loads(Path(split_json).read_text())
        counts = info.get("counts", {})
        sizes = (
            f"{counts.get('train', '?')} train / {counts.get('val', '?')} val / "
            f"{counts.get('test', '?')} test rows"
        )
        bootstrap = bool(info.get("bootstrap", False))
    except (OSError, ValueError):
        pass
    if bootstrap:
        bootstrap_note = (
            "BOOTSTRAP MODE: fewer than 30 examples. Build and compare baseline "
            "candidates only — fine-grained mutation loops are refused because "
            "score differences at this data size are one row of noise, and at "
            "most 5 distinct candidates may be scored on val. Offer to generate "
            "synthetic examples for the user to validate."
        )
    else:
        bootstrap_note = "Full optimization mode."
    return string.Template(OPTIMIZER_PROMPT).substitute(
        workspace=str(root),
        schema=harness.schema.describe(),
        data_sizes=sizes,
        bootstrap_note=bootstrap_note,
        budget=budget,
        context=json.dumps(context, sort_keys=True),
    )


@runtime_checkable
class AgentBackend(Protocol):
    """Anything that can drive one optimization run against a harness."""

    def run(self, harness: "AgentHarness", context: dict) -> None:
        """Drive the loop against ``harness``; return when done."""
        ...


class NoOpBackend:
    """A backend that launches nothing and tells you how to drive ``prg`` yourself."""

    def run(self, harness: "AgentHarness", context: dict) -> None:
        """Print the workspace path, the attach snippet, and the prg API."""
        root = getattr(harness.workspace, "root", harness.workspace)
        print(
            "\n".join(
                [
                    "[autoprogramming] no agent backend ran (NoOpBackend).",
                    f"Workspace ready at: {root}",
                    "The workspace already contains the candidate-optimizer skill",
                    "(.agents/skills/ and .claude/skills/), so any skills-capable",
                    "coding agent cd'd into it picks the expertise up automatically.",
                    "Or attach and optimize manually:",
                    "",
                    "    import autoprogramming as ap",
                    f'    prg = ap.attach("{root}")',
                    "",
                    "    prg.schema                                  # inputs/outputs & docstrings",
                    "    prg.propose_metric(code, examples)          # metric sign-off comes first",
                    "    prg.web_search('latest ...')                # run 2+ current-source queries",
                    "    prg.plan_portfolio(specs)                   # same-session host strategy",
                    "    prg.orchestrate_portfolio('breadth')       # isolated implementation workers",
                    "    prg.portfolio_status()                      # inspect failures/audits/objectives",
                    "    prg.new_candidate(source=...)               # manual/legacy candidate path",
                    '    prg.eval("candidate_0")                     # val: aggregate + CI only',
                    '    prg.eval("candidate_0", split="train", per_instance=True)',
                    '    prg.run("candidate_0", split="train", row=0)  # traces: train rows only',
                    '    prg.compare("candidate_0", "candidate_1")   # improved iff CI excludes 0',
                    "    prg.frontier()                              # Pareto frontier over train",
                    "    prg.tradeoffs()                             # quality/cost frontier on val",
                    "    prg.budget                                  # remaining dollars/calls/minutes",
                    "    prg.finalize()                              # one-time test eval + activate",
                ]
            )
        )


class ClaudeCodeBackend:
    """Launches the Claude Code CLI with OPTIMIZER_PROMPT, cwd'd to the workspace."""

    def __init__(self, command: tuple[str, ...] = ("claude",)):
        self.command = tuple(command)

    def run(self, harness: "AgentHarness", context: dict) -> None:
        """Build the prompt and run ``<command> -p <prompt>`` in the workspace."""
        if shutil.which(self.command[0]) is None:
            raise RunnerError(
                f"Agent binary {self.command[0]!r} was not found on PATH, so the "
                f"optimization agent cannot be launched — optimize() needs a "
                f"coding agent to drive the loop. Install the Claude Code CLI, "
                f"or pass backend=NoOpBackend() and drive the loop yourself via "
                f"ap.attach(<workspace>)."
            )
        prompt = build_prompt(harness, context)
        root = getattr(harness.workspace, "root", harness.workspace)
        subprocess.run([*self.command, "-p", prompt], cwd=str(root), check=False)


def default_backend() -> AgentBackend:
    """ClaudeCodeBackend when the ``claude`` CLI is on PATH, else NoOpBackend."""
    if shutil.which("claude"):
        return ClaudeCodeBackend()
    return NoOpBackend()
