"""Workspace scaffolding — the optimization directory that IS the shipped package.

A workspace (``translate_ap/``) is simultaneously the coding agent's working
directory and a valid, installable Python package with zero runtime dependence
on the optimizer. This module creates the workspace, loads it back, and flips
its active candidate; the generated ``__init__.py`` template here is the
product users ship.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.resources
import importlib.util
import json
import keyword
import os
import re
import secrets
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template

from .errors import CandidateError, WorkspaceError
from .schema import Schema

TEST_CSV_PERMS = 0o400

#: Skill-discovery roots written into every workspace. Together they cover
#: Claude Code, Codex, Copilot, Cursor, Gemini, OpenCode, Amp, and Windsurf.
SKILL_DISCOVERY_ROOTS = (".agents", ".claude")


def _optimizer_skill_bytes() -> bytes:
    """The packaged candidate-optimizer SKILL.md, byte-exact.

    The same file backs OPTIMIZER_PROMPT (see backend.py); workspaces get
    verbatim copies so a coding agent cd'd into one picks the expertise up
    through normal skill discovery.
    """
    return (
        importlib.resources.files("autoprogramming")
        / "skills" / "candidate-optimizer" / "SKILL.md"
    ).read_bytes()

_PEP723_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    head = re.split(r"[\s\[\](<>=!~;@,]", requirement.strip(), maxsplit=1)[0]
    return _normalize_dist(head)


@dataclass(frozen=True)
class _MinimalCandidate:
    """Just enough of a candidate to regenerate pyproject.toml.

    Used only while ``candidates.py`` is unavailable; otherwise
    :func:`autoprogramming.candidates.load_candidate` supplies the real thing.
    """

    name: str
    path: Path
    requires_python: str | None
    dependencies: tuple[str, ...]


_PATHS_TEMPLATE = Template(
    '"""Filesystem anchors for $package — candidates use these instead of guessing."""\n'
    "\n"
    "from pathlib import Path\n"
    "\n"
    'artifacts = Path(__file__).parent / "artifacts"\n'
)


_INIT_TEMPLATE = Template('''"""${program} — optimized by autoprogramming."""

from __future__ import annotations

import importlib.util as _ap_importlib
import json as _ap_json
from datetime import datetime as _ap_datetime, timezone as _ap_timezone
from pathlib import Path as _ap_Path

${schema_imports}

_ROOT = _ap_Path(__file__).resolve().parent

_INPUTS = ${input_spec}
_OUTPUTS = ${output_spec}


def _lift(value, tp, base):
    """Lift a raw candidate value into the schema type (via its builtin base)."""
    if isinstance(value, tp):
        return value
    if tp is not base and not isinstance(value, base):
        value = base(value)
    return tp(value)


class _ActiveCandidateProgram:
    """Callable facade over the active candidate.

    Reads active.json and loads candidates/<active>.py lazily on the first
    call, so importing this package never needs an API key or the network.
    """

    def __init__(self):
        self._predict = None
        self._candidate = None
        self._log_enabled = False

    def enable_logging(self):
        """Append every call to logs/<UTC-date>.jsonl; returns self."""
        self._log_enabled = True
        return self

    def disable_logging(self):
        """Stop appending calls to the logs; returns self."""
        self._log_enabled = False
        return self

    def _load(self):
        info = _ap_json.loads((_ROOT / "active.json").read_text())
        name = info.get("active")
        if not name:
            raise RuntimeError(
                "${program} has no active candidate yet: active.json says no "
                "implementation has been activated, so there is nothing to run. "
                "Finish an optimization run (finalize() activates the winner), "
                'or point "active" in active.json at a candidate name.'
            )
        path = _ROOT / "candidates" / (name + ".py")
        if not path.exists():
            raise RuntimeError(
                "${program}'s active candidate " + repr(name) + " is missing "
                "(" + str(path) + " does not exist). active.json must point at "
                "a file in candidates/; restore it or activate another candidate."
            )
        spec = _ap_importlib.spec_from_file_location(
            "${package}._active_" + name, path
        )
        module = _ap_importlib.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._predict = module.predict
        self._candidate = name

    def _bind(self, args, kwargs):
        names = [n for n, _base in _INPUTS]
        if len(args) > len(names):
            raise TypeError(
                "${program}() takes " + str(len(names)) + " argument(s) ("
                + ", ".join(names) + ") but " + str(len(args))
                + " positional argument(s) were given"
            )
        bound = dict(zip(names, args))
        for key, value in kwargs.items():
            if key not in names:
                raise TypeError(
                    "${program}() got an unexpected keyword argument " + repr(key)
                )
            if key in bound:
                raise TypeError(
                    "${program}() got multiple values for argument " + repr(key)
                )
            bound[key] = value
        missing = [n for n in names if n not in bound]
        if missing:
            raise TypeError(
                "${program}() missing required argument(s): " + ", ".join(missing)
            )
        return bound

    def _outputs_dict(self, raw):
        names = [n for n, _tp, _base in _OUTPUTS]
        if isinstance(raw, dict):
            missing = [n for n in names if n not in raw]
            if missing:
                raise TypeError(
                    "${program}'s candidate returned a dict missing output(s): "
                    + ", ".join(missing)
                )
            return {n: raw[n] for n in names}
        if len(names) == 1:
            return {names[0]: raw}
        if not isinstance(raw, (tuple, list)) or len(raw) != len(names):
            raise TypeError(
                "${program}'s candidate must return a tuple of "
                + str(len(names)) + " outputs (" + ", ".join(names)
                + "), got " + repr(raw)
            )
        return dict(zip(names, raw))

    def _log(self, bound, outputs):
        now = _ap_datetime.now(_ap_timezone.utc)
        entry = {
            "inputs": {n: base(bound[n]) for n, base in _INPUTS},
            "outputs": {n: base(outputs[n]) for n, _tp, base in _OUTPUTS},
            "candidate": self._candidate,
            "n_repeat": 1,
            "timestamp": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        log_dir = _ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        day_file = log_dir / (now.strftime("%Y-%m-%d") + ".jsonl")
        with day_file.open("a", encoding="utf-8") as fh:
            fh.write(_ap_json.dumps(entry, default=str) + "\\n")

    def __call__(self, *args, **kwargs):
        bound = self._bind(args, kwargs)
        if self._predict is None:
            self._load()
        outputs = self._outputs_dict(self._predict(**bound))
        typed = tuple(_lift(outputs[n], tp, base) for n, tp, base in _OUTPUTS)
        if self._log_enabled:
            self._log(bound, outputs)
        return typed[0] if len(typed) == 1 else typed


${program} = _ActiveCandidateProgram()

__all__ = ${all_names}
''')


def _render_paths(package_name: str) -> str:
    return _PATHS_TEMPLATE.substitute(package=package_name)


def _render_init(schema: Schema, package_name: str) -> str:
    """Render the generated ``__init__.py`` for a schema.

    The template's own imports are aliased (``_ap_json``, ``_ap_Path``, ...)
    so a custom output type that happens to be named ``Path`` or ``json``
    can be re-exported from ``.schema`` without shadowing the machinery.
    ``__all__`` re-exports only the custom output types: a builtin output
    (``-> str``) has no name defined in the module, and listing it would
    break ``from <pkg> import *``.
    """
    custom = list(dict.fromkeys(
        f.type_name for f in schema.outputs if f.type is not f.base
    ))
    schema_imports = (
        f"from .schema import {', '.join(custom)}" if custom else ""
    )
    input_spec = "[" + ", ".join(
        f'("{f.name}", {f.base.__name__})' for f in schema.inputs
    ) + "]"
    output_spec = "[" + ", ".join(
        f'("{f.name}", {f.type_name}, {f.base.__name__})' for f in schema.outputs
    ) + "]"
    all_names = repr([schema.name, *custom])
    return _INIT_TEMPLATE.substitute(
        program=schema.name,
        package=package_name,
        schema_imports=schema_imports,
        input_spec=input_spec,
        output_spec=output_spec,
        all_names=all_names,
    )


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


class Workspace:
    """Handle on a workspace directory (``<name>_ap/``).

    The directory doubles as the shipped Python package: its name must be a
    valid Python identifier, its ``__init__.py`` runs the active candidate
    with no dependence on autoprogramming, and its ``pyproject.toml`` always
    mirrors the active candidate's runtime dependencies.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        name = self.root.name
        if not name.isidentifier() or keyword.iskeyword(name):
            base = re.sub(r"\W+", "_", name).strip("_") or "translate"
            if not base.isidentifier() or keyword.iskeyword(base):
                base = f"pkg_{base}"
            suggestion = base if base.endswith("_ap") else f"{base}_ap"
            raise WorkspaceError(
                f"Refusing to use {str(self.root)!r} as a workspace: {name!r} is "
                f"not a valid Python identifier. The workspace directory IS the "
                f"shipped package — you import it (`from {suggestion} import "
                f"...`) — so it must be importable. Name it with letters, digits "
                f"and underscores, e.g. {suggestion!r}."
            )
        self._schema: Schema | None = None

    # ----------------------------------------------------------------- paths

    @property
    def data_dir(self) -> Path:
        """``data/`` — public development data plus split metadata."""
        return self.root / "data"

    @property
    def private_data_dir(self) -> Path | None:
        """Controller-only val/test storage for resource-confirmed runs.

        It deliberately lives outside the coding-agent workspace. OS-level
        worker sandboxing remains the strong boundary; this also prevents
        accidental leakage through ordinary workspace inspection.
        """
        try:
            private_id = self.active.get("private_data_id")
        except (OSError, ValueError, KeyError):
            private_id = None
        if not private_id:
            return None
        base = Path(os.environ.get(
            "AP_PRIVATE_DATA_DIR",
            Path.home() / ".cache" / "autoprogramming" / "private",
        ))
        return base / str(private_id)

    def split_path(self, split: str) -> Path:
        """Trusted location of a split; val/test may be outside the workspace."""
        if split in ("val", "test") and self.private_data_dir is not None:
            return self.private_data_dir / f"{split}.csv"
        return self.data_dir / f"{split}.csv"

    @property
    def candidates_dir(self) -> Path:
        """``candidates/`` — one PEP 723 single-file script per candidate."""
        return self.root / "candidates"

    @property
    def artifacts_dir(self) -> Path:
        """``artifacts/`` — model weights, pickles, lookup tables."""
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        """``logs/`` — production-traffic JSONL files (created lazily)."""
        return self.root / "logs"

    @property
    def tmp_dir(self) -> Path:
        """``.ap/`` — scratch space for runner drivers; created on access."""
        p = self.root / ".ap"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def outputs_dir(self) -> Path:
        """``.ap/outputs/`` — cached candidate run outputs (created on write).

        Reuses the ``.ap/`` scratch tree; the outputs cache lets a metric edit
        re-score cached predictions instead of re-running candidates. Pure path
        (no mkdir) so accessing it never materializes ``.ap/`` prematurely.
        """
        return self.root / ".ap" / "outputs"

    @property
    def schema_py(self) -> Path:
        """Generated ``schema.py`` — the immutable program definition."""
        return self.root / "schema.py"

    @property
    def metric_py(self) -> Path:
        """``metric.py`` — the user-approved evaluation metric."""
        return self.root / "metric.py"

    @property
    def metric_approval(self) -> Path:
        """``metric_approval.json`` — who signed the metric off, and when."""
        return self.root / "metric_approval.json"

    @property
    def resources_json(self) -> Path:
        """Confirmed search, runtime, and data-policy capabilities."""
        return self.root / "resources.json"

    @property
    def research_json(self) -> Path:
        """Timestamped web queries and citations used to plan the portfolio."""
        return self.root / ".ap" / "controller" / "research.json"

    @property
    def portfolio_json(self) -> Path:
        """Controller-owned portfolio state (development-time only)."""
        return self.root / ".ap" / "controller" / "portfolio.json"

    @property
    def active_json(self) -> Path:
        """``active.json`` — which candidate is live plus pinned scores."""
        return self.root / "active.json"

    @property
    def scores_json(self) -> Path:
        """``scores.json`` — per-candidate, per-row scores and flags."""
        return self.root / "scores.json"

    @property
    def budget_json(self) -> Path:
        """``budget.json`` — BudgetLedger state."""
        return self.root / "budget.json"

    @property
    def split_json(self) -> Path:
        """``data/split.json`` — seed, ratios, counts, data sha, bootstrap."""
        return self.data_dir / "split.json"

    @property
    def pyproject(self) -> Path:
        """``pyproject.toml`` — generated; rewritten on every activation."""
        return self.root / "pyproject.toml"

    @property
    def init_py(self) -> Path:
        """Generated ``__init__.py`` — the shipped entry point."""
        return self.root / "__init__.py"

    @property
    def paths_py(self) -> Path:
        """Generated ``paths.py`` — filesystem anchors for candidates."""
        return self.root / "paths.py"

    @property
    def final_report(self) -> Path:
        """``final_report.json`` — written once by finalize()."""
        return self.root / "final_report.json"

    # ----------------------------------------------------------------- names

    @property
    def package_name(self) -> str:
        """The importable package name — the directory name itself."""
        return self.root.name

    @property
    def dist_name(self) -> str:
        """The distribution name (package name with underscores as dashes)."""
        return self.package_name.replace("_", "-")

    @property
    def program_name(self) -> str:
        """The program this workspace optimizes, per active.json."""
        return self.active["program"]

    # ----------------------------------------------------------------- state

    @property
    def active(self) -> dict:
        """The parsed contents of active.json."""
        return json.loads(self.active_json.read_text())

    def save_active(self, d: dict) -> None:
        """Persist a full active.json dict."""
        self.active_json.write_text(json.dumps(d, indent=2) + "\n")

    @property
    def schema(self) -> Schema:
        """The pinned program schema, imported from generated ``schema.py``."""
        expected_sha = self.active.get("schema_sha")
        if expected_sha:
            actual_sha = hashlib.sha256(self.schema_py.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise WorkspaceError(
                    f"schema.py in {self.root} changed after workspace creation. "
                    "The implementation search satisfies an immutable contract; "
                    "restore schema.py or start a new workspace for a new schema."
                )
        if self._schema is None:
            module_name = f"_ap_schema_{self.package_name}"
            spec = importlib.util.spec_from_file_location(module_name, self.schema_py)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            program_name = self.program_name
            obj = getattr(module, program_name, None)
            if obj is None:
                raise WorkspaceError(
                    f"schema.py in {self.root} does not define {program_name!r}, "
                    f"but active.json names that program. The generated schema.py "
                    f"is immutable during optimization — restore it (or recreate "
                    f"the workspace) instead of editing it."
                )
            self._schema = Schema.from_object(obj)
        return self._schema

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def create(
        cls,
        root: str | Path,
        schema: Schema,
        splits: dict[str, list[dict]],
        seed: int,
        ratios,
        data_sha: str,
        bootstrap: bool,
        secure_data: bool = False,
    ) -> "Workspace":
        """Scaffold a complete workspace: package files, data splits, state files.

        Refuses to write into an existing non-empty directory; a workspace is
        generated wholesale and must not clobber unrelated files.
        """
        ws = cls(root)
        if ws.root.exists():
            if not ws.root.is_dir():
                raise WorkspaceError(
                    f"Refusing to create a workspace at {ws.root}: a file (not a "
                    f"directory) is already there. Workspaces are directories; "
                    f"move the file or pick another path."
                )
            if any(ws.root.iterdir()):
                raise WorkspaceError(
                    f"Refusing to create a workspace in {ws.root}: the directory "
                    f"is not empty. A workspace is generated wholesale, so writing "
                    f"into a directory with existing files could clobber them. Use "
                    f"a fresh directory — or, if this already is an autoprogramming "
                    f"workspace, open it with Workspace.load({str(ws.root)!r}) "
                    f"(optimize() does that automatically when the data matches)."
                )
        missing = [s for s in ("train", "val", "test") if s not in splits]
        if missing:
            raise WorkspaceError(
                f"Cannot create a workspace without all three splits; missing "
                f"{missing}. The data is split once — train for reflection, val "
                f"for selection, test for the final report — so all three must "
                f"exist from the start. Use data.split_rows(...) to produce them."
            )

        ws.root.mkdir(parents=True, exist_ok=True)
        ws.data_dir.mkdir()
        ws.candidates_dir.mkdir()
        ws.artifacts_dir.mkdir()
        (ws.artifacts_dir / ".gitkeep").write_text("")

        ws.schema_py.write_text(schema.render_module())
        schema_sha = hashlib.sha256(ws.schema_py.read_bytes()).hexdigest()
        ws.paths_py.write_text(_render_paths(ws.package_name))
        ws.init_py.write_text(_render_init(schema, ws.package_name))
        ws.pyproject.write_text(
            ws._render_pyproject(">=3.11", (), schema.doc)
        )
        private_data_id = secrets.token_hex(16) if secure_data else None
        ws.save_active({
            "program": schema.name,
            "active": None,
            "test_score": None,
            "activated_at": None,
            "metric_sha": None,
            "schema_sha": schema_sha,
            "finalized": False,
            "created_at": _utcnow_iso(),
            "private_data_id": private_data_id,
        })

        columns = schema.expected_columns
        _write_csv(ws.data_dir / "train.csv", splits["train"], columns)
        if secure_data:
            private = ws.private_data_dir
            private.mkdir(parents=True, mode=0o700)
            os.chmod(private, 0o700)
            for split in ("val", "test"):
                _write_csv(private / f"{split}.csv", splits[split], columns)
                os.chmod(private / f"{split}.csv", 0o600)
        else:
            for split in ("val", "test"):
                _write_csv(ws.data_dir / f"{split}.csv", splits[split], columns)
            os.chmod(ws.data_dir / "test.csv", TEST_CSV_PERMS)

        ws.split_json.write_text(json.dumps({
            "seed": seed,
            "ratios": list(ratios),
            "counts": {s: len(splits[s]) for s in ("train", "val", "test")},
            "data_sha": data_sha,
            "bootstrap": bool(bootstrap),
        }, indent=2) + "\n")
        ws.scores_json.write_text(json.dumps({
            "metric_sha": None,
            "candidates": {},
            "val_scored": [],
            "flags": {},
        }, indent=2) + "\n")

        # Embed the candidate-optimizer skill so any skills-capable coding
        # agent working in the workspace discovers the optimizer expertise.
        # Dev-time only: deliberately NOT listed in the generated pyproject's
        # package-data — skills are not shipped runtime files.
        skill = _optimizer_skill_bytes()
        for discovery_root in SKILL_DISCOVERY_ROOTS:
            skill_dir = ws.root / discovery_root / "skills" / "candidate-optimizer"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_bytes(skill)

        ws._schema = schema
        return ws

    @classmethod
    def load(cls, root: str | Path) -> "Workspace":
        """Open an existing workspace, verifying its load-bearing files."""
        path = Path(root).expanduser().resolve()
        if not path.exists():
            raise WorkspaceError(
                f"No workspace at {path}: the directory does not exist. "
                f"optimize() creates one, or scaffold it with Workspace.create(...)."
            )
        ws = cls(path)
        if not ws.active_json.exists():
            raise WorkspaceError(
                f"{path} exists but has no active.json, so it is not an "
                f"autoprogramming workspace. Workspaces carry their state in "
                f"active.json; pick the right directory or create a fresh "
                f"workspace with optimize()."
            )
        if not ws.schema_py.exists():
            raise WorkspaceError(
                f"{path} has active.json but no schema.py; the workspace is "
                f"corrupt. schema.py is the immutable program definition — "
                f"restore it from version control or recreate the workspace."
            )
        if ws.private_data_dir is not None:
            missing_private = [
                split for split in ("val", "test")
                if not (ws.private_data_dir / f"{split}.csv").exists()
            ]
            if missing_private:
                raise WorkspaceError(
                    f"{path} references controller-private data but is missing "
                    f"{missing_private}. Restore the private run data under "
                    "$AP_PRIVATE_DATA_DIR (or ~/.cache/autoprogramming/private); "
                    "re-splitting would invalidate the run."
                )
        return ws

    def secure_splits(self) -> None:
        """Move legacy val/test CSVs out of the coding-agent workspace."""
        if self.private_data_dir is not None:
            return
        sources = {split: self.data_dir / f"{split}.csv" for split in ("val", "test")}
        missing = [str(path) for path in sources.values() if not path.exists()]
        if missing:
            raise WorkspaceError(
                f"Cannot secure fixed splits because files are missing: {missing}. "
                "Restore them rather than regenerating the split."
            )
        info = self.active
        info["private_data_id"] = secrets.token_hex(16)
        self.save_active(info)
        private = self.private_data_dir
        private.mkdir(parents=True, mode=0o700)
        os.chmod(private, 0o700)
        for split in ("val", "test"):
            source = sources[split]
            target = private / f"{split}.csv"
            target.write_bytes(source.read_bytes())
            os.chmod(target, 0o600)
            source.unlink()

    # ------------------------------------------------------------ activation

    def activate(self, candidate_name: str, test_score: float | None) -> None:
        """Point the shipped package at one candidate.

        Writes active.json (active, test_score, activated_at, current
        metric_sha) and regenerates pyproject.toml so the package's
        dependencies are exactly the candidate's runtime deps. Switching
        candidates is a one-line diff you can review, commit, and revert.
        """
        candidate = self._load_candidate(candidate_name)
        info = self.active
        info["active"] = candidate.name
        info["test_score"] = test_score
        info["activated_at"] = _utcnow_iso()
        info["metric_sha"] = self._metric_sha()
        self.save_active(info)
        self.regen_pyproject(candidate)

    def mark_finalized(self, report: dict) -> None:
        """Write final_report.json and flag the workspace as finalized."""
        self.final_report.write_text(json.dumps(report, indent=2) + "\n")
        info = self.active
        info["finalized"] = True
        self.save_active(info)

    def regen_pyproject(self, candidate) -> None:
        """Rewrite pyproject.toml wholesale from the given candidate.

        The file is generated, never hand-edited: dependencies become the
        candidate's PEP 723 list minus the self-reference, and
        requires-python follows the candidate (default ``>=3.11``).
        """
        deps = self._runtime_deps(candidate)
        requires = candidate.requires_python or ">=3.11"
        try:
            description = tomllib.loads(self.pyproject.read_text())["project"]["description"]
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            description = self.schema.doc
        self.pyproject.write_text(self._render_pyproject(requires, deps, description))

    # -------------------------------------------------------------- internal

    def _render_pyproject(self, requires_python: str, dependencies, description: str) -> str:
        pkg = self.package_name
        return (
            "# Generated by autoprogramming; activate() rewrites this file — do not hand-edit.\n"
            "[project]\n"
            f"name = {json.dumps(self.dist_name)}\n"
            'version = "0.1.0"\n'
            f"description = {json.dumps(description)}\n"
            f"requires-python = {json.dumps(requires_python)}\n"
            f"dependencies = {json.dumps(list(dependencies))}\n"
            "\n"
            "[build-system]\n"
            'requires = ["setuptools>=69"]\n'
            'build-backend = "setuptools.build_meta"\n'
            "\n"
            "[tool.setuptools]\n"
            f"packages = [{json.dumps(pkg)}]\n"
            f'package-dir = {{{json.dumps(pkg)} = "."}}\n'
            "\n"
            "[tool.setuptools.package-data]\n"
            f'{json.dumps(pkg)} = ["active.json", "candidates/*.py", "artifacts/*", "artifacts/**/*"]\n'
        )

    def _metric_sha(self) -> str | None:
        if not self.metric_py.exists():
            return None
        return hashlib.sha256(self.metric_py.read_bytes()).hexdigest()

    def _runtime_deps(self, candidate) -> tuple[str, ...]:
        try:
            from .candidates import runtime_deps
        except ModuleNotFoundError:
            target = _normalize_dist(self.dist_name)
            return tuple(
                d for d in candidate.dependencies if _requirement_name(d) != target
            )
        return tuple(runtime_deps(candidate, self.dist_name))

    def _load_candidate(self, name: str):
        try:
            from .candidates import load_candidate
        except ModuleNotFoundError:
            return self._load_candidate_fallback(name)
        return load_candidate(self, name)

    def _load_candidate_fallback(self, name: str) -> _MinimalCandidate:
        stem = name[:-3] if name.endswith(".py") else name
        path = self.candidates_dir / f"{stem}.py"
        if not path.exists():
            available = sorted(p.stem for p in self.candidates_dir.glob("*.py"))
            raise CandidateError(
                f"Cannot activate {stem!r}: {path} does not exist. Activation "
                f"points the shipped package at one candidate file, so the file "
                f"must be in candidates/. Available candidates: "
                f"{', '.join(available) if available else '(none yet)'}."
            )
        meta = self._parse_pep723(path)
        return _MinimalCandidate(
            name=stem,
            path=path,
            requires_python=meta.get("requires-python"),
            dependencies=tuple(meta.get("dependencies", ())),
        )

    def _parse_pep723(self, path: Path) -> dict:
        source = path.read_text()
        blocks = [
            m for m in _PEP723_BLOCK.finditer(source) if m.group("type") == "script"
        ]
        if not blocks:
            return {}
        if len(blocks) > 1:
            raise CandidateError(
                f"{path} contains {len(blocks)} PEP 723 script blocks; a "
                f"candidate carries exactly one so its dependencies are "
                f"unambiguous. Merge them into a single '# /// script' block."
            )
        content = "".join(
            line[2:] if line.startswith("# ") else line[1:]
            for line in blocks[0].group("content").splitlines(keepends=True)
        )
        try:
            return tomllib.loads(content)
        except tomllib.TOMLDecodeError as exc:
            raise CandidateError(
                f"The PEP 723 block in {path} is not valid TOML ({exc}). The "
                f"block is the candidate's dependency manifest — uv and the "
                f"packager both read it — so it must parse. Fix the TOML."
            ) from exc
