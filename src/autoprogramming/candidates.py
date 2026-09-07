"""Candidate implementations as PEP 723 single-file scripts.

A candidate is a plain ``candidates/candidate_<n>.py`` file defining
``predict(...)``. Its optional ``# /// script`` block declares dependencies,
``[tool.uv.sources]``, and ``[tool.ap]`` hints (deterministic, cost_per_call,
fetch). This module parses, lists, and creates candidate files; it never
executes them — runner.py does that.
"""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import CandidateError

PEP723_REGEX = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"

_CANDIDATE_NAME_RE = re.compile(r"^candidate_(\d+)$")
_REQUIREMENT_NAME_RE = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def pep503_normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503 (runs of ``-_.`` → ``-``, lowercased)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    """The bare distribution name of a PEP 508 requirement string."""
    match = _REQUIREMENT_NAME_RE.match(requirement)
    return match.group(1) if match else requirement.strip()


def _normalize_newlines(source: str) -> str:
    """CRLF/CR line endings become LF — the form ``read_text()`` yields.

    Candidate files are always read back with universal-newline translation,
    so parsing and writing must see the same bytes; the PEP 723 reference
    regex (``^# ///$``) cannot match a line that still ends in ``\\r``.
    """
    return source.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class Candidate:
    """One candidate implementation: a single ``.py`` file plus its parsed metadata.

    A candidate with no ``# /// script`` block is valid — it is stdlib-only,
    with ``dependencies=()`` (the README's regex candidate has no block).
    """

    name: str
    path: Path
    source: str
    requires_python: str | None = None
    dependencies: tuple[str, ...] = ()
    tool_ap: dict = field(default_factory=dict)
    uv_sources: dict = field(default_factory=dict)

    @property
    def deterministic(self) -> bool:
        """Whether ``[tool.ap] deterministic`` was declared true (default False).

        Deterministic candidates are evaluated with a single repeat; stochastic
        ones (LLM calls) get repeated runs so their variance is reported honestly.
        """
        return bool(self.tool_ap.get("deterministic", False))

    @property
    def cost_per_call(self) -> float | None:
        """``[tool.ap] cost_per_call`` as a float, or None when not declared."""
        value = self.tool_ap.get("cost_per_call")
        if value is None:
            return None
        cost = float(value)
        if not math.isfinite(cost) or cost < 0:
            raise CandidateError(
                f"[tool.ap] cost_per_call must be a finite non-negative dollar "
                f"amount, got {value!r}."
            )
        return cost

    @property
    def fetch(self) -> tuple[str, ...]:
        """``[tool.ap] fetch`` entries (artifact download steps), possibly empty."""
        return tuple(self.tool_ap.get("fetch", ()))

    @property
    def pi_runtime(self) -> bool:
        """Whether prediction calls the authenticated local Pi runtime."""
        if bool(self.tool_ap.get("pi_runtime", False)):
            return True
        # Placement defense for composed/legacy candidates that omitted the
        # metadata. Pi-assigned avenues still fail adherence until they declare
        # it explicitly.
        return bool(
            re.search(r"(?is)subprocess.{0,500}?[\"']pi[\"']", self.source)
            or re.search(
                r"(?is)[\"']pi[\"']\s*,\s*[\"']--(?:mode|print)",
                self.source,
            )
        )

    @property
    def network_required(self) -> bool:
        """Whether prediction depends on a network service at runtime."""
        return bool(self.tool_ap.get("network_required", False))

    @property
    def api_providers(self) -> tuple[str, ...]:
        """Declared runtime provider names, without credentials."""
        return tuple(str(v) for v in self.tool_ap.get("api_providers", ()))

    @property
    def compute_heavy(self) -> bool:
        """Explicit or dependency-derived search-compute placement hint."""
        declared = self.tool_ap.get("compute_heavy")
        if declared is not None:
            return bool(declared)
        heavy = " ".join((*self.dependencies, *self.fetch)).casefold()
        return any(token in heavy for token in (
            "torch", "tensorflow", "jax", "transformers", "onnxruntime",
            "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
            "opencv", "sentence-transformers", "huggingface:",
        ))

    @property
    def artifact_namespace(self) -> str | None:
        """Optional isolated subdirectory under the package's artifacts/."""
        value = self.tool_ap.get("artifact_namespace")
        if value is None:
            return None
        value = str(value)
        if not value.replace("-", "_").isidentifier() or "/" in value or "\\" in value:
            raise CandidateError(
                f"Invalid [tool.ap] artifact_namespace {value!r}; use one simple "
                "identifier-like directory name."
            )
        return value


def parse_pep723(source: str) -> dict | None:
    """Parse the ``# /// script`` metadata block out of candidate source.

    Uses the PEP 723 reference regex. Returns the parsed TOML dict, or None
    when the source has no script block. Raises CandidateError on malformed
    TOML or on more than one script block. Line endings are normalized first,
    so CRLF source declares the same metadata it will show once on disk.
    """
    source = _normalize_newlines(source)
    matches = [
        m for m in re.finditer(PEP723_REGEX, source) if m.group("type") == "script"
    ]
    if len(matches) > 1:
        raise CandidateError(
            f"Refusing to parse this candidate: it contains {len(matches)} "
            f"`# /// script` blocks, and PEP 723 allows at most one per file so "
            f"tools know which metadata is authoritative. Merge them into a "
            f"single block."
        )
    if not matches:
        return None
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in matches[0].group("content").splitlines(keepends=True)
    )
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        raise CandidateError(
            f"Refusing to parse this candidate: its `# /// script` block is not "
            f"valid TOML ({exc}). That block is what `uv run` and the packager "
            f"read to build the candidate's environment, so it must parse. Fix "
            f"the TOML, or delete the block for a stdlib-only candidate."
        ) from exc


def _candidate_from_source(name: str, path: Path, source: str) -> Candidate:
    try:
        meta = parse_pep723(source) or {}
    except CandidateError as exc:
        raise CandidateError(f"{path.name}: {exc}") from exc
    tool = meta.get("tool") or {}
    return Candidate(
        name=name,
        path=path,
        source=source,
        requires_python=meta.get("requires-python"),
        dependencies=tuple(str(dep) for dep in meta.get("dependencies", ())),
        tool_ap=dict(tool.get("ap") or {}),
        uv_sources=dict((tool.get("uv") or {}).get("sources") or {}),
    )


def _indexed_paths(workspace) -> list[tuple[int, Path]]:
    directory = Path(workspace.candidates_dir)
    if not directory.is_dir():
        return []
    found = [
        (int(m.group(1)), path)
        for path in directory.glob("candidate_*.py")
        if (m := _CANDIDATE_NAME_RE.match(path.stem))
    ]
    found.sort()
    return found


def list_candidates(workspace) -> dict[str, Candidate]:
    """All candidates in the workspace, keyed by name, sorted by numeric index."""
    return {
        path.stem: _candidate_from_source(path.stem, path, path.read_text(encoding="utf-8"))
        for _, path in _indexed_paths(workspace)
    }


def load_candidate(workspace, name: str) -> Candidate:
    """Load one candidate by name (``"candidate_3"`` or ``"candidate_3.py"``)."""
    base = name[:-3] if name.endswith(".py") else name
    path = Path(workspace.candidates_dir) / f"{base}.py"
    if not path.is_file():
        available = ", ".join(path.stem for _, path in _indexed_paths(workspace)) or "none yet"
        raise CandidateError(
            f"No candidate named {base!r} in {workspace.candidates_dir} — "
            f"candidates are plain .py files on disk, not registry entries, so "
            f"only files that exist can be loaded. Available: {available}. "
            f"Check the name, or create one with new_candidate(source=...)."
        )
    return _candidate_from_source(base, path, path.read_text(encoding="utf-8"))


def next_name(workspace) -> str:
    """The next unused candidate name: ``candidate_<max+1>``, or ``candidate_0``."""
    indices = [index for index, _ in _indexed_paths(workspace)]
    return f"candidate_{max(indices) + 1 if indices else 0}"


def new_candidate(workspace, source: str | None = None, from_: str | None = None) -> Candidate:
    """Create the next candidate file from fresh source or by copying another.

    Exactly one of ``source`` (new file contents) or ``from_`` (name of an
    existing candidate to copy) must be given. The metadata is validated
    before anything is written, so a malformed block never lands on disk.
    """
    if (source is None) == (from_ is None):
        got = "both" if source is not None else "neither"
        raise CandidateError(
            f"new_candidate() takes exactly one of source= (fresh file contents) "
            f"or from_= (an existing candidate to copy), but got {got}. Every "
            f"candidate has exactly one origin so its lineage stays reviewable; "
            f"pass the one you mean."
        )
    if from_ is not None:
        source = load_candidate(workspace, from_).source
    source = _normalize_newlines(source)
    name = next_name(workspace)
    path = Path(workspace.candidates_dir) / f"{name}.py"
    candidate = _candidate_from_source(name, path, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return candidate


def source_sha(candidate: Candidate) -> str:
    """Stable identity of the implementation source alone."""
    return hashlib.sha256(candidate.source.encode("utf-8")).hexdigest()


def bundle_sha(workspace, candidate: Candidate) -> str:
    """Identity of source plus the candidate's declared artifact namespace."""
    digest = hashlib.sha256(candidate.source.encode("utf-8"))
    namespace = getattr(candidate, "artifact_namespace", None)
    if namespace:
        root = Path(workspace.artifacts_dir) / namespace
        if root.exists():
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(b"\0")
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def runtime_deps(candidate: Candidate, dist_name: str) -> tuple[str, ...]:
    """The candidate's dependencies minus the self-reference to the workspace package.

    Candidates depend on their own package (``"translate-ap"``) for schema and
    paths imports; at run time the runner injects the package via sys.path
    instead, so that requirement is stripped. Matching is by PEP 503
    normalization of the requirement's name part.
    """
    self_norm = pep503_normalize(dist_name)
    return tuple(
        dep
        for dep in candidate.dependencies
        if pep503_normalize(_requirement_name(dep)) != self_norm
    )
