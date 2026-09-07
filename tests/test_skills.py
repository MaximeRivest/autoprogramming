"""Agent Skills spec validation for every SKILL.md this repo produces.

Validates the agentskills.io hard requirements — frontmatter first, portable
name/description metadata, body size — against:

- src/autoprogramming/skills/  (package data, the canonical optimizer skill)
- skills/                      (repo-level skills, discovered when present)
- the copies Workspace.create() embeds in a freshly generated workspace

Discovery happens at test time (glob), so skills added by other authors are
covered automatically without failing while their directory is absent.
"""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path

import pytest

from autoprogramming.schema import Schema
from autoprogramming.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]
SKILL_ROOTS = (REPO / "skills", REPO / "src" / "autoprogramming" / "skills")

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Tool-specific / non-portable frontmatter fields; portable skills carry only
# name and description (plus spec-optional extras like license or version).
FORBIDDEN_FRONTMATTER_FIELDS = {
    "context", "agent", "hooks", "paths", "allowed-tools", "openai",
}

MAX_BODY_LINES = 500


# ------------------------------------------------------------ the validator


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into (frontmatter fields, body).

    Tolerant single-purpose YAML subset: top-level ``key: value`` lines with
    indented continuation lines folded in (covers plain and block scalars).
    """
    assert text.startswith("---\n"), (
        "frontmatter must be the FIRST content (file must open with ---)"
    )
    end = text.find("\n---\n", 4)
    assert end != -1, "frontmatter is never closed with ---"
    block, body = text[4:end], text[end + len("\n---\n"):]

    fields: dict[str, str] = {}
    current: str | None = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] not in " \t":
            key, sep, value = line.partition(":")
            assert sep, f"unparseable frontmatter line: {line!r}"
            current = key.strip()
            fields[current] = value.strip()
        else:
            assert current is not None, f"stray indented line: {line!r}"
            fields[current] = (fields[current] + " " + line.strip()).strip()
    return fields, body


def _scalar(value: str) -> str:
    """Strip block-scalar indicators and surrounding quotes from a value."""
    parts = value.split(None, 1)
    if parts and re.fullmatch(r"[>|][+-]?\d*", parts[0]):
        value = parts[1] if len(parts) > 1 else ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value.strip()


def validate_skill(path: Path) -> None:
    """Assert one SKILL.md meets the Agent Skills spec hard requirements."""
    text = path.read_text(encoding="utf-8")
    fields, body = _parse_frontmatter(text)

    name = _scalar(fields.get("name", ""))
    assert 1 <= len(name) <= 64, f"{path}: name must be 1-64 chars, got {name!r}"
    assert NAME_RE.fullmatch(name), (
        f"{path}: name must be lowercase alphanumerics and hyphens with no "
        f"leading/trailing/consecutive hyphens, got {name!r}"
    )
    assert name == path.parent.name, (
        f"{path}: name {name!r} must equal the parent directory name "
        f"{path.parent.name!r}"
    )

    description = _scalar(fields.get("description", ""))
    assert 1 <= len(description) <= 1024, (
        f"{path}: description must be 1-1024 chars, got {len(description)}"
    )
    assert "when" in description.lower(), (
        f"{path}: description must state when to use the skill"
    )

    bad = FORBIDDEN_FRONTMATTER_FIELDS & set(fields)
    assert not bad, (
        f"{path}: tool-specific frontmatter fields hurt portability: {sorted(bad)}"
    )

    assert len(body.splitlines()) < MAX_BODY_LINES, (
        f"{path}: body must stay under {MAX_BODY_LINES} lines; move long "
        f"reference material to references/*.md"
    )


def repo_skill_files() -> list[Path]:
    """Every SKILL.md under the repo skill roots, discovered right now."""
    files: list[Path] = []
    for root in SKILL_ROOTS:
        if root.is_dir():
            files.extend(sorted(root.rglob("SKILL.md")))
    return files


# ----------------------------------------------------------------- the tests


def test_repo_skills_pass_the_spec():
    files = repo_skill_files()
    # The packaged candidate-optimizer skill must always be there; repo-level
    # skills/ is validated whenever it exists but its absence is not a failure.
    assert any(
        f.parent.name == "candidate-optimizer" and "src" in f.parts for f in files
    ), f"packaged candidate-optimizer skill not found; discovered: {files}"
    for path in files:
        validate_skill(path)


def shout(text: str) -> str:
    """Uppercase the text."""


def test_generated_workspace_skill_copies_pass_the_spec(tmp_path):
    rows = [{"text": t, "str": t.upper()} for t in ("a", "b", "c")]
    ws = Workspace.create(
        tmp_path / "shout_ap",
        Schema.from_function(shout),
        {"train": rows[:1], "val": rows[1:2], "test": rows[2:]},
        seed=0,
        ratios=(0.6, 0.2, 0.2),
        data_sha="deadbeef",
        bootstrap=True,
    )
    source = (
        importlib.resources.files("autoprogramming")
        / "skills" / "candidate-optimizer" / "SKILL.md"
    ).read_bytes()
    copies = [
        ws.root / d / "skills" / "candidate-optimizer" / "SKILL.md"
        for d in (".agents", ".claude")
    ]
    for copy in copies:
        assert copy.is_file(), copy
        assert copy.read_bytes() == source, copy
        validate_skill(copy)


def test_packaged_skill_is_readable_via_importlib_resources():
    resource = (
        importlib.resources.files("autoprogramming")
        / "skills" / "candidate-optimizer" / "SKILL.md"
    )
    text = resource.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: candidate-optimizer" in text.split("\n---\n", 1)[0]


def test_validator_rejects_frontmatter_not_first(tmp_path):
    bad = tmp_path / "not-first" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text("# heading before frontmatter\n---\nname: not-first\n---\n")
    with pytest.raises(AssertionError, match="FIRST content"):
        validate_skill(bad)


def test_validator_rejects_name_dir_mismatch(tmp_path):
    bad = tmp_path / "some-dir" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text(
        "---\nname: other-name\ndescription: Use when testing.\n---\nBody.\n"
    )
    with pytest.raises(AssertionError, match="parent directory"):
        validate_skill(bad)


def test_validator_rejects_tool_specific_fields(tmp_path):
    bad = tmp_path / "gadget" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text(
        "---\nname: gadget\ndescription: Use when testing.\n"
        "allowed-tools: Bash\n---\nBody.\n"
    )
    with pytest.raises(AssertionError, match="portability"):
        validate_skill(bad)
