"""``autoprogramming`` command line: install the agent skills where Pi finds them.

``uv add autoprogramming`` puts the library on the path, but a coding agent
discovers skills from ``~/.agents/skills`` (or a project's ``.agents/skills``),
never from site-packages. Without this step a fresh Pi session has no idea what
the library is for. The skills ship inside the package as the single source of
truth; this command copies them out, byte for byte.
"""

from __future__ import annotations

import argparse
import importlib.resources
import shutil
import sys
from pathlib import Path

#: Skill directories shipped in the package, in install order.
SKILLS = ("autoprogramming", "candidate-optimizer")

#: Discovery roots understood by Claude Code, Codex, Cursor, Gemini, OpenCode,
#: Amp, Windsurf, Pi. The first is the portable default.
DISCOVERY_DIRNAMES = (".agents", ".claude")


def _packaged_skill(name: str) -> Path:
    resource = importlib.resources.files("autoprogramming") / "skills" / name
    path = Path(str(resource))
    if not (path / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"Packaged skill {name!r} is missing its SKILL.md at {path}; the "
            "installed autoprogramming distribution is incomplete."
        )
    return path


def install_skills(
    target_root: Path, *, names: tuple[str, ...] = SKILLS, force: bool = False
) -> list[Path]:
    """Copy each packaged skill to ``target_root/<name>/``; return written paths.

    Refuses to overwrite an existing, different skill unless ``force`` so a
    user's local edits are never clobbered silently. Identical copies are
    refreshed without complaint.
    """
    target_root = Path(target_root).expanduser()
    written: list[Path] = []
    for name in names:
        source = _packaged_skill(name)
        target = target_root / name
        if target.exists() and not force and not _same_tree(source, target):
            raise FileExistsError(
                f"{target} already exists and differs from the packaged skill. "
                "Re-run with --force to replace it, or remove it first."
            )
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        written.append(target)
    return written


def _same_tree(a: Path, b: Path) -> bool:
    files_a = {p.relative_to(a): p.read_bytes() for p in a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(b): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    return files_a == files_b


def default_skill_root(*, project: bool) -> Path:
    base = Path.cwd() if project else Path.home()
    return base / DISCOVERY_DIRNAMES[0] / "skills"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autoprogramming",
        description="AutoProgramming helper commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser(
        "install-skill",
        help="Copy the agent skills into a skills directory Pi/Claude discover.",
    )
    install.add_argument(
        "--project",
        action="store_true",
        help="Install into ./.agents/skills of the current project instead of ~/.agents/skills.",
    )
    install.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Explicit skills directory (overrides --project / the home default).",
    )
    install.add_argument(
        "--force", action="store_true", help="Replace an existing, edited copy."
    )
    args = parser.parse_args(argv)

    if args.command == "install-skill":
        root = args.dir if args.dir is not None else default_skill_root(project=args.project)
        try:
            written = install_skills(root, force=args.force)
        except (FileExistsError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for path in written:
            print(f"installed {path}")
        print(
            "Start Pi in your project and ask it to use the 'autoprogramming' "
            "skill to build a program from examples."
        )
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
