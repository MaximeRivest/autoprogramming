"""`autoprogramming install-skill` puts the packaged skills where agents look."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from autoprogramming import cli


def _packaged(name: str) -> Path:
    return Path(str(importlib.resources.files("autoprogramming") / "skills" / name))


def test_front_end_skill_ships_in_the_package():
    assert (_packaged("autoprogramming") / "SKILL.md").is_file()
    assert (_packaged("autoprogramming") / "references" / "prg-api.md").is_file()


def test_install_skill_copies_both_skills_byte_for_byte(tmp_path, capsys):
    assert cli.main(["install-skill", "--dir", str(tmp_path)]) == 0
    for name in cli.SKILLS:
        src = _packaged(name)
        for file in src.rglob("*"):
            if file.is_file():
                rel = file.relative_to(src)
                assert (tmp_path / name / rel).read_bytes() == file.read_bytes()
    out = capsys.readouterr().out
    assert f"installed {tmp_path / 'autoprogramming'}" in out


def test_install_skill_refuses_to_clobber_edits_without_force(tmp_path, capsys):
    cli.main(["install-skill", "--dir", str(tmp_path)])
    edited = tmp_path / "autoprogramming" / "SKILL.md"
    edited.write_text(edited.read_text() + "\nlocal note\n")
    assert cli.main(["install-skill", "--dir", str(tmp_path)]) == 1
    assert "differs" in capsys.readouterr().err
    assert "local note" in edited.read_text()
    assert cli.main(["install-skill", "--dir", str(tmp_path), "--force"]) == 0
    assert "local note" not in edited.read_text()
    # An identical copy is refreshed silently.
    assert cli.main(["install-skill", "--dir", str(tmp_path)]) == 0


def test_default_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cli.default_skill_root(project=True) == tmp_path / ".agents" / "skills"
    assert cli.default_skill_root(project=False) == Path.home() / ".agents" / "skills"
