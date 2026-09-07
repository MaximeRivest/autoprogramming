"""Tests for workspace.py — scaffolding, activation, and the generated package."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.resources
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

import autoprogramming
from autoprogramming.errors import CandidateError, WorkspaceError
from autoprogramming.schema import Schema
from autoprogramming.workspace import TEST_CSV_PERMS, Workspace


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


class Answer(str):
    """Direct answer to the question, one sentence."""


class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""


def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""


class Fancy(str):
    """Output."""


def fancy(text: str) -> Fancy:
    'He said "bonjour" and used a \\ backslash.'


SHOUT_ROWS = [
    {"text": t, "Loud": t.upper() + "!"}
    for t in ("hello", "goodbye", "thanks", "please", "sorry", "welcome")
]

CANDIDATE_SRC = 'def predict(text):\n    return text.upper() + "!"\n'

DEPS_CANDIDATE_SRC = '''\
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.0", "Shout_AP"]
#
# [tool.uv.sources]
# Shout_AP = { path = "..", editable = true }
# ///
def predict(text):
    return text.upper() + "!"
'''


def shout_splits():
    return {"train": SHOUT_ROWS[:4], "val": SHOUT_ROWS[4:5], "test": SHOUT_ROWS[5:]}


def make_workspace(tmp_path, name="shout_ap", fn=shout, splits=None):
    schema = Schema.from_function(fn)
    return Workspace.create(
        tmp_path / name,
        schema,
        splits or shout_splits(),
        seed=0,
        ratios=(0.6, 0.2, 0.2),
        data_sha="deadbeef",
        bootstrap=True,
    )


def write_candidate(ws, name, source=CANDIDATE_SRC):
    path = ws.candidates_dir / f"{name}.py"
    path.write_text(source)
    return path


@pytest.fixture
def import_pkg(monkeypatch):
    """Import a generated package from a directory, hermetically per test.

    Ensures autoprogramming.program exists (the generated schema.py references
    it when the optimizer is importable) and clears sys.modules afterwards.
    """
    monkeypatch.setattr(autoprogramming, "program", lambda fn: fn, raising=False)
    imported: list[str] = []

    def _import(directory, pkg):
        monkeypatch.syspath_prepend(str(directory))
        for key in [k for k in sys.modules if k == pkg or k.startswith(pkg + ".")]:
            sys.modules.pop(key)
        module = importlib.import_module(pkg)
        imported.append(pkg)
        return module

    yield _import
    for pkg in imported:
        for key in [k for k in sys.modules if k == pkg or k.startswith(pkg + ".")]:
            sys.modules.pop(key)


# --------------------------------------------------------------- scaffolding


def test_create_scaffolds_all_files(tmp_path):
    ws = make_workspace(tmp_path)
    root = tmp_path / "shout_ap"
    for rel in (
        "schema.py", "paths.py", "__init__.py", "pyproject.toml", "active.json",
        "scores.json", "data/train.csv", "data/val.csv", "data/test.csv",
        "data/split.json", "artifacts/.gitkeep",
    ):
        assert (root / rel).exists(), rel
    assert (root / "candidates").is_dir()

    active = json.loads((root / "active.json").read_text())
    assert active["program"] == "shout"
    assert active["active"] is None
    assert active["test_score"] is None
    assert active["activated_at"] is None
    assert active["metric_sha"] is None
    assert active["finalized"] is False
    datetime.fromisoformat(active["created_at"])

    split = json.loads((root / "data" / "split.json").read_text())
    assert split == {
        "seed": 0,
        "ratios": [0.6, 0.2, 0.2],
        "counts": {"train": 4, "val": 1, "test": 1},
        "data_sha": "deadbeef",
        "bootstrap": True,
    }

    scores = json.loads((root / "scores.json").read_text())
    assert scores == {"metric_sha": None, "candidates": {}, "val_scored": [], "flags": {}}

    assert "class Loud" in (root / "schema.py").read_text()
    paths_src = (root / "paths.py").read_text()
    assert 'artifacts = Path(__file__).parent / "artifacts"' in paths_src
    assert ws.package_name == "shout_ap"
    assert ws.dist_name == "shout-ap"
    assert ws.program_name == "shout"


def test_create_embeds_candidate_optimizer_skill(tmp_path):
    """Both discovery roots get byte-identical copies of the packaged skill."""
    ws = make_workspace(tmp_path)
    source = (
        importlib.resources.files("autoprogramming")
        / "skills" / "candidate-optimizer" / "SKILL.md"
    ).read_bytes()
    for discovery_root in (".agents", ".claude"):
        copy = ws.root / discovery_root / "skills" / "candidate-optimizer" / "SKILL.md"
        assert copy.is_file(), discovery_root
        assert copy.read_bytes() == source, discovery_root


def test_embedded_skill_is_not_in_generated_package_data(tmp_path):
    """Skills are dev-time files; the shipped package must not carry them."""
    ws = make_workspace(tmp_path)
    doc = tomllib.loads(ws.pyproject.read_text())
    package_data = doc["tool"]["setuptools"]["package-data"]["shout_ap"]
    assert not any("skill" in pattern.lower() for pattern in package_data)
    assert not any(pattern.startswith(".") for pattern in package_data)


def test_created_pyproject_is_valid_and_installable_shape(tmp_path):
    make_workspace(tmp_path)
    doc = tomllib.loads((tmp_path / "shout_ap" / "pyproject.toml").read_text())
    assert doc["project"]["name"] == "shout-ap"
    assert doc["project"]["version"] == "0.1.0"
    assert doc["project"]["description"] == "Uppercase the text."
    assert doc["project"]["requires-python"] == ">=3.11"
    assert doc["project"]["dependencies"] == []
    assert doc["build-system"]["build-backend"] == "setuptools.build_meta"
    assert doc["tool"]["setuptools"]["packages"] == ["shout_ap"]
    assert doc["tool"]["setuptools"]["package-dir"] == {"shout_ap": "."}
    assert doc["tool"]["setuptools"]["package-data"]["shout_ap"] == [
        "active.json", "candidates/*.py", "artifacts/*", "artifacts/**/*",
    ]


def test_pyproject_description_with_quotes_survives_toml(tmp_path):
    ws = make_workspace(tmp_path, name="fancy_ap", fn=fancy, splits={
        "train": [{"text": "a", "Fancy": "b"}],
        "val": [{"text": "c", "Fancy": "d"}],
        "test": [{"text": "e", "Fancy": "f"}],
    })
    doc = tomllib.loads(ws.pyproject.read_text())
    assert doc["project"]["description"] == 'He said "bonjour" and used a \\ backslash.'


def test_test_csv_is_readonly(tmp_path):
    ws = make_workspace(tmp_path)
    mode = (ws.data_dir / "test.csv").stat().st_mode & 0o777
    assert mode == TEST_CSV_PERMS == 0o400


def test_split_csvs_roundtrip(tmp_path):
    ws = make_workspace(tmp_path)
    with (ws.data_dir / "train.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["text", "Loud"]
        rows = list(reader)
    assert rows == SHOUT_ROWS[:4]


def test_create_refuses_nonempty_dir(tmp_path):
    root = tmp_path / "busy_ap"
    root.mkdir()
    (root / "precious.txt").write_text("do not clobber")
    with pytest.raises(WorkspaceError) as exc:
        make_workspace(tmp_path, name="busy_ap")
    msg = str(exc.value)
    assert "not empty" in msg
    assert "load" in msg
    assert (root / "precious.txt").read_text() == "do not clobber"


def test_create_accepts_empty_existing_dir(tmp_path):
    (tmp_path / "shout_ap").mkdir()
    ws = make_workspace(tmp_path)
    assert ws.active_json.exists()


def test_create_refuses_missing_split(tmp_path):
    with pytest.raises(WorkspaceError) as exc:
        make_workspace(tmp_path, splits={"train": SHOUT_ROWS[:4], "val": SHOUT_ROWS[4:5]})
    assert "test" in str(exc.value)


@pytest.mark.parametrize("bad", ["shout-ap", "1shout", "class"])
def test_invalid_package_name_refused(tmp_path, bad):
    with pytest.raises(WorkspaceError) as exc:
        Workspace(tmp_path / bad)
    msg = str(exc.value)
    assert "identifier" in msg
    assert "_ap" in msg


# ---------------------------------------------------------------- load/state


def test_load_roundtrip(tmp_path):
    make_workspace(tmp_path)
    ws = Workspace.load(tmp_path / "shout_ap")
    assert ws.program_name == "shout"
    assert ws.package_name == "shout_ap"
    assert ws.dist_name == "shout-ap"
    assert ws.split_json.exists()


def test_load_refusals(tmp_path):
    with pytest.raises(WorkspaceError) as exc:
        Workspace.load(tmp_path / "absent_ap")
    assert "does not exist" in str(exc.value)

    bare = tmp_path / "bare_ap"
    bare.mkdir()
    with pytest.raises(WorkspaceError) as exc:
        Workspace.load(bare)
    assert "active.json" in str(exc.value)

    broken = make_workspace(tmp_path, name="broken_ap")
    broken.schema_py.unlink()
    with pytest.raises(WorkspaceError) as exc:
        Workspace.load(broken.root)
    assert "schema.py" in str(exc.value)


def test_schema_property_from_generated_module(tmp_path, monkeypatch):
    monkeypatch.setattr(autoprogramming, "program", lambda fn: fn, raising=False)
    make_workspace(tmp_path)
    ws = Workspace.load(tmp_path / "shout_ap")
    schema = ws.schema
    assert schema.name == "shout"
    assert schema.input_names == ("text",)
    assert schema.output_names == ("Loud",)
    assert ws.schema is schema


def test_tmp_dir_created_lazily(tmp_path):
    ws = make_workspace(tmp_path)
    assert not (ws.root / ".ap").exists()
    assert ws.tmp_dir == ws.root / ".ap"
    assert (ws.root / ".ap").is_dir()


# ---------------------------------------------------------------- activation


def test_activate_writes_active_json_and_regens_pyproject(tmp_path):
    ws = make_workspace(tmp_path)
    write_candidate(ws, "candidate_0")
    ws.activate("candidate_0", 0.9)

    active = ws.active
    assert active["active"] == "candidate_0"
    assert active["test_score"] == 0.9
    assert active["metric_sha"] is None
    datetime.fromisoformat(active["activated_at"])

    doc = tomllib.loads(ws.pyproject.read_text())
    assert doc["project"]["dependencies"] == []
    assert doc["project"]["requires-python"] == ">=3.11"
    assert doc["project"]["description"] == "Uppercase the text."


def test_activate_strips_self_reference_and_uses_candidate_python(tmp_path):
    ws = make_workspace(tmp_path)
    write_candidate(ws, "candidate_0")
    write_candidate(ws, "candidate_1", DEPS_CANDIDATE_SRC)
    ws.metric_py.write_text("def metric(predicted, expected):\n    return 1.0\n")
    ws.activate("candidate_1", 0.8)

    active = ws.active
    assert active["active"] == "candidate_1"
    assert active["metric_sha"] == hashlib.sha256(ws.metric_py.read_bytes()).hexdigest()

    doc = tomllib.loads(ws.pyproject.read_text())
    assert doc["project"]["dependencies"] == ["openai>=1.0"]
    assert doc["project"]["requires-python"] == ">=3.12"
    assert doc["project"]["description"] == "Uppercase the text."


def test_activate_missing_candidate_refused(tmp_path):
    ws = make_workspace(tmp_path)
    write_candidate(ws, "candidate_0")
    with pytest.raises(CandidateError) as exc:
        ws.activate("candidate_7", None)
    msg = str(exc.value)
    assert "candidate_7" in msg
    assert "candidate_0" in msg


def test_regen_pyproject_ducktyped_candidate(tmp_path):
    ws = make_workspace(tmp_path)

    class Fake:
        name = "candidate_9"
        requires_python = ">=3.13"
        dependencies = ("requests>=2", "Shout.AP==0.1.0", "numpy")

    ws.regen_pyproject(Fake())
    doc = tomllib.loads(ws.pyproject.read_text())
    assert doc["project"]["dependencies"] == ["requests>=2", "numpy"]
    assert doc["project"]["requires-python"] == ">=3.13"


def test_mark_finalized(tmp_path):
    ws = make_workspace(tmp_path)
    ws.mark_finalized({"entries": [], "activated": None, "val_reliability": "ok"})
    assert json.loads(ws.final_report.read_text())["val_reliability"] == "ok"
    assert ws.active["finalized"] is True


# ------------------------------------------------------- the shipped package


def test_generated_package_end_to_end(tmp_path, import_pkg):
    ws = make_workspace(tmp_path, name="shout_e2e_ap")
    write_candidate(ws, "candidate_0")
    ws.activate("candidate_0", 0.95)

    pkg = import_pkg(tmp_path, "shout_e2e_ap")
    result = pkg.shout("hello")
    assert result == "HELLO!"
    assert type(result).__name__ == "Loud"
    assert isinstance(result, str)
    assert pkg.shout(text="good day") == "GOOD DAY!"
    assert pkg.__all__ == ["shout", "Loud"]

    with pytest.raises(TypeError, match="missing"):
        pkg.shout()
    with pytest.raises(TypeError, match="positional"):
        pkg.shout("a", "b")
    with pytest.raises(TypeError, match="multiple values"):
        pkg.shout("a", text="b")
    with pytest.raises(TypeError, match="unexpected keyword"):
        pkg.shout("a", nope="b")


def test_generated_package_loads_candidate_lazily(tmp_path, import_pkg):
    ws = make_workspace(tmp_path, name="shout_lazy_ap")
    write_candidate(ws, "candidate_0")

    pkg = import_pkg(tmp_path, "shout_lazy_ap")
    with pytest.raises(RuntimeError, match="active"):
        pkg.shout("hello")

    ws.activate("candidate_0", None)
    assert pkg.shout("hello") == "HELLO!"


def test_generated_init_never_imports_autoprogramming(tmp_path):
    ws = make_workspace(tmp_path)
    src = ws.init_py.read_text()
    assert not re.search(r"^\s*(import|from)\s+autoprogramming", src, re.M)
    assert src.startswith('"""shout — optimized by autoprogramming."""')


def test_generated_package_logging_format(tmp_path, import_pkg):
    ws = make_workspace(tmp_path, name="shout_log_ap")
    write_candidate(ws, "candidate_0")
    ws.activate("candidate_0", None)

    pkg = import_pkg(tmp_path, "shout_log_ap")
    assert pkg.shout.enable_logging() is pkg.shout
    pkg.shout("Where is the train station?")

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = ws.logs_dir / f"{day}.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert list(entry.keys()) == ["inputs", "outputs", "candidate", "n_repeat", "timestamp"]
    assert entry["inputs"] == {"text": "Where is the train station?"}
    assert entry["outputs"] == {"Loud": "WHERE IS THE TRAIN STATION?!"}
    assert entry["candidate"] == "candidate_0"
    assert entry["n_repeat"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["timestamp"])

    pkg.shout.disable_logging()
    pkg.shout("quiet now")
    assert len(log_file.read_text().splitlines()) == 1


def test_generated_package_multi_output(tmp_path, import_pkg):
    splits = {
        s: [{"question": "why?", "Answer": "because", "Confidence": "0.5"}]
        for s in ("train", "val", "test")
    }
    ws = make_workspace(tmp_path, name="qa_multi_ap", fn=qa, splits=splits)
    write_candidate(
        ws, "candidate_0",
        'def predict(question):\n    return ("It is blue", 0.75)\n',
    )
    write_candidate(
        ws, "candidate_1",
        'def predict(question):\n'
        '    return {"Confidence": "0.5", "Answer": "No"}\n',
    )
    write_candidate(
        ws, "candidate_2",
        'def predict(question):\n    return "only one thing"\n',
    )
    ws.activate("candidate_0", None)

    pkg = import_pkg(tmp_path, "qa_multi_ap")
    answer, confidence = pkg.qa("Why is the sky blue?")
    assert answer == "It is blue" and type(answer).__name__ == "Answer"
    assert confidence == 0.75 and type(confidence).__name__ == "Confidence"
    assert isinstance(confidence, float)
    assert pkg.__all__ == ["qa", "Answer", "Confidence"]

    ws.activate("candidate_1", None)
    pkg = import_pkg(tmp_path, "qa_multi_ap")
    answer, confidence = pkg.qa("Really?")
    assert answer == "No"
    assert confidence == 0.5 and type(confidence).__name__ == "Confidence"

    ws.activate("candidate_2", None)
    pkg = import_pkg(tmp_path, "qa_multi_ap")
    with pytest.raises(TypeError, match="tuple of 2 outputs"):
        pkg.qa("what?")


def test_generated_multi_output_logging(tmp_path, import_pkg):
    splits = {
        s: [{"question": "why?", "Answer": "because", "Confidence": "0.5"}]
        for s in ("train", "val", "test")
    }
    ws = make_workspace(tmp_path, name="qa_log_ap", fn=qa, splits=splits)
    write_candidate(
        ws, "candidate_0",
        'def predict(question):\n    return ("Sure", 0.9)\n',
    )
    ws.activate("candidate_0", None)

    pkg = import_pkg(tmp_path, "qa_log_ap")
    pkg.qa.enable_logging()
    pkg.qa("Is it?")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = json.loads((ws.logs_dir / f"{day}.jsonl").read_text().splitlines()[0])
    assert entry["inputs"] == {"question": "Is it?"}
    assert entry["outputs"] == {"Answer": "Sure", "Confidence": 0.9}


def test_generated_package_runs_without_optimizer_installed(tmp_path):
    ws = make_workspace(tmp_path, name="shout_nosite_ap")
    write_candidate(ws, "candidate_0")
    ws.activate("candidate_0", None)

    code = (
        "import sys\n"
        "import shout_nosite_ap as pkg\n"
        "v = pkg.shout('hey there')\n"
        "assert v == 'HEY THERE!', v\n"
        "assert type(v).__name__ == 'Loud', type(v)\n"
        "assert isinstance(v, str)\n"
        "assert 'autoprogramming' not in sys.modules\n"
        "print('OK')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ------------------------------------------------- awkward output type names


def echo(text: str) -> str:
    """Echo the text back unchanged."""


ECHO_ROWS = [{"text": t, "str": t} for t in ("a", "b", "c", "d", "e", "f")]


def test_builtin_output_type_star_import_and_all(tmp_path, import_pkg):
    ws = make_workspace(
        tmp_path, name="echo_ap", fn=echo,
        splits={"train": ECHO_ROWS[:4], "val": ECHO_ROWS[4:5], "test": ECHO_ROWS[5:]},
    )
    write_candidate(ws, "candidate_0", "def predict(text):\n    return text\n")
    ws.activate("candidate_0", None)

    pkg = import_pkg(tmp_path, "echo_ap")
    assert pkg.__all__ == ["echo"]
    ns: dict = {}
    exec("from echo_ap import *", ns)
    assert "echo" in ns
    result = pkg.echo("hi")
    assert result == "hi"
    assert type(result) is str


def test_output_type_named_like_a_template_import_still_works(tmp_path, import_pkg):
    path_type = type("Path", (str,), {"__doc__": "A located filesystem path."})

    def locate(text):
        ...

    locate.__doc__ = "Locate the thing."
    locate.__annotations__ = {"text": str, "return": path_type}
    rows = [{"text": t, "Path": "/" + t} for t in ("a", "b", "c", "d", "e", "f")]
    ws = make_workspace(
        tmp_path, name="locate_ap", fn=locate,
        splits={"train": rows[:4], "val": rows[4:5], "test": rows[5:]},
    )
    write_candidate(ws, "candidate_0", 'def predict(text):\n    return "/" + text\n')
    ws.activate("candidate_0", None)

    pkg = import_pkg(tmp_path, "locate_ap")
    result = pkg.locate("x")
    assert result == "/x"
    assert type(result).__name__ == "Path"
    assert issubclass(type(result), str)
    assert pkg.__all__ == ["locate", "Path"]


def test_invalid_name_suggestion_is_not_doubly_suffixed(tmp_path):
    with pytest.raises(WorkspaceError) as exc:
        Workspace(tmp_path / "translate-ap")
    msg = str(exc.value)
    assert "'translate_ap'" in msg
    assert "translate_ap_ap" not in msg
