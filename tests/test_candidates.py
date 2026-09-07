"""Tests for autoprogramming.candidates — PEP 723 parsing and candidate files."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoprogramming.candidates import (
    Candidate,
    list_candidates,
    load_candidate,
    new_candidate,
    next_name,
    parse_pep723,
    pep503_normalize,
    runtime_deps,
)
from autoprogramming.errors import CandidateError

README_STYLE = """\
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "translate-ap"]
#
# [tool.uv.sources]
# translate-ap = { path = "..", editable = true }
# ///
def predict(english):
    return english
"""

STDLIB_ONLY = 'def predict(text):\n    return text.upper() + "!"\n'


def make_ws(tmp_path):
    d = tmp_path / "ws" / "candidates"
    d.mkdir(parents=True)
    return SimpleNamespace(candidates_dir=d)


def write(ws, name, source):
    path = ws.candidates_dir / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


# ------------------------------------------------------------- parse_pep723


def test_parse_readme_style_block():
    meta = parse_pep723(README_STYLE)
    assert meta["requires-python"] == ">=3.11"
    assert meta["dependencies"] == ["openai>=1.0", "translate-ap"]
    assert meta["tool"]["uv"]["sources"]["translate-ap"] == {
        "path": "..",
        "editable": True,
    }


def test_parse_no_block_returns_none():
    assert parse_pep723(STDLIB_ONLY) is None
    assert parse_pep723("") is None


def test_parse_ignores_non_script_blocks():
    source = "# /// test\n# x = 1\n# ///\ndef predict(t):\n    return t\n"
    assert parse_pep723(source) is None


def test_parse_bare_hash_continuation_lines():
    source = '# /// script\n# dependencies = [\n#     "numpy",\n# ]\n#\n# ///\n'
    assert parse_pep723(source) == {"dependencies": ["numpy"]}


def test_parse_malformed_toml_refused():
    source = "# /// script\n# dependencies = [not toml\n# ///\n"
    with pytest.raises(CandidateError) as exc:
        parse_pep723(source)
    assert "TOML" in str(exc.value)


def test_parse_multiple_script_blocks_refused():
    source = (
        "# /// script\n# dependencies = []\n# ///\n"
        "x = 1\n"
        "# /// script\n# dependencies = []\n# ///\n"
    )
    with pytest.raises(CandidateError) as exc:
        parse_pep723(source)
    assert "script" in str(exc.value)


def test_parse_unclosed_block_is_no_block():
    source = '# /// script\n# dependencies = ["numpy"]\ndef predict(t):\n    return t\n'
    assert parse_pep723(source) is None


# ---------------------------------------------------------------- Candidate


def test_candidate_without_block_is_valid(tmp_path):
    ws = make_ws(tmp_path)
    write(ws, "candidate_0", STDLIB_ONLY)
    cand = load_candidate(ws, "candidate_0")
    assert cand.name == "candidate_0"
    assert cand.source == STDLIB_ONLY
    assert cand.dependencies == ()
    assert cand.requires_python is None
    assert cand.tool_ap == {}
    assert cand.uv_sources == {}
    assert cand.deterministic is False
    assert cand.cost_per_call is None
    assert cand.fetch == ()


def test_candidate_tool_ap_properties(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "# /// script\n"
        '# dependencies = ["transformers>=4.40"]\n'
        "# [tool.ap]\n"
        "# deterministic = true\n"
        "# cost_per_call = 0.001\n"
        "# pi_runtime = true\n"
        "# network_required = true\n"
        "# compute_heavy = false\n"
        '# api_providers = ["openai-codex"]\n'
        '# fetch = ["huggingface:Helsinki-NLP/opus-mt-en-fr"]\n'
        "# ///\n"
        "def predict(text):\n    return text\n"
    )
    write(ws, "candidate_0", source)
    cand = load_candidate(ws, "candidate_0")
    assert cand.deterministic is True
    assert cand.cost_per_call == 0.001
    assert cand.pi_runtime is True
    assert cand.network_required is True
    assert cand.compute_heavy is False
    assert cand.api_providers == ("openai-codex",)
    assert cand.fetch == ("huggingface:Helsinki-NLP/opus-mt-en-fr",)
    assert cand.dependencies == ("transformers>=4.40",)


def test_candidate_rejects_negative_or_nonfinite_cost():
    for value in (-0.1, float("inf")):
        candidate = Candidate(
            name="candidate_0", path=Path("candidate_0.py"), source="",
            tool_ap={"cost_per_call": value},
        )
        with pytest.raises(CandidateError, match="non-negative"):
            _ = candidate.cost_per_call


def test_candidate_uv_sources_parsed(tmp_path):
    ws = make_ws(tmp_path)
    write(ws, "candidate_0", README_STYLE)
    cand = load_candidate(ws, "candidate_0")
    assert cand.uv_sources == {"translate-ap": {"path": "..", "editable": True}}
    assert cand.requires_python == ">=3.11"


# ------------------------------------------------------- list / load / next


def test_list_candidates_sorted_numerically(tmp_path):
    ws = make_ws(tmp_path)
    for name in ("candidate_10", "candidate_0", "candidate_2"):
        write(ws, name, STDLIB_ONLY)
    write(ws, "notes", "# not a candidate\n")
    write(ws, "candidate_x", "# not a candidate either\n")
    listed = list_candidates(ws)
    assert list(listed) == ["candidate_0", "candidate_2", "candidate_10"]
    assert all(isinstance(c, Candidate) for c in listed.values())


def test_list_candidates_empty_and_missing_dir(tmp_path):
    ws = make_ws(tmp_path)
    assert list_candidates(ws) == {}
    ws2 = SimpleNamespace(candidates_dir=tmp_path / "nope" / "candidates")
    assert list_candidates(ws2) == {}


def test_load_accepts_py_suffix(tmp_path):
    ws = make_ws(tmp_path)
    write(ws, "candidate_3", STDLIB_ONLY)
    assert load_candidate(ws, "candidate_3.py").name == "candidate_3"
    assert load_candidate(ws, "candidate_3").name == "candidate_3"


def test_load_missing_lists_available(tmp_path):
    ws = make_ws(tmp_path)
    write(ws, "candidate_0", STDLIB_ONLY)
    with pytest.raises(CandidateError) as exc:
        load_candidate(ws, "candidate_7")
    msg = str(exc.value)
    assert "candidate_7" in msg
    assert "candidate_0" in msg


def test_load_missing_when_none_exist(tmp_path):
    ws = make_ws(tmp_path)
    with pytest.raises(CandidateError) as exc:
        load_candidate(ws, "candidate_0")
    assert "none yet" in str(exc.value)


def test_next_name(tmp_path):
    ws = make_ws(tmp_path)
    assert next_name(ws) == "candidate_0"
    write(ws, "candidate_0", STDLIB_ONLY)
    write(ws, "candidate_2", STDLIB_ONLY)
    assert next_name(ws) == "candidate_3"


# ------------------------------------------------------------ new_candidate


def test_new_candidate_from_source(tmp_path):
    ws = make_ws(tmp_path)
    cand = new_candidate(ws, source=STDLIB_ONLY)
    assert cand.name == "candidate_0"
    assert cand.path == ws.candidates_dir / "candidate_0.py"
    assert cand.path.read_text(encoding="utf-8") == STDLIB_ONLY


def test_new_candidate_copy(tmp_path):
    ws = make_ws(tmp_path)
    new_candidate(ws, source=README_STYLE)
    copy = new_candidate(ws, from_="candidate_0")
    assert copy.name == "candidate_1"
    assert copy.source == README_STYLE
    assert copy.dependencies == ("openai>=1.0", "translate-ap")


def test_new_candidate_requires_exactly_one_origin(tmp_path):
    ws = make_ws(tmp_path)
    with pytest.raises(CandidateError):
        new_candidate(ws)
    with pytest.raises(CandidateError):
        new_candidate(ws, source=STDLIB_ONLY, from_="candidate_0")


def test_new_candidate_malformed_block_writes_nothing(tmp_path):
    ws = make_ws(tmp_path)
    bad = "# /// script\n# dependencies = [broken\n# ///\ndef predict(t):\n    return t\n"
    with pytest.raises(CandidateError):
        new_candidate(ws, source=bad)
    assert list(ws.candidates_dir.iterdir()) == []


# ------------------------------------------------------------- runtime_deps


def test_pep503_normalize():
    assert pep503_normalize("Translate_AP") == "translate-ap"
    assert pep503_normalize("translate.ap") == "translate-ap"
    assert pep503_normalize("translate--ap") == "translate-ap"


def test_runtime_deps_strips_self_reference_variants():
    cand = Candidate(
        name="c",
        path=Path("c.py"),
        source="",
        dependencies=(
            "openai>=1.0",
            "Translate_AP",
            "translate.ap==0.1",
            "translate-ap[extra]>=0",
            "translate-ap ; python_version >= '3.11'",
            "numpy",
        ),
    )
    assert runtime_deps(cand, "translate-ap") == ("openai>=1.0", "numpy")
    assert runtime_deps(cand, "translate_ap") == ("openai>=1.0", "numpy")


def test_runtime_deps_no_block_candidate():
    cand = Candidate(name="c", path=Path("c.py"), source="")
    assert runtime_deps(cand, "translate-ap") == ()


def test_runtime_deps_keeps_similar_names():
    cand = Candidate(
        name="c",
        path=Path("c.py"),
        source="",
        dependencies=("translate-ap-extras", "translate"),
    )
    assert runtime_deps(cand, "translate-ap") == ("translate-ap-extras", "translate")


# ------------------------------------------------------------- CRLF sources


CRLF_SOURCE = (
    "# /// script\r\n"
    '# dependencies = ["requests"]\r\n'
    "# ///\r\n"
    "def predict(text):\r\n"
    "    return text\r\n"
)


def test_parse_pep723_accepts_crlf_line_endings():
    assert parse_pep723(CRLF_SOURCE) == {"dependencies": ["requests"]}


def test_new_candidate_crlf_source_agrees_with_reload(tmp_path):
    ws = make_ws(tmp_path)
    created = new_candidate(ws, source=CRLF_SOURCE)
    assert created.dependencies == ("requests",)
    reloaded = load_candidate(ws, created.name)
    assert reloaded.dependencies == created.dependencies
    assert reloaded.source == created.source
