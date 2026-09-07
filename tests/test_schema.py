"""Unit tests for autoprogramming.schema (Field, Schema, builtin_base)."""

from __future__ import annotations

import dataclasses

import pytest

from autoprogramming.errors import SchemaError
from autoprogramming.schema import Field, Schema, builtin_base


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


class Answer(str):
    """Direct answer to the question, one sentence."""


class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""


class Bare(str):
    pass


class Count(int):
    """How many times it happened."""


class Weird:
    """Not a subclass of any builtin."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""


# ---------------------------------------------------------------- building


def test_single_output_schema():
    schema = Schema.from_function(shout)
    assert schema.name == "shout"
    assert schema.doc == "Uppercase the text."
    assert schema.input_names == ("text",)
    assert schema.inputs[0].type is str
    assert schema.inputs[0].base is str
    assert schema.output_names == ("Loud",)
    assert schema.outputs[0].type is Loud
    assert schema.outputs[0].base is str
    assert schema.outputs[0].description == "The input, uppercased, with an exclamation mark."


def test_multi_output_schema():
    schema = Schema.from_function(qa)
    assert schema.input_names == ("question",)
    assert schema.output_names == ("Answer", "Confidence")
    assert [f.base for f in schema.outputs] == [str, float]
    assert schema.outputs[1].description.startswith("Calibrated probability")


def test_duplicate_output_types_refused():
    def f(x: str) -> tuple[Answer, Answer]:
        """Two of the same."""

    with pytest.raises(SchemaError, match="distinct"):
        Schema.from_function(f)


def test_missing_input_annotation_refused():
    def f(x) -> Loud:
        """No annotation."""

    with pytest.raises(SchemaError, match="missing a type annotation"):
        Schema.from_function(f)


def test_missing_return_annotation_refused():
    def f(x: str):
        """No return."""

    with pytest.raises(SchemaError, match="return annotation"):
        Schema.from_function(f)


def test_var_positional_refused():
    def f(*args: str) -> Loud:
        """Star args."""

    with pytest.raises(SchemaError, match=r"\*args"):
        Schema.from_function(f)


def test_var_keyword_refused():
    def f(x: str, **kwargs: str) -> Loud:
        """Star kwargs."""

    with pytest.raises(SchemaError, match=r"\*kwargs"):
        Schema.from_function(f)


def test_default_value_refused():
    def f(x: str = "hi") -> Loud:
        """Default."""

    with pytest.raises(SchemaError, match="default"):
        Schema.from_function(f)


def test_no_inputs_refused():
    def f() -> Loud:
        """Nullary."""

    with pytest.raises(SchemaError, match="at least one"):
        Schema.from_function(f)


def test_generic_input_annotation_refused():
    def f(x: list[str]) -> Loud:
        """Generic input."""

    with pytest.raises(SchemaError, match="plain classes"):
        Schema.from_function(f)


def test_union_return_refused():
    def f(x: str) -> str | None:
        """Union return."""

    with pytest.raises(SchemaError, match="plain class"):
        Schema.from_function(f)


def test_tuple_ellipsis_return_refused():
    def f(x: str) -> tuple[Loud, ...]:
        """Variable-length tuple."""

    with pytest.raises(SchemaError, match="explicitly"):
        Schema.from_function(f)


def test_empty_tuple_return_refused():
    def f(x: str) -> tuple:
        """Empty tuple return."""

    f.__annotations__["return"] = tuple[()]
    with pytest.raises(SchemaError):
        Schema.from_function(f)


def test_non_builtin_output_refused():
    def f(x: str) -> Weird:
        """Bad output."""

    with pytest.raises(SchemaError, match="subclass a builtin"):
        Schema.from_function(f)


# ------------------------------------------------------------ builtin_base


def test_builtin_base_bool_wins_over_int():
    assert builtin_base(bool) is bool
    assert builtin_base(Count) is int
    assert builtin_base(int) is int


def test_builtin_base_subclasses():
    assert builtin_base(Loud) is str
    assert builtin_base(Confidence) is float
    assert builtin_base(dict) is dict


def test_builtin_base_object_refused():
    with pytest.raises(SchemaError, match="subclass a builtin"):
        builtin_base(object)


# ------------------------------------------------------------------ fields


def test_field_wire_roundtrip():
    field = Field(name="Loud", type=Loud, base=str)
    wired = field.to_wire(Loud("HI!"))
    assert type(wired) is str
    lifted = field.from_wire("HI!")
    assert type(lifted) is Loud and lifted == "HI!"


def test_field_from_wire_string_to_float_subclass():
    field = Field(name="Confidence", type=Confidence, base=float)
    lifted = field.from_wire("0.9")
    assert type(lifted) is Confidence
    assert lifted == pytest.approx(0.9)


def test_field_coerce_base_bool():
    field = Field(name="bool", type=bool, base=bool)
    assert field.coerce_base("true") is True
    assert field.coerce_base("YES") is True
    assert field.coerce_base("1") is True
    assert field.coerce_base("0") is False
    assert field.coerce_base("no") is False
    assert field.coerce_base(True) is True
    assert field.coerce_base(False) is False


def test_bool_output_program_coerces_expected():
    def f(x: str) -> bool:
        """Truthiness."""

    schema = Schema.from_function(f)
    assert schema.outputs[0].base is bool
    assert schema.coerce_expected({"bool": "yes"}) == {"bool": True}
    assert schema.coerce_expected({"bool": "0"}) == {"bool": False}


# ------------------------------------------------------------- conversions


def test_outputs_to_dict_single_bare_value():
    schema = Schema.from_function(shout)
    assert schema.outputs_to_dict("HI!") == {"Loud": "HI!"}


def test_outputs_to_dict_multi_tuple_and_list():
    schema = Schema.from_function(qa)
    assert schema.outputs_to_dict(("yes", 0.9)) == {"Answer": "yes", "Confidence": 0.9}
    assert schema.outputs_to_dict(["yes", 0.9]) == {"Answer": "yes", "Confidence": 0.9}


def test_outputs_to_dict_dict_passthrough_and_order():
    schema = Schema.from_function(qa)
    out = schema.outputs_to_dict({"Confidence": 0.5, "Answer": "no", "extra": 1})
    assert list(out) == ["Answer", "Confidence"]
    assert out == {"Answer": "no", "Confidence": 0.5}


def test_outputs_to_dict_dict_missing_key_refused():
    schema = Schema.from_function(qa)
    with pytest.raises(SchemaError, match="missing"):
        schema.outputs_to_dict({"Answer": "no"})


def test_outputs_to_dict_wrong_arity_refused():
    schema = Schema.from_function(qa)
    with pytest.raises(SchemaError, match="tuple"):
        schema.outputs_to_dict(("only one",))
    with pytest.raises(SchemaError, match="tuple"):
        schema.outputs_to_dict("bare value")


def test_dict_to_outputs_single_typed():
    schema = Schema.from_function(shout)
    value = schema.dict_to_outputs({"Loud": "HI!"})
    assert type(value) is Loud and value == "HI!"


def test_dict_to_outputs_multi_tuple_in_schema_order():
    schema = Schema.from_function(qa)
    value = schema.dict_to_outputs({"Confidence": "0.25", "Answer": "yes"})
    assert isinstance(value, tuple) and len(value) == 2
    assert type(value[0]) is Answer and value[0] == "yes"
    assert type(value[1]) is Confidence and value[1] == pytest.approx(0.25)


def test_coerce_inputs_from_csv_strings():
    def f(n: int, text: str) -> Loud:
        """Mixed inputs."""

    schema = Schema.from_function(f)
    assert schema.coerce_inputs({"n": "7", "text": 42}) == {"n": 7, "text": "42"}


# ------------------------------------------------------------- introspection


def test_expected_columns_inputs_then_outputs():
    schema = Schema.from_function(qa)
    assert schema.expected_columns == ("question", "Answer", "Confidence")


def test_expected_columns_collision_refused():
    def paint(Loud: str) -> Loud:
        """Input named after the output type."""

    schema = Schema.from_function(paint)
    with pytest.raises(SchemaError, match="collide"):
        schema.expected_columns


def test_describe_mentions_everything():
    text = Schema.from_function(qa).describe()
    assert "qa" in text
    assert "question" in text
    assert "Answer" in text and "Confidence" in text
    assert "Direct answer to the question" in text
    assert "calibrated confidence" in text


def test_from_object_prefers_schema_attribute():
    schema = Schema.from_function(shout)

    class Holder:
        pass

    holder = Holder()
    holder.schema = schema
    assert Schema.from_object(holder) is schema
    assert Schema.from_object(shout).name == "shout"


def test_schema_is_frozen():
    schema = Schema.from_function(shout)
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.name = "other"


# --------------------------------------------------------------- rendering


def test_render_module_roundtrip_via_exec():
    schema = Schema.from_function(qa)
    src = schema.render_module()
    ns = {"__name__": "qa_ap_schema"}
    exec(compile(src, "schema.py", "exec"), ns)

    assert ns["Answer"].__doc__ == "Direct answer to the question, one sentence."
    assert issubclass(ns["Answer"], str)
    assert issubclass(ns["Confidence"], float)

    rebuilt = Schema.from_object(ns["qa"])
    assert rebuilt.name == "qa"
    assert rebuilt.doc == "Answer a factual question with a calibrated confidence."
    assert rebuilt.input_names == ("question",)
    assert rebuilt.output_names == ("Answer", "Confidence")
    assert [f.base for f in rebuilt.inputs] == [str]
    assert [f.base for f in rebuilt.outputs] == [str, float]
    assert rebuilt.outputs[0].description == "Direct answer to the question, one sentence."


def test_render_module_handles_docless_type():
    def f(x: str) -> Bare:
        """Docless output type."""

    src = Schema.from_function(f).render_module()
    ns = {"__name__": "bare_ap_schema"}
    exec(compile(src, "schema.py", "exec"), ns)
    assert issubclass(ns["Bare"], str)
    rebuilt = Schema.from_object(ns["f"])
    assert rebuilt.output_names == ("Bare",)
    assert rebuilt.outputs[0].description == ""


def test_render_module_does_not_require_the_optimizer():
    src = Schema.from_function(shout).render_module()
    assert "except ImportError" in src
    assert "Immutable" in src.splitlines()[0]


def test_builtin_annotations_have_empty_descriptions():
    schema = Schema.from_function(qa)
    assert schema.inputs[0].description == ""
    text = schema.describe()
    assert "question: str" in text
    assert "Create a new string object" not in text
    assert "str(object" not in text


def _dynamic_program(output_doc, program_doc=None):
    """A program whose output type carries an arbitrary (hostile) docstring."""
    out_type = type("Fancy", (str,), {"__doc__": output_doc})

    def prog(text):
        ...

    prog.__doc__ = program_doc
    prog.__annotations__ = {"text": str, "return": out_type}
    return prog


@pytest.mark.parametrize("nasty", [
    'Uppercased. Quote outputs like """this""" for emphasis.',
    'Ends with a quote: "',
    "Ends with a backslash \\",
    "Match \\d+ digits, join with \\n newline.",
    'Multi line\nwith "quotes" inside',
])
def test_render_module_survives_hostile_type_docstrings(nasty):
    schema = Schema.from_function(_dynamic_program(nasty))
    src = schema.render_module()
    ns = {"__name__": "prog_ap_schema"}
    exec(compile(src, "schema.py", "exec"), ns)
    assert ns["Fancy"].__doc__ == schema.outputs[0].description


def test_render_module_survives_hostile_program_docstring():
    doc = 'Say it with """triple quotes""" and a regex like \\d+.'
    schema = Schema.from_function(_dynamic_program("Plain.", program_doc=doc))
    src = schema.render_module()
    ns = {"__name__": "prog_ap_schema"}
    exec(compile(src, "schema.py", "exec"), ns)
    rebuilt = Schema.from_object(ns["prog"])
    assert rebuilt.doc == schema.doc
    assert "\\d+" in rebuilt.doc
