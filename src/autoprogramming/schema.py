"""Program schemas.

A schema is extracted once from the decorated function and is immutable
afterwards. Types are subclasses of builtins; docstrings become descriptions
the agent uses. Output names come from their types, which is why two outputs
of the same type is a schema error.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass

from .errors import SchemaError

# bool must precede int (bool subclasses int); every base here is JSON-adjacent.
_BUILTIN_BASES: tuple[type, ...] = (
    bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset,
)


def builtin_base(tp: type) -> type:
    """The builtin ancestor of a type — the wire format for serialization."""
    for base in _BUILTIN_BASES:
        if issubclass(tp, base):
            return base
    raise SchemaError(
        f"Type {tp.__name__!r} must subclass a builtin (str, int, float, ...) "
        f"so its values survive CSV/JSON serialization and remain usable "
        f"everywhere the builtin is."
    )


def _own_doc(tp: type) -> str:
    """A type's own docstring, never the inherited builtin one.

    Builtins carry their C docstring in their own ``__dict__`` (there is
    nothing to inherit from), so annotating an input plainly as ``str`` must
    not turn str's multi-line implementation doc into a field description —
    descriptions are user-written docstrings only.
    """
    if tp.__module__ == "builtins":
        return ""
    doc = tp.__dict__.get("__doc__")
    return inspect.cleandoc(doc) if doc else ""


def _doc_literal(text: str) -> str:
    """A Python literal whose value is exactly ``text``, for generated docstrings.

    Triple quotes keep the generated module readable, but content that a
    triple-quoted literal would break on (embedded triple quotes, a trailing
    quote) or silently reinterpret (any backslash, e.g. regex ``\\d``) falls
    back to ``repr()``, which round-trips every string.
    """
    if '"""' not in text and "\\" not in text and not text.endswith('"'):
        return f'"""{text}"""'
    return repr(text)


@dataclass(frozen=True)
class Field:
    """One input or output of a program.

    For inputs, ``name`` is the parameter name; for outputs it is the type
    name (guaranteed unique by the schema rule). ``type`` is the concrete
    class in this session; ``base`` is its builtin ancestor and the wire
    format.
    """

    name: str
    type: type
    base: type
    description: str = ""

    @property
    def type_name(self) -> str:
        return self.type.__name__

    def to_wire(self, value: object) -> object:
        """Downcast to the builtin base for JSON/CSV transport."""
        return self.base(value)

    def from_wire(self, value: object) -> object:
        """Lift a wire value back into the schema type (French, Confidence, ...)."""
        if isinstance(value, self.type):
            return value
        if self.base is not self.type and not isinstance(value, self.base):
            value = self.base(value)  # e.g. "0.9" -> 0.9 before Confidence(0.9)
        return self.type(value)

    def coerce_base(self, value: object) -> object:
        """Coerce a raw (usually CSV string) value to the builtin base."""
        if self.base is bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes")
        if isinstance(value, self.base):
            return value
        return self.base(value)


@dataclass(frozen=True)
class Schema:
    """The immutable contract of a program: name, doc, inputs, outputs."""

    name: str
    doc: str
    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]

    # ---------------------------------------------------------------- build

    @classmethod
    def from_function(cls, fn) -> "Schema":
        name = fn.__name__
        doc = inspect.getdoc(fn) or ""
        try:
            hints = typing.get_type_hints(fn)
        except Exception as exc:  # unresolvable forward refs, etc.
            raise SchemaError(f"Could not resolve type annotations on {name}(): {exc}") from exc

        sig = inspect.signature(fn)
        if not sig.parameters:
            raise SchemaError(f"{name}() has no inputs; a program needs at least one.")

        inputs: list[Field] = []
        for pname, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise SchemaError(
                    f"{name}() uses *{pname} — programs take a fixed set of named "
                    f"inputs so data columns can map onto them."
                )
            if pname not in hints:
                raise SchemaError(f"Input {pname!r} of {name}() is missing a type annotation.")
            if param.default is not param.empty:
                raise SchemaError(
                    f"Input {pname!r} of {name}() has a default value; program inputs "
                    f"are required — every data row must provide every input."
                )
            tp = hints[pname]
            if not isinstance(tp, type):
                raise SchemaError(
                    f"Input {pname!r} of {name}() is annotated {tp!r}; annotations "
                    f"must be plain classes (builtins or subclasses of builtins), "
                    f"not generics or unions."
                )
            inputs.append(Field(name=pname, type=tp, base=builtin_base(tp), description=_own_doc(tp)))

        if "return" not in hints:
            raise SchemaError(
                f"{name}() is missing a return annotation; outputs are declared "
                f"as the return type (a type, or a tuple of distinct types)."
            )
        ret = hints["return"]
        out_types: list[type]
        if typing.get_origin(ret) is tuple:
            args = typing.get_args(ret)
            if not args or Ellipsis in args:
                raise SchemaError(
                    f"{name}() returns {ret!r}; variable-length tuples are not a "
                    f"schema — list each output type explicitly, e.g. tuple[Answer, Confidence]."
                )
            out_types = list(args)
        elif isinstance(ret, type):
            out_types = [ret]
        else:
            raise SchemaError(
                f"{name}() returns {ret!r}; the return annotation must be a plain "
                f"class or a tuple of plain classes (no generics, unions, or Optional)."
            )

        outputs: list[Field] = []
        seen: set[str] = set()
        for tp in out_types:
            if not isinstance(tp, type):
                raise SchemaError(f"Output annotation {tp!r} of {name}() is not a class.")
            if tp.__name__ in seen:
                raise SchemaError(
                    f"{name}() declares two outputs of type {tp.__name__!r}; output "
                    f"names come from their types, so each output needs a distinct type."
                )
            seen.add(tp.__name__)
            outputs.append(Field(name=tp.__name__, type=tp, base=builtin_base(tp), description=_own_doc(tp)))

        return cls(name=name, doc=doc, inputs=tuple(inputs), outputs=tuple(outputs))

    @classmethod
    def from_object(cls, obj) -> "Schema":
        """Schema of a decorated Program, or of a plain annotated function."""
        schema = getattr(obj, "schema", None)
        if isinstance(schema, Schema):
            return schema
        return cls.from_function(obj)

    # ------------------------------------------------------------- introspect

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.outputs)

    @property
    def expected_columns(self) -> tuple[str, ...]:
        """Data columns a training row must provide: inputs then outputs.

        Input keys are parameter names; output keys are output type names —
        they live in separate namespaces on the wire, so a clash between an
        input name and an output name is still a schema error here to keep
        CSV columns unambiguous.
        """
        cols = self.input_names + self.output_names
        dupes = {c for c in cols if cols.count(c) > 1}
        if dupes:
            raise SchemaError(
                f"Input and output names collide on {sorted(dupes)!r}; rename the "
                f"parameter or the output type so data columns are unambiguous."
            )
        return cols

    def describe(self) -> str:
        """Human/agent-readable summary (what prg.schema prints)."""
        lines = [f"{self.name}: {self.doc}".rstrip().rstrip(":")]
        lines.append("inputs:")
        for f in self.inputs:
            d = f" — {f.description}" if f.description else ""
            lines.append(f"  {f.name}: {f.type_name}{d}")
        lines.append("outputs:")
        for f in self.outputs:
            d = f" — {f.description}" if f.description else ""
            lines.append(f"  {f.name} ({f.base.__name__}){d}")
        return "\n".join(lines)

    # ------------------------------------------------------------ conversions

    def outputs_to_dict(self, value: object) -> dict[str, object]:
        """A predict() return value → {output_name: value}.

        Single-output programs return the bare value; multi-output programs
        return a tuple in schema order. A dict already keyed by output names
        passes through.
        """
        if isinstance(value, dict):
            missing = [n for n in self.output_names if n not in value]
            if missing:
                raise SchemaError(f"Output dict is missing {missing!r}.")
            return {n: value[n] for n in self.output_names}
        if len(self.outputs) == 1:
            return {self.outputs[0].name: value}
        if not isinstance(value, (tuple, list)) or len(value) != len(self.outputs):
            raise SchemaError(
                f"{self.name} declares {len(self.outputs)} outputs "
                f"({', '.join(self.output_names)}); predict() must return a tuple "
                f"of that length, got {value!r}."
            )
        return {f.name: v for f, v in zip(self.outputs, value)}

    def dict_to_outputs(self, d: dict[str, object]) -> object:
        """{output_name: wire value} → typed value (or tuple, in schema order)."""
        typed = [f.from_wire(d[f.name]) for f in self.outputs]
        return typed[0] if len(typed) == 1 else tuple(typed)

    def coerce_expected(self, row: dict[str, object]) -> dict[str, object]:
        """Base-coerce a data row's expected outputs (CSV gives strings)."""
        return {f.name: f.coerce_base(row[f.name]) for f in self.outputs}

    def coerce_inputs(self, row: dict[str, object]) -> dict[str, object]:
        """Base-coerce a data row's inputs for the wire."""
        return {f.name: f.coerce_base(row[f.name]) for f in self.inputs}

    # -------------------------------------------------------------- rendering

    def render_module(self) -> str:
        """Source of the workspace's generated schema.py.

        Defines the custom types plainly and re-declares the program stub.
        The shipped package must not depend on the optimizer, so the
        @program decoration is guarded: without autoprogramming installed the
        stub stays a plain function and the types still import fine.
        """
        chunks = [
            f'"""Schema for `{self.name}` — generated by autoprogramming. '
            f'Immutable during optimization."""\n'
        ]
        rendered: set[str] = set()
        for f in (*self.inputs, *self.outputs):
            if f.type in _BUILTIN_BASES or f.type_name in rendered:
                continue
            rendered.add(f.type_name)
            body = f"    {_doc_literal(f.description)}" if f.description else "    pass"
            chunks.append(f"class {f.type_name}({f.base.__name__}):\n{body}\n")

        chunks.append(
            "try:\n"
            "    import autoprogramming as _ap\n"
            "    _program = _ap.program\n"
            "except ImportError:  # the shipped package must not require the optimizer\n"
            "    def _program(fn):\n"
            "        return fn\n"
        )

        params = ", ".join(f"{f.name}: {f.type_name}" for f in self.inputs)
        if len(self.outputs) == 1:
            ret = self.outputs[0].type_name
        else:
            ret = f"tuple[{', '.join(f.type_name for f in self.outputs)}]"
        doc = f"    {_doc_literal(self.doc)}\n" if self.doc else ""
        chunks.append(f"@_program\ndef {self.name}({params}) -> {ret}:\n{doc}    ...\n")
        return "\n\n".join(chunks)
