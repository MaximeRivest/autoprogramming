"""Exception types.

Every refusal in autoprogramming explains itself: what was refused, why the
rule exists, and what to do instead. Raise these with full messages — a bare
``DataDisciplineError()`` is a bug.
"""


class AutoProgrammingError(Exception):
    """Base class for every error this library raises on purpose."""


class SchemaError(AutoProgrammingError):
    """The @program definition or the data does not satisfy the schema rules.

    Examples: missing annotations, two outputs of the same type (output names
    come from their types), data columns that don't cover the schema.
    """


class DataDisciplineError(AutoProgrammingError):
    """A harness-enforced data rule was violated.

    Examples: trace inspection on val/test rows, eval on test, per-instance
    scores on val, re-splitting data that was already split, optimizing on
    unreviewed logs.
    """


class MetricNotApprovedError(AutoProgrammingError):
    """metric.py exists but has not been signed off by the user.

    The entire search optimizes whatever metric.py says, so a wrong metric
    produces a confidently-scored wrong program. Demonstrate the metric on
    real examples and record approval before any scoring happens.
    """


class MetricChangedError(AutoProgrammingError):
    """metric.py changed after scores were recorded.

    Scores under different metrics are never comparable; the harness archives
    and clears scores.json, and the new metric needs a fresh approval.
    """


class BudgetError(AutoProgrammingError):
    """The budget specification itself is invalid (there is no default)."""


class BudgetExceededError(AutoProgrammingError):
    """A budget limit was hit; optimization stops at the first limit."""


class BootstrapModeError(AutoProgrammingError):
    """An operation was refused because the dataset is too small.

    Below ~30 examples the harness builds and compares baseline candidates but
    refuses fine-grained mutation loops — a 0.92-vs-0.86 difference on 5 val
    rows is one row of noise. Offer to generate synthetic examples instead.
    """


class NotOptimizedError(AutoProgrammingError):
    """The program has no active candidate yet — run optimize() first."""


class FinalizedError(AutoProgrammingError):
    """finalize() already ran; test is evaluated exactly once."""


class WorkspaceError(AutoProgrammingError):
    """The workspace directory is missing, malformed, or inconsistent."""


class CandidateError(AutoProgrammingError):
    """A candidate file is missing or its PEP 723 metadata is malformed."""


class RunnerError(AutoProgrammingError):
    """The execution harness itself failed (uv missing, driver crashed).

    A candidate raising during predict() is NOT a RunnerError — that comes
    back as a failed RunResult and scores 0.
    """


class ValReliabilityWarning(UserWarning):
    """Val has absorbed many selection decisions relative to its size;
    its scores are losing meaning and will not be reported as final."""


class MemorizationWarning(UserWarning):
    """A candidate looks like a memorizer (train >> val, or verbatim
    training outputs in its source); it is excluded from selection."""
