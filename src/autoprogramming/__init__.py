"""AutoProgramming — define your inputs and outputs; a coding agent finds the best implementation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .budget import Budget
from .errors import (
    AutoProgrammingError,
    BootstrapModeError,
    BudgetError,
    BudgetExceededError,
    CandidateError,
    DataDisciplineError,
    FinalizedError,
    MemorizationWarning,
    MetricChangedError,
    MetricNotApprovedError,
    NotOptimizedError,
    RunnerError,
    SchemaError,
    ValReliabilityWarning,
    WorkspaceError,
)
from .harness import attach
from .host_backend import HostOrchestratorBackend
from .objectives import MetricSuite, SelectionPolicy, approve_suite
from .pi_backend import PiOrchestratorBackend
from .portfolio import ApproachTier, AvenueSpec, PortfolioPolicy
from .program import PreparedRun, Program, program
from .research import SearchReport, SearchResult, WebResearchError
from .resources import (
    DataPolicy,
    RemoteCompute,
    ResourceError,
    Resources,
    RuntimeResources,
    SearchResources,
)

try:
    __version__ = _dist_version("autoprogramming")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "program",
    "Program",
    "PreparedRun",
    "Budget",
    "attach",
    "Resources",
    "SearchResources",
    "RemoteCompute",
    "RuntimeResources",
    "DataPolicy",
    "ResourceError",
    "MetricSuite",
    "SelectionPolicy",
    "approve_suite",
    "ApproachTier",
    "AvenueSpec",
    "PortfolioPolicy",
    "PiOrchestratorBackend",
    "HostOrchestratorBackend",
    "SearchReport",
    "SearchResult",
    "WebResearchError",
    "AutoProgrammingError",
    "SchemaError",
    "DataDisciplineError",
    "MetricNotApprovedError",
    "MetricChangedError",
    "BudgetError",
    "BudgetExceededError",
    "BootstrapModeError",
    "NotOptimizedError",
    "FinalizedError",
    "WorkspaceError",
    "CandidateError",
    "RunnerError",
    "ValReliabilityWarning",
    "MemorizationWarning",
    "__version__",
]
