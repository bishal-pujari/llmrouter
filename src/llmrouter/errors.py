"""Typed errors shared by configuration, storage, and routing layers."""


class LLMRouterError(Exception):
    """Base class for errors callers may handle explicitly."""


class ConfigError(LLMRouterError):
    """The router configuration is missing or invalid."""


class NoModelAvailable(LLMRouterError):
    """No eligible model has quota and concurrency capacity for a request."""


class BudgetExceeded(LLMRouterError):
    """A monthly or per-call cost limit would be exceeded."""

    def __init__(self, message: str, *, budget_group: str | None = None) -> None:
        super().__init__(message)
        self.budget_group = budget_group


class OutputValidationError(LLMRouterError):
    """Structured model output could not be parsed or validated."""

    def __init__(self, message: str, problems: list[str], attempts: list[object] | None = None) -> None:
        super().__init__(message)
        self.problems = problems
        self.attempts = attempts or []
