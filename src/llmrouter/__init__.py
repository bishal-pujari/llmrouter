"""Local-first model routing library."""

from llmrouter.router import BatchResult, CallResult, Router
from llmrouter.errors import BudgetExceeded

__version__ = "0.1.0"

__all__ = ["BatchResult", "BudgetExceeded", "CallResult", "Router", "__version__"]
