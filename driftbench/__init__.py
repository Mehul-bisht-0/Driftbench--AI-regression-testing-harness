"""driftbench - regression and flakiness testing for LLM agents.

Importing the package registers the mock tool surface, so ``tools.REGISTRY`` is
populated for anything that imports ``driftbench``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import tools_impl as _tools_impl  # noqa: F401  (import registers tools)
from . import tools  # noqa: F401

__all__ = ["__version__", "tools"]
