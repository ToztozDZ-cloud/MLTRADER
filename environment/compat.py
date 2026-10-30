from __future__ import annotations

import warnings


def suppress_dependency_warnings() -> None:
    """Hide third-party deprecation warnings that this project cannot migrate itself."""
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.legacy is deprecated.*",
        category=DeprecationWarning,
    )
