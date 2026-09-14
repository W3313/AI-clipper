"""The five workflow pipelines: ``clip``, ``story``, ``texts``, ``reddit``, ``split``.

Each submodule exposes a ``run(...)`` that drives the core modules end to end and
returns a :class:`~aiclipper.models.ProjectResult` (a list of them for ``clip``).
:mod:`aiclipper.pipelines.common` holds the helpers they share.

Submodules are resolved lazily through :pep:`562` ``__getattr__``, so importing
this package costs nothing: the CLI can list subcommands, and a caller can reach
for one workflow, without dragging in the transcriber, the overlay renderer and
every asset generator behind the other four.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["common", "clip", "story", "texts", "reddit", "split"]


def __getattr__(name: str) -> ModuleType:
    """Import ``aiclipper.pipelines.<name>`` on first attribute access."""
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
