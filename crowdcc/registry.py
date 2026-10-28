"""Corruption registry."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Iterable, Mapping

from crowdcc.base import (
    KNOWN_NEEDS,
    LEVELS,
    CorruptionSpec,
    GTImpact,
    LevelParams,
)

_REGISTRY: dict[str, CorruptionSpec] = {}

_FAMILIES_PACKAGE = "crowdcc.families"
_families_loaded = False


def register(
    *,
    name: str,
    family: str,
    gt_impact: GTImpact,
    needs: Iterable[str] = (),
    levels: Mapping[str, LevelParams],
    verify: Callable[..., dict[str, float]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a corruption and validate its declaration."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(
                f"corruption {name!r} is already registered by "
                f"{_REGISTRY[name].fn.__module__}"
            )
        if gt_impact not in ("preserving", "altering"):
            raise ValueError(
                f"{name}: gt_impact must be 'preserving' or 'altering', got {gt_impact!r}"
            )
        unknown_needs = set(needs) - KNOWN_NEEDS
        if unknown_needs:
            raise ValueError(f"{name}: unknown needs {sorted(unknown_needs)}")
        if not levels:
            raise ValueError(f"{name}: at least one severity level is required")
        unknown_levels = set(levels) - set(LEVELS)
        if unknown_levels:
            raise ValueError(
                f"{name}: unknown levels {sorted(unknown_levels)}; allowed: {list(LEVELS)}"
            )
        for level_name, params in levels.items():
            if not isinstance(params, LevelParams):
                raise TypeError(
                    f"{name}[{level_name}]: levels values must be LevelParams, "
                    f"got {type(params).__name__}"
                )
        if not (fn.__doc__ or "").strip():
            raise ValueError(f"{name}: a docstring is mandatory")

        _REGISTRY[name] = CorruptionSpec(
            name=name,
            family=family,
            gt_impact=gt_impact,
            needs=tuple(needs),
            levels=dict(levels),
            fn=fn,
            verify=verify,
        )
        return fn

    return decorator


def load_families() -> None:
    """Import family modules to populate the registry."""
    global _families_loaded
    if _families_loaded:
        return
    package = importlib.import_module(_FAMILIES_PACKAGE)
    module_names = sorted(
        m.name for m in pkgutil.iter_modules(package.__path__) if not m.ispkg
    )
    for module_name in module_names:
        importlib.import_module(f"{_FAMILIES_PACKAGE}.{module_name}")
    _families_loaded = True


def get(name: str) -> CorruptionSpec:
    """Return the named corruption."""
    load_families()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown corruption {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def all_specs() -> dict[str, CorruptionSpec]:
    """Return registered corruptions sorted by name."""
    load_families()
    return {k: _REGISTRY[k] for k in sorted(_REGISTRY)}


def names() -> list[str]:
    return sorted(all_specs())


def families() -> list[str]:
    return sorted({spec.family for spec in all_specs().values()})


def resolve(selector: str) -> list[CorruptionSpec]:
    """Resolve corruption names, families, and special selectors."""
    specs = all_specs()
    chosen: dict[str, CorruptionSpec] = {}
    for token in (t.strip() for t in selector.split(",") if t.strip()):
        if token == "all":
            chosen.update(specs)
        elif token == "altering":
            chosen.update({n: s for n, s in specs.items()
                           if s.gt_impact == "altering"})
        elif token in specs:
            chosen[token] = specs[token]
        else:
            in_family = {n: s for n, s in specs.items() if s.family == token}
            if not in_family:
                raise KeyError(
                    f"{token!r} is neither a corruption nor a family. "
                    f"Corruptions: {sorted(specs)}. Families: {families()}."
                )
            chosen.update(in_family)
    return [chosen[n] for n in sorted(chosen)]
