"""
registry.py

Self-registration system for pipeline modules.

Each module decorates its class with @register_module, supplying its Operator
identity, dependency list, and an is_completed predicate. The orchestrator
queries this registry instead of maintaining hardcoded lookup tables.

Adding a new module only requires:
  1. novainsight/modules/my_module.py  — class + @register_module decorator
  2. Operator enum value in schemas/analysis_report.py
  3. AnalysisReport field in schemas/analysis_report.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type, TYPE_CHECKING

from novainsight.schemas.analysis_report import AnalysisReport, Operator
from novainsight.exceptions import OrchestratorError

if TYPE_CHECKING:
    from novainsight.modules.base_module import BaseModule

_REGISTRY: dict[Operator, ModuleSpec] = {}


@dataclass
class ModuleSpec:
    operator: Operator
    cls: Type[BaseModule]
    dependencies: list[Operator]
    is_completed: Callable[[AnalysisReport], bool]


def register_module(
    operator: Operator,
    dependencies: list[Operator],
    is_completed: Callable[[AnalysisReport], bool],
) -> Callable[[Type[BaseModule]], Type[BaseModule]]:
    """
    Class decorator that registers a pipeline module with its metadata.

    Usage:
        @register_module(
            operator=Operator.PROFILER,
            dependencies=[],
            is_completed=lambda report: bool(report.profile.column_details),
        )
        class DataProfiler(BaseModule):
            ...
    """
    def decorator(cls: Type[BaseModule]) -> Type[BaseModule]:
        _REGISTRY[operator] = ModuleSpec(
            operator=operator,
            cls=cls,
            dependencies=dependencies,
            is_completed=is_completed,
        )
        return cls
    return decorator


def get_registry() -> dict[Operator, ModuleSpec]:
    return _REGISTRY


def topological_order() -> list[Operator]:
    """
    Returns all registered operators in a valid execution order via Kahn's algorithm.
    Raises OrchestratorError if an unknown dependency or cycle is detected.
    """
    in_degree: dict[Operator, int] = {op: 0 for op in _REGISTRY}
    dependents: dict[Operator, list[Operator]] = {op: [] for op in _REGISTRY}

    for op, spec in _REGISTRY.items():
        for dep in spec.dependencies:
            if dep not in _REGISTRY:
                raise OrchestratorError(
                    f"Module '{op}' declares dependency on unregistered module '{dep}'"
                )
            dependents[dep].append(op)
            in_degree[op] += 1

    queue: list[Operator] = [op for op, deg in in_degree.items() if deg == 0]
    order: list[Operator] = []

    while queue:
        op = queue.pop(0)
        order.append(op)
        for dependent in sorted(dependents[op], key=lambda o: o.value):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(_REGISTRY):
        raise OrchestratorError("Cycle detected in module dependency graph")

    return order
