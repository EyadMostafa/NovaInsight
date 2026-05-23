"""Unit tests for the module registry and topological sort."""
from __future__ import annotations

import pytest

import sleuth.modules  # noqa: F401 — triggers @register_module decorators
from sleuth.exceptions import OrchestratorError
from sleuth.modules.registry import (
    ModuleSpec,
    _REGISTRY,
    get_registry,
    register_module,
    topological_order,
)
from sleuth.schemas.analysis_report import Operator


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------

class TestGetRegistry:
    def test_all_expected_operators_registered(self):
        registry = get_registry()
        expected = {
            Operator.PROFILER,
            Operator.TARGET,
            Operator.STATS,
            Operator.DIM_REDUCTION,
            Operator.VIZ,
            Operator.LLM,
        }
        assert expected.issubset(set(registry.keys()))

    def test_each_spec_has_cls_and_dependencies(self):
        for op, spec in get_registry().items():
            assert isinstance(spec, ModuleSpec)
            assert spec.cls is not None
            assert isinstance(spec.dependencies, list)
            assert callable(spec.is_completed)


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------

class TestTopologicalOrder:
    def test_returns_all_registered_operators(self):
        order = topological_order()
        assert set(order) == set(get_registry().keys())

    def test_dependency_always_before_dependent(self):
        order = topological_order()
        registry = get_registry()
        for op in order:
            op_idx = order.index(op)
            for dep in registry[op].dependencies:
                dep_idx = order.index(dep)
                assert dep_idx < op_idx, (
                    f"Dependency {dep} appears after {op} in topological order"
                )

    def test_profiler_is_first(self):
        order = topological_order()
        assert order[0] == Operator.PROFILER

    def test_llm_after_stats(self):
        order = topological_order()
        assert order.index(Operator.LLM) > order.index(Operator.STATS)

    def test_viz_after_stats_and_dim_reduction(self):
        order = topological_order()
        assert order.index(Operator.VIZ) > order.index(Operator.STATS)
        assert order.index(Operator.VIZ) > order.index(Operator.DIM_REDUCTION)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_cycle_raises_orchestrator_error(self, monkeypatch):
        from sleuth.modules.base_module import BaseModule
        from sleuth.schemas.analysis_report import AnalysisReport
        import pandas as pd

        # Build a fake registry with a cycle: A → B → A
        fake_registry: dict = {}

        class FakeModuleA(BaseModule):
            def run(self, df, report): return report

        class FakeModuleB(BaseModule):
            def run(self, df, report): return report

        fake_registry[Operator.PROFILER] = ModuleSpec(
            operator=Operator.PROFILER,
            cls=FakeModuleA,
            dependencies=[Operator.LLM],
            is_completed=lambda r: False,
        )
        fake_registry[Operator.LLM] = ModuleSpec(
            operator=Operator.LLM,
            cls=FakeModuleB,
            dependencies=[Operator.PROFILER],
            is_completed=lambda r: False,
        )

        import sleuth.modules.registry as reg_module
        monkeypatch.setattr(reg_module, "_REGISTRY", fake_registry)

        with pytest.raises(OrchestratorError, match="[Cc]ycle"):
            topological_order()

    def test_unknown_dependency_raises_orchestrator_error(self, monkeypatch):
        from sleuth.modules.base_module import BaseModule

        fake_registry: dict = {}

        class FakeModule(BaseModule):
            def run(self, df, report): return report

        # PROFILER declares a dependency on TARGET which is absent from this fake registry,
        # so topological_order() must raise OrchestratorError for the unknown dep.
        fake_registry[Operator.PROFILER] = ModuleSpec(
            operator=Operator.PROFILER,
            cls=FakeModule,
            dependencies=[Operator.TARGET],
            is_completed=lambda r: False,
        )

        import sleuth.modules.registry as reg_module
        monkeypatch.setattr(reg_module, "_REGISTRY", fake_registry)

        with pytest.raises(OrchestratorError):
            topological_order()


# ---------------------------------------------------------------------------
# register_module decorator
# ---------------------------------------------------------------------------

class TestRegisterModuleDecorator:
    def test_decorator_returns_class_unchanged(self):
        """register_module must return the decorated class unmodified."""
        import sleuth.modules.registry as reg_module
        from sleuth.modules.base_module import BaseModule

        # Temporarily borrow the PROFILER slot; save and restore so no other
        # test is affected by the transient overwrite.
        saved = reg_module._REGISTRY.get(Operator.PROFILER)
        try:
            @register_module(
                operator=Operator.PROFILER,
                dependencies=[],
                is_completed=lambda r: False,
            )
            class DummyModule(BaseModule):
                def run(self, df, report): return report

            assert DummyModule.__name__ == "DummyModule"
            assert reg_module._REGISTRY[Operator.PROFILER].cls is DummyModule
        finally:
            if saved is not None:
                reg_module._REGISTRY[Operator.PROFILER] = saved
            else:
                reg_module._REGISTRY.pop(Operator.PROFILER, None)
