# Import all pipeline module files to trigger @register_module decorators.
# The order here does not matter — topological_order() derives execution order
# from the declared dependencies at runtime.
from sleuth.modules import (  # noqa: F401
    data_profiler,
    target_detector,
    statistical_analyzer,
    dimensionality_reducer,
    visualizer,
    llm_summarizer,
)
