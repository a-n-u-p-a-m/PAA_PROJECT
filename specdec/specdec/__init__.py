"""Energy profiling of speculative decoding (CLEAR adaptation).

Phase 1-2 deliverable: a correct, instrumented baseline draft-verify pipeline.
Phases 3-5 (CLEAR activation store, amplification, K* analysis) attach to the
phase boundaries exposed by ``specdec.meters.PhaseMeter``.
"""

from .configs import (
    DEFAULT_SWEEP,
    MODEL_PAIRS,
    PAPER_STRATEGIES,
    DraftStrategy,
    ModelPair,
    feasibility_table,
    parse_strategy,
)
from .engine import (
    GenerationResult,
    RoundRecord,
    autoregressive_generate,
    speculative_generate,
)
from .memory import ARCHS, ArchSpec, Budget, budget
from .meters import EnergyProbe, PhaseMeter

__all__ = [
    "GenerationResult",
    "RoundRecord",
    "autoregressive_generate",
    "speculative_generate",
    "EnergyProbe",
    "PhaseMeter",
    "MODEL_PAIRS",
    "ModelPair",
    "DraftStrategy",
    "parse_strategy",
    "PAPER_STRATEGIES",
    "DEFAULT_SWEEP",
    "feasibility_table",
    "ARCHS",
    "ArchSpec",
    "Budget",
    "budget",
]
