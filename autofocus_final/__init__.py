"""Research-grade autofocus for OpenFlexure-style microscopes."""
from .preprocessor import Preprocessor, PreprocResult
from .metrics import MetricBank, AdaptiveCombiner, ConfidenceEstimator, SCALE
from .coarse import CoarseSweepAutofocus, CoarseResult
from .fine import FineFocusController, FineState, FineDecision
from .stage_iface import BaseStage, NullStage, StageError, SerialStage, SERIAL_OK
from .system import AutofocusSystem, FrameRecord

__all__ = [
    "Preprocessor", "PreprocResult",
    "MetricBank", "AdaptiveCombiner", "ConfidenceEstimator", "SCALE",
    "CoarseSweepAutofocus", "CoarseResult",
    "FineFocusController", "FineState", "FineDecision",
    "BaseStage", "NullStage", "StageError", "SerialStage", "SERIAL_OK",
    "AutofocusSystem", "FrameRecord",
]
