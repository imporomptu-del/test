"""PSF matched filtering and later synthetic-detection stages."""

from .matched_filter import (
    MatchedFilterConfig,
    MatchedFilterError,
    PsfMatchedFilter,
    build_kernel_bank,
    correlate_reference,
    integrated_gaussian_kernel,
)
from .types import MatchedFilterFrame, PsfKernelBank
from .synthetic_reference import (
    ReferenceShiftAndStack,
    ReferenceSyntheticWindow,
    SyntheticTrackingConfig,
    SyntheticTrackingError,
)
from .synthetic_types import SyntheticTrackWindow
from .synthetic_cuda import CudaShiftAndStack, CudaSyntheticTrackingError
from .candidates import (
    CandidateBatch,
    CandidateExtractionConfig,
    CandidateExtractionError,
    CandidateExtractor,
    CandidateRecord,
    CandidateRankingSurface,
    TrackPredictionHint,
)

__all__ = [
    "MatchedFilterConfig",
    "MatchedFilterError",
    "MatchedFilterFrame",
    "PsfKernelBank",
    "PsfMatchedFilter",
    "ReferenceShiftAndStack",
    "ReferenceSyntheticWindow",
    "CudaShiftAndStack",
    "CudaSyntheticTrackingError",
    "SyntheticTrackWindow",
    "SyntheticTrackingConfig",
    "SyntheticTrackingError",
    "CandidateBatch",
    "CandidateExtractionConfig",
    "CandidateExtractionError",
    "CandidateExtractor",
    "CandidateRecord",
    "CandidateRankingSurface",
    "TrackPredictionHint",
    "build_kernel_bank",
    "correlate_reference",
    "integrated_gaussian_kernel",
]
