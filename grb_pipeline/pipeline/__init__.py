"""Pipeline orchestration for GRB analysis."""
from .orchestrator import PipelineOrchestrator
from .stages import (
    PipelineStage,
    DataAcquisitionStage,
    TemporalAnalysisStage,
    SpectralAnalysisStage,
    AfterglowAnalysisStage,
    ClassificationStage,
    AIAnalysisStage,
)
from .runner import main

__all__ = [
    "PipelineOrchestrator",
    "PipelineStage",
    "DataAcquisitionStage",
    "TemporalAnalysisStage",
    "SpectralAnalysisStage",
    "AfterglowAnalysisStage",
    "ClassificationStage",
    "AIAnalysisStage",
    "main",
]

__version__ = "1.0.0"
