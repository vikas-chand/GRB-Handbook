"""Core components of GRB analysis pipeline.

Exports:
    Models: GRBEvent, Observation, LightCurveData, SpectralData, SpectralFit,
            AfterglowParams, GCNCircular, FluxMeasurement, AnalysisResult
    Database: GRBDatabase for persistent data storage
    Config: PipelineConfig for configuration management
    Constants: Mission, Instrument, GRBClass, SpectralModel enums and physical constants
"""

from .models import (
    GRBEvent,
    Observation,
    LightCurveData,
    SpectralData,
    SpectralFit,
    AfterglowParams,
    GCNCircular,
    FluxMeasurement,
    AnalysisResult,
)
from .database import GRBDatabase
from .config import PipelineConfig
from .constants import (
    Mission,
    Instrument,
    GRBClass,
    SpectralModel,
    FitQuality,
    WavelengthBand,
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT,
    ELECTRON_MASS,
    PROTON_MASS,
    HUBBLE_CONSTANT,
    OMEGA_MATTER,
    OMEGA_LAMBDA,
    KEV_TO_ERG,
    ERG_TO_JOULE,
    SWIFT_BAT_ENERGY_MIN,
    SWIFT_BAT_ENERGY_MAX,
    SWIFT_XRT_ENERGY_MIN,
    SWIFT_XRT_ENERGY_MAX,
    FERMI_GBM_ENERGY_MIN,
    FERMI_GBM_ENERGY_MAX,
    FERMI_LAT_ENERGY_MIN,
    FERMI_LAT_ENERGY_MAX,
)

__all__ = [
    "GRBEvent",
    "Observation",
    "LightCurveData",
    "SpectralData",
    "SpectralFit",
    "AfterglowParams",
    "GCNCircular",
    "FluxMeasurement",
    "AnalysisResult",
    "GRBDatabase",
    "PipelineConfig",
    "Mission",
    "Instrument",
    "GRBClass",
    "SpectralModel",
    "FitQuality",
    "WavelengthBand",
    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "HUBBLE_CONSTANT",
    "OMEGA_MATTER",
    "OMEGA_LAMBDA",
    "KEV_TO_ERG",
    "ERG_TO_JOULE",
    "SWIFT_BAT_ENERGY_MIN",
    "SWIFT_BAT_ENERGY_MAX",
    "SWIFT_XRT_ENERGY_MIN",
    "SWIFT_XRT_ENERGY_MAX",
    "FERMI_GBM_ENERGY_MIN",
    "FERMI_GBM_ENERGY_MAX",
    "FERMI_LAT_ENERGY_MIN",
    "FERMI_LAT_ENERGY_MAX",
]
