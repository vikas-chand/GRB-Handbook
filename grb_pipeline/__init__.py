"""GRB Analysis Pipeline - Comprehensive Gamma-Ray Burst Analysis Framework

A complete pipeline for analyzing gamma-ray burst data from multiple instruments
and missions, including data acquisition, spectral fitting, light curve analysis,
and automated multi-wavelength correlation studies powered by Claude AI.

Modules:
    core: Core data models, database, and configuration
    utils: Utility functions for FITS handling and HEASoft integration
    analysis: Spectral fitting, light curve analysis, afterglow parameters
    catalog: GCN circular parsing and online catalog queries
    visualization: Light curves, spectra, and SED plots
"""

__version__ = "0.1.0"
__author__ = "Vikas"
__all__ = [
    "GRBEvent",
    "GRBDatabase",
    "PipelineConfig",
    "PHYSICAL_CONSTANTS",
    "COSMOLOGICAL_CONSTANTS",
]
