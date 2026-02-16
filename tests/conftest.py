"""
Pytest configuration and fixtures for GRB Pipeline tests.

This module provides reusable test fixtures for all test modules,
including sample data objects, database connections, and configuration.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime
from scipy import stats

# GRB Pipeline imports
from grb_pipeline.core import (
    GRBEvent,
    TimingAnalysis,
    SpectralAnalysis,
    AftergowAnalysis,
    CorrelationAnalysis,
    Classification,
    GRBDatabase,
    PipelineConfig,
    PHYSICAL_CONSTANTS,
    COSMOLOGICAL_CONSTANTS,
)


@pytest.fixture
def sample_grb_event():
    """
    Create a realistic sample GRB event with typical properties.

    Returns:
        GRBEvent: A GRBEvent object with representative values for GRB230307A
    """
    grb = GRBEvent(
        name="GRB230307A",
        ra=165.4231,
        dec=-65.3246,
        discovery_time="2023-03-07T01:30:45.123",
        redshift=0.6655,
        instruments=["Swift/XRT", "Swift/UVOT", "Fermi/GBM"],
    )

    # Add simulated XRT light curve data
    t_start = 1.0  # seconds
    t_end = 1000.0  # seconds
    n_points = 50

    times = np.logspace(np.log10(t_start), np.log10(t_end), n_points)
    # Simple power-law decay: F(t) = F0 * (t/t0)^(-alpha)
    flux = 1e-11 * (times / 10.0) ** (-1.3)
    flux_error = 0.1 * flux  # 10% statistical error

    grb.xrt_data = pd.DataFrame({
        'time': times,
        'flux': flux,
        'flux_error': flux_error,
        'rate': flux * 1e10,  # Convert to count rate
    })

    # Add simulated optical/infrared data
    opt_times = np.array([1000, 2000, 5000, 10000])  # seconds
    opt_mag = np.array([18.5, 19.2, 20.1, 20.8])  # magnitudes
    opt_mag_error = np.array([0.1, 0.12, 0.15, 0.18])

    grb.optical_data = pd.DataFrame({
        'time': opt_times,
        'magnitude': opt_mag,
        'magnitude_error': opt_mag_error,
        'filter': ['Sloan_r', 'Sloan_r', 'Sloan_i', 'Sloan_i'],
    })

    return grb


@pytest.fixture
def sample_lightcurve():
    """
    Create a synthetic light curve with Gaussian pulse profile.

    Returns:
        tuple: (time_array, flux_array, flux_error_array) representing
               a simple burst followed by power-law decay
    """
    # Time array
    time = np.logspace(-2, 3, 200)  # 0.01 to 1000 seconds

    # Prompt emission component (Gaussian pulse)
    t0, sigma, amplitude = 5.0, 1.5, 1e-10
    prompt = amplitude * np.exp(-0.5 * ((time - t0) / sigma) ** 2)

    # Afterglow component (power-law decay)
    t_break = 100.0
    alpha1, alpha2 = 0.8, 1.5
    afterglow = np.where(
        time < t_break,
        2e-12 * (time / 10.0) ** (-alpha1),
        2e-12 * (t_break / 10.0) ** (-alpha1) * (time / t_break) ** (-alpha2),
    )

    # Combined flux
    flux = prompt + afterglow

    # Add realistic noise
    flux_error = np.sqrt(flux * 1e-12 + (0.05 * flux) ** 2)  # Poisson + 5% systematic
    flux_data = flux + np.random.normal(0, flux_error)
    flux_data = np.maximum(flux_data, 0)  # Ensure non-negative

    return time, flux_data, flux_error


@pytest.fixture
def sample_spectral_data():
    """
    Create synthetic spectral data following a Band function.

    The Band function is:
        N(E) = K * [(E/E_c)^α * exp(-E/E_c)], E < E_p(α+2)
        N(E) = K * [((α+2)E_p/E_c)^α * exp(-(α+2))] * (E/E_p)^β, E ≥ E_p(α+2)

    Returns:
        tuple: (energy_array, flux_array, flux_error_array)
    """
    # Energy array (logarithmic)
    energy = np.logspace(0, 3, 100)  # 1 to 1000 keV

    # Band function parameters
    epeak = 200.0  # keV
    alpha = -0.8  # low-energy photon index
    beta = -2.3  # high-energy photon index
    norm = 1e-2  # normalization

    # Calculate Band function
    e_c = epeak / (2 + alpha)  # characteristic energy
    flux = np.zeros_like(energy, dtype=float)

    # Low-energy component (E < E_p(α+2))
    ebreak = epeak * (alpha + 2)
    low_energy = energy < ebreak
    flux[low_energy] = norm * (energy[low_energy] / e_c) ** alpha * \
                       np.exp(-energy[low_energy] / e_c)

    # High-energy component (E ≥ E_p(α+2))
    high_energy = ~low_energy
    flux[high_energy] = norm * ((alpha + 2) * epeak / e_c) ** alpha * \
                        np.exp(-(alpha + 2)) * (energy[high_energy] / epeak) ** beta

    # Add realistic noise
    flux_error = np.sqrt(flux * 0.01 + (0.1 * flux) ** 2)  # Poisson + 10% systematic
    flux_data = flux + np.random.normal(0, flux_error)
    flux_data = np.maximum(flux_data, 0)

    return energy, flux_data, flux_error


@pytest.fixture
def temp_database():
    """
    Create a temporary SQLite database for testing.

    Returns:
        tuple: (GRBDatabase instance, path_to_database)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_grb_catalog.db"
        db = GRBDatabase(database_path=str(db_path))

        # Create tables
        db.create_tables()

        yield db, db_path


@pytest.fixture
def populated_database(temp_database, sample_grb_event):
    """
    Create a temporary database pre-populated with sample GRBs.

    Returns:
        tuple: (GRBDatabase instance, path_to_database)
    """
    db, db_path = temp_database

    # Create multiple sample GRBs
    grb_names = [
        "GRB230307A",
        "GRB230101B",
        "GRB221015C",
        "GRB230515A",
        "GRB230820D",
    ]

    for grb_name in grb_names:
        grb = sample_grb_event
        grb.name = grb_name
        grb.discovery_time = f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}T12:00:00"
        grb.redshift = np.random.uniform(0.1, 5.0)

        db.insert_grb(grb)

    return db, db_path


@pytest.fixture
def pipeline_config():
    """
    Create a test pipeline configuration.

    Returns:
        PipelineConfig: Configuration object for testing
    """
    config = PipelineConfig()

    # Set testing parameters
    config.pipeline.stages = [1, 2, 3, 4, 5, 6]
    config.pipeline.parallel = False  # Disable parallelization for testing
    config.pipeline.timeout = 60  # 60 second timeout

    config.data_acquisition.cache_data = True
    config.data_acquisition.query_timeout = 10

    config.analysis.spectral_model = "band"
    config.analysis.fit_method = "iminuit"
    config.analysis.mcmc_samples = 100  # Reduced for faster testing

    config.visualization.format = "png"
    config.visualization.dpi = 100  # Lower DPI for testing

    config.ai.enabled = False  # Disable AI for testing without API key

    return config


@pytest.fixture
def sample_timing_analysis():
    """
    Create sample temporal analysis results.

    Returns:
        TimingAnalysis: Temporal analysis object with realistic values
    """
    return TimingAnalysis(
        t90=2.345,
        t90_error=0.123,
        t50=1.234,
        t50_error=0.089,
        rise_time=0.5,
        decay_slope_early=0.8,
        decay_slope_late=1.5,
        fractional_variability=0.45,
        hardness_ratio=2.1,
        variability_timescale=0.1,
    )


@pytest.fixture
def sample_spectral_analysis():
    """
    Create sample spectral analysis results.

    Returns:
        SpectralAnalysis: Spectral analysis object with Band function parameters
    """
    return SpectralAnalysis(
        epeak=195.5,
        epeak_error=15.3,
        alpha=-0.78,
        beta=-2.31,
        photon_index=-1.2,
        photon_fluence=3.45e-6,
        energy_fluence=2.67e-5,
        eiso=2.5e54,
        eiso_error=0.3e54,
        chisq=127.3,
        dof=98,
        fit_method="iminuit",
    )


@pytest.fixture
def sample_afterglow_analysis():
    """
    Create sample afterglow analysis results.

    Returns:
        AftergowAnalysis: Afterglow modeling results
    """
    return AftergowAnalysis(
        gamma=300.0,
        gamma_error=50.0,
        jet_angle=0.02,
        opening_angle=0.035,
        decay_index=1.2,
        jet_break_time=450.0,
        energy_injection=1.5e53,
        eiso=2.5e54,
        ej=3.0e51,
        collimation_factor=150.0,
    )


@pytest.fixture
def sample_correlation_analysis():
    """
    Create sample correlation analysis results.

    Returns:
        CorrelationAnalysis: Correlation analysis object
    """
    return CorrelationAnalysis(
        epeak=195.5,
        eiso=2.5e54,
        liso=1.2e52,
        amati_residual=0.15,  # 0.15 sigma from relation
        yonetoku_residual=-0.08,
        t90=2.345,
        hardness_duration_class="Long-Bright",
    )


@pytest.fixture
def sample_classification():
    """
    Create sample GRB classification results.

    Returns:
        Classification: Classification object
    """
    return Classification(
        grb_type="Long GRB",
        confidence=0.92,
        progenitor_type="Massive Star / Collapsar",
        progenitor_description="Core collapse of massive star forming GRB-supernova",
        gamma=300.0,
        jet_angle=0.02,
        scores={
            "long": 0.92,
            "short": 0.08,
            "extended": 0.00,
        },
    )


# Session fixtures for database setup
@pytest.fixture(scope="session")
def test_data_dir():
    """
    Provide path to test data directory.

    Returns:
        Path: Path to test fixtures directory
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_config_dir():
    """
    Provide path to test configuration directory.

    Returns:
        Path: Path to test config directory
    """
    return Path(__file__).parent.parent / "config"


# Markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "database: marks tests requiring database (deselect with '-m \"not database\"')"
    )
    config.addinivalue_line(
        "markers", "network: marks tests requiring network (deselect with '-m \"not network\"')"
    )
    config.addinivalue_line(
        "markers", "ai: marks tests requiring AI/API (deselect with '-m \"not ai\"')"
    )
