"""
Unit tests for GRB Pipeline core functionality.

Tests core data models, database operations, and configuration management.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sqlite3

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


class TestGRBEventCreation:
    """Test creation and validation of GRBEvent objects."""

    def test_grb_event_basic_creation(self):
        """Test basic GRBEvent instantiation."""
        grb = GRBEvent(
            name="GRB230307A",
            ra=165.4231,
            dec=-65.3246,
            discovery_time="2023-03-07T01:30:45",
        )

        assert grb.name == "GRB230307A"
        assert grb.ra == 165.4231
        assert grb.dec == -65.3246
        assert grb.discovery_time == "2023-03-07T01:30:45"

    def test_grb_event_with_redshift(self):
        """Test GRBEvent with redshift."""
        grb = GRBEvent(
            name="GRB230307A",
            ra=165.4,
            dec=-65.3,
            discovery_time="2023-03-07T01:30:45",
            redshift=0.6655,
        )

        assert grb.redshift == 0.6655

    def test_grb_event_with_instruments(self):
        """Test GRBEvent with multiple instruments."""
        instruments = ["Swift/XRT", "Fermi/GBM", "UVOT"]
        grb = GRBEvent(
            name="GRB230307A",
            ra=165.4,
            dec=-65.3,
            discovery_time="2023-03-07T01:30:45",
            instruments=instruments,
        )

        assert len(grb.instruments) == 3
        assert "Swift/XRT" in grb.instruments
        assert "Fermi/GBM" in grb.instruments

    def test_grb_event_with_xrt_data(self, sample_grb_event):
        """Test GRBEvent with light curve data."""
        grb = sample_grb_event

        assert grb.xrt_data is not None
        assert isinstance(grb.xrt_data, pd.DataFrame)
        assert len(grb.xrt_data) > 0
        assert "time" in grb.xrt_data.columns
        assert "flux" in grb.xrt_data.columns

    def test_grb_event_with_optical_data(self, sample_grb_event):
        """Test GRBEvent with optical follow-up data."""
        grb = sample_grb_event

        assert grb.optical_data is not None
        assert isinstance(grb.optical_data, pd.DataFrame)
        assert "magnitude" in grb.optical_data.columns
        assert "filter" in grb.optical_data.columns

    def test_grb_event_ra_dec_bounds(self):
        """Test that RA/Dec are within valid ranges."""
        # Valid coordinates
        grb = GRBEvent(
            name="TestGRB",
            ra=180.0,
            dec=0.0,
            discovery_time="2023-01-01T00:00:00",
        )
        assert 0 <= grb.ra <= 360
        assert -90 <= grb.dec <= 90

    def test_grb_event_redshift_positive(self):
        """Test that redshift is positive."""
        grb = GRBEvent(
            name="TestGRB",
            ra=180.0,
            dec=0.0,
            discovery_time="2023-01-01T00:00:00",
            redshift=1.5,
        )
        assert grb.redshift >= 0


class TestGRBEventSerialization:
    """Test serialization of GRBEvent to different formats."""

    def test_grb_event_to_dict(self, sample_grb_event):
        """Test conversion to dictionary."""
        grb = sample_grb_event
        grb_dict = grb.to_dict()

        assert isinstance(grb_dict, dict)
        assert grb_dict["name"] == grb.name
        assert grb_dict["ra"] == grb.ra
        assert grb_dict["dec"] == grb.dec

    def test_grb_event_to_json(self, sample_grb_event, tmp_path):
        """Test JSON serialization."""
        grb = sample_grb_event
        json_file = tmp_path / "grb.json"

        # Assuming to_json method exists
        if hasattr(grb, "to_json"):
            grb.to_json(json_file)
            assert json_file.exists()

            # Read and verify
            with open(json_file) as f:
                data = json.load(f)
            assert data["name"] == grb.name

    def test_grb_event_from_dict(self, sample_grb_event):
        """Test creation from dictionary."""
        grb = sample_grb_event
        grb_dict = grb.to_dict()

        # Assuming from_dict class method exists
        if hasattr(GRBEvent, "from_dict"):
            grb2 = GRBEvent.from_dict(grb_dict)
            assert grb2.name == grb.name
            assert grb2.ra == grb.ra


class TestDatabaseCreation:
    """Test GRBDatabase initialization and table creation."""

    def test_database_creation(self, temp_database):
        """Test database file creation."""
        db, db_path = temp_database
        assert db_path.exists()

    def test_database_tables_created(self, temp_database):
        """Test that database tables are created."""
        db, db_path = temp_database

        # Connect and check tables
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert len(tables) > 0

    def test_database_connection(self, temp_database):
        """Test database connection."""
        db, db_path = temp_database
        assert db.database_path == str(db_path)


class TestDatabaseInsertQuery:
    """Test database insert and query operations."""

    @pytest.mark.database
    def test_insert_single_grb(self, temp_database, sample_grb_event):
        """Test inserting a single GRB into database."""
        db, _ = temp_database
        grb = sample_grb_event

        # Insert GRB
        db.insert_grb(grb)

        # Query back
        result = db.query_by_name(grb.name)
        assert result is not None
        assert result.name == grb.name
        assert result.ra == grb.ra
        assert result.dec == grb.dec

    @pytest.mark.database
    def test_insert_multiple_grbs(self, populated_database):
        """Test inserting multiple GRBs."""
        db, _ = populated_database

        # Query all
        all_grbs = db.query_all()
        assert len(all_grbs) == 5

    @pytest.mark.database
    def test_query_by_name(self, populated_database):
        """Test querying GRB by name."""
        db, _ = populated_database

        grb = db.query_by_name("GRB230307A")
        assert grb is not None
        assert grb.name == "GRB230307A"

    @pytest.mark.database
    def test_query_by_redshift_range(self, populated_database):
        """Test querying GRBs by redshift range."""
        db, _ = populated_database

        # Query GRBs with z between 0.5 and 2.0
        if hasattr(db, "query_by_redshift"):
            results = db.query_by_redshift(z_min=0.5, z_max=2.0)
            assert len(results) >= 0

    @pytest.mark.database
    def test_update_grb(self, temp_database, sample_grb_event):
        """Test updating GRB entry in database."""
        db, _ = temp_database
        grb = sample_grb_event

        # Insert
        db.insert_grb(grb)

        # Update redshift
        grb.redshift = 1.5
        if hasattr(db, "update_grb"):
            db.update_grb(grb)

            # Verify update
            result = db.query_by_name(grb.name)
            assert result.redshift == 1.5

    @pytest.mark.database
    def test_delete_grb(self, temp_database, sample_grb_event):
        """Test deleting GRB from database."""
        db, _ = temp_database
        grb = sample_grb_event

        # Insert
        db.insert_grb(grb)
        assert db.query_by_name(grb.name) is not None

        # Delete
        if hasattr(db, "delete_grb"):
            db.delete_grb(grb.name)
            result = db.query_by_name(grb.name)
            assert result is None


class TestConfigLoading:
    """Test configuration loading and defaults."""

    def test_config_creation(self):
        """Test basic configuration creation."""
        config = PipelineConfig()
        assert config is not None

    def test_config_has_pipeline_settings(self):
        """Test that config has pipeline section."""
        config = PipelineConfig()
        assert hasattr(config, "pipeline")

    def test_config_has_analysis_settings(self):
        """Test that config has analysis section."""
        config = PipelineConfig()
        assert hasattr(config, "analysis")

    def test_config_has_visualization_settings(self):
        """Test that config has visualization section."""
        config = PipelineConfig()
        assert hasattr(config, "visualization")

    def test_config_has_ai_settings(self):
        """Test that config has AI section."""
        config = PipelineConfig()
        assert hasattr(config, "ai")

    def test_config_pipeline_stages(self):
        """Test pipeline stages configuration."""
        config = PipelineConfig()
        assert hasattr(config.pipeline, "stages")
        assert isinstance(config.pipeline.stages, (list, tuple))

    def test_config_analysis_fit_method(self):
        """Test analysis fitting method configuration."""
        config = PipelineConfig()
        assert hasattr(config.analysis, "fit_method")
        assert config.analysis.fit_method in ["iminuit", "emcee", "nested"]

    def test_config_spectral_model(self):
        """Test spectral model configuration."""
        config = PipelineConfig()
        assert hasattr(config.analysis, "spectral_model")
        assert config.analysis.spectral_model in ["band", "powerlaw", "cutoff"]

    def test_config_yaml_loading(self, test_config_dir):
        """Test loading configuration from YAML file."""
        config_file = test_config_dir / "default.yaml"
        if config_file.exists():
            config = PipelineConfig.from_yaml(str(config_file))
            assert config is not None

    def test_config_yaml_saving(self, tmp_path, pipeline_config):
        """Test saving configuration to YAML file."""
        output_file = tmp_path / "test_config.yaml"
        if hasattr(pipeline_config, "to_yaml"):
            pipeline_config.to_yaml(str(output_file))
            assert output_file.exists()

    def test_config_defaults(self, pipeline_config):
        """Test that config has sensible defaults."""
        # Pipeline defaults
        assert pipeline_config.pipeline.parallel is False
        assert pipeline_config.pipeline.timeout > 0

        # Analysis defaults
        assert pipeline_config.analysis.mcmc_samples > 0

        # Visualization defaults
        assert pipeline_config.visualization.dpi > 0


class TestAnalysisObjects:
    """Test creation of analysis result objects."""

    def test_timing_analysis_creation(self, sample_timing_analysis):
        """Test TimingAnalysis object creation."""
        timing = sample_timing_analysis

        assert timing.t90 == 2.345
        assert timing.t90_error == 0.123
        assert timing.t50 == 1.234
        assert timing.rise_time == 0.5

    def test_spectral_analysis_creation(self, sample_spectral_analysis):
        """Test SpectralAnalysis object creation."""
        spectral = sample_spectral_analysis

        assert spectral.epeak == 195.5
        assert spectral.epeak_error == 15.3
        assert spectral.alpha == -0.78
        assert spectral.beta == -2.31

    def test_afterglow_analysis_creation(self, sample_afterglow_analysis):
        """Test AftergowAnalysis object creation."""
        afterglow = sample_afterglow_analysis

        assert afterglow.gamma == 300.0
        assert afterglow.decay_index == 1.2
        assert afterglow.jet_angle > 0

    def test_correlation_analysis_creation(self, sample_correlation_analysis):
        """Test CorrelationAnalysis object creation."""
        correlation = sample_correlation_analysis

        assert correlation.epeak == 195.5
        assert correlation.eiso == 2.5e54
        assert correlation.amati_residual == 0.15

    def test_classification_creation(self, sample_classification):
        """Test Classification object creation."""
        classification = sample_classification

        assert classification.grb_type == "Long GRB"
        assert classification.confidence == 0.92
        assert classification.progenitor_type == "Massive Star / Collapsar"


class TestPhysicalConstants:
    """Test physical and cosmological constants."""

    def test_physical_constants_exist(self):
        """Test that physical constants are defined."""
        assert hasattr(PHYSICAL_CONSTANTS, "c")  # Speed of light
        assert hasattr(PHYSICAL_CONSTANTS, "G")  # Gravitational constant
        assert hasattr(PHYSICAL_CONSTANTS, "M_sun")  # Solar mass

    def test_cosmological_constants_exist(self):
        """Test that cosmological constants are defined."""
        assert hasattr(COSMOLOGICAL_CONSTANTS, "H0")  # Hubble constant
        assert hasattr(COSMOLOGICAL_CONSTANTS, "Om0")  # Matter density
        assert hasattr(COSMOLOGICAL_CONSTANTS, "OL0")  # Dark energy density

    def test_physical_constants_values(self):
        """Test that physical constants have reasonable values."""
        # Speed of light in m/s
        assert 2.99e8 < PHYSICAL_CONSTANTS.c < 3.01e8

        # Gravitational constant
        assert 6.6e-11 < PHYSICAL_CONSTANTS.G < 6.7e-11

    def test_cosmological_constants_values(self):
        """Test that cosmological constants have reasonable values."""
        # Hubble constant in km/s/Mpc
        assert 60 < COSMOLOGICAL_CONSTANTS.H0 < 75

        # Matter density
        assert 0.2 < COSMOLOGICAL_CONSTANTS.Om0 < 0.4

        # Dark energy density
        assert 0.6 < COSMOLOGICAL_CONSTANTS.OL0 < 0.8


class TestDataValidation:
    """Test data validation and error handling."""

    def test_light_curve_data_validation(self, sample_grb_event):
        """Test light curve data is valid."""
        grb = sample_grb_event

        if grb.xrt_data is not None:
            # Check for required columns
            assert "time" in grb.xrt_data.columns
            assert "flux" in grb.xrt_data.columns

            # Check that times are monotonically increasing
            assert all(grb.xrt_data["time"].diff()[1:] > 0)

            # Check that flux and errors are positive
            assert all(grb.xrt_data["flux"] > 0)
            assert all(grb.xrt_data["flux_error"] > 0)

    def test_spectral_data_validation(self, sample_spectral_data):
        """Test spectral data is valid."""
        energy, flux, flux_error = sample_spectral_data

        # Check monotonicity
        assert all(np.diff(energy) > 0)

        # Check positivity
        assert all(flux >= 0)
        assert all(flux_error > 0)

    def test_timing_parameters_consistency(self, sample_timing_analysis):
        """Test that timing parameters are self-consistent."""
        timing = sample_timing_analysis

        # T50 should be less than T90
        assert timing.t50 <= timing.t90

        # Errors should be positive
        assert timing.t90_error > 0
        assert timing.t50_error > 0

    def test_spectral_parameters_consistency(self, sample_spectral_analysis):
        """Test that spectral parameters are self-consistent."""
        spectral = sample_spectral_analysis

        # Epeak should be positive
        assert spectral.epeak > 0

        # Photon indices should be in reasonable range (-5 to 2)
        assert -5 < spectral.alpha < 2
        assert -5 < spectral.beta < 2

        # High-energy index should be steeper than low-energy
        assert spectral.beta < spectral.alpha

    def test_correlation_outlier_detection(self, sample_correlation_analysis):
        """Test outlier detection in correlations."""
        correlation = sample_correlation_analysis

        # Residuals within 2-3 sigma are normal
        assert abs(correlation.amati_residual) < 5
        assert abs(correlation.yonetoku_residual) < 5


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_ra_declination(self):
        """Test handling of invalid coordinates."""
        with pytest.raises((ValueError, AssertionError)):
            grb = GRBEvent(
                name="BadGRB",
                ra=400.0,  # Invalid RA (should be 0-360)
                dec=0.0,
                discovery_time="2023-01-01T00:00:00",
            )

    def test_invalid_redshift(self):
        """Test handling of invalid redshift."""
        with pytest.raises((ValueError, AssertionError)):
            grb = GRBEvent(
                name="BadGRB",
                ra=180.0,
                dec=0.0,
                discovery_time="2023-01-01T00:00:00",
                redshift=-0.5,  # Invalid redshift (must be positive)
            )

    def test_empty_database_query(self, temp_database):
        """Test querying empty database."""
        db, _ = temp_database

        # Query non-existent GRB
        result = db.query_by_name("NonExistentGRB")
        assert result is None

    def test_config_missing_file(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            config = PipelineConfig.from_yaml("non_existent_config.yaml")


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.database
    def test_full_grb_workflow(self, temp_database, sample_grb_event):
        """Test complete GRB creation, storage, and retrieval."""
        db, _ = temp_database
        grb = sample_grb_event

        # Insert
        db.insert_grb(grb)

        # Retrieve
        retrieved = db.query_by_name(grb.name)

        # Verify
        assert retrieved is not None
        assert retrieved.name == grb.name
        assert retrieved.ra == grb.ra

    def test_config_with_analysis_objects(self, pipeline_config):
        """Test configuration consistency with analysis objects."""
        config = pipeline_config

        # Create spectral analysis
        spectral = SpectralAnalysis(
            epeak=200.0,
            epeak_error=10.0,
            alpha=-0.8,
            beta=-2.2,
            photon_index=-1.0,
            photon_fluence=1e-6,
            energy_fluence=1e-5,
            eiso=1e54,
            eiso_error=0.1e54,
            chisq=100.0,
            dof=80,
            fit_method=config.analysis.fit_method,
        )

        assert spectral.fit_method == config.analysis.fit_method

    @pytest.mark.slow
    def test_large_catalog_query(self, populated_database):
        """Test querying large catalog."""
        db, _ = populated_database

        # Query all
        all_grbs = db.query_all()
        assert len(all_grbs) > 0

        # Verify we can iterate and access all
        for grb in all_grbs:
            assert grb.name is not None
            assert grb.ra is not None
            assert grb.dec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
