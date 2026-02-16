"""SQLite database backend for GRB analysis pipeline."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import pandas as pd

from .models import (
    GRBEvent,
    Observation,
    LightCurveData,
    SpectralData,
    SpectralFit,
    AfterglowParams,
    GCNCircular,
    FluxMeasurement,
)
from .constants import GRBClass, Mission, Instrument

logger = logging.getLogger(__name__)


class GRBDatabase:
    """
    SQLite database interface for GRB analysis pipeline.

    Manages persistent storage of GRB events, observations, spectral fits,
    light curves, and related analysis data. Supports CRUD operations,
    complex queries, and export to pandas DataFrames.

    Example:
        >>> db = GRBDatabase("grb_database.db")
        >>> grb = GRBEvent(grb_name="GRB230101A", ...)
        >>> db.insert_grb(grb)
        >>> result = db.get_grb("GRB230101A")
        >>> db.close()
    """

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Initialized database at {self.db_path}")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.connection.cursor()

        # GRBs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS grbs (
                grb_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT UNIQUE NOT NULL,
                trigger_time REAL,
                ra REAL NOT NULL,
                dec REAL NOT NULL,
                error_radius REAL,
                redshift REAL,
                redshift_err REAL,
                t90 REAL,
                t90_err REAL,
                classification TEXT,
                discovery_mission TEXT,
                galactic_lat REAL,
                galactic_lon REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Observations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT NOT NULL,
                mission TEXT,
                instrument TEXT,
                start_time REAL,
                end_time REAL,
                exposure REAL,
                energy_min REAL,
                energy_max REAL,
                data_path TEXT,
                data_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grb_name) REFERENCES grbs(grb_name)
            )
        """
        )

        # Spectral fits table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spectral_fits (
                fit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT NOT NULL,
                model_name TEXT,
                epeak REAL,
                epeak_err REAL,
                alpha REAL,
                alpha_err REAL,
                beta REAL,
                beta_err REAL,
                norm REAL,
                norm_err REAL,
                energy_flux REAL,
                photon_flux REAL,
                chi_sq REAL,
                dof INTEGER,
                aic REAL,
                bic REAL,
                fit_quality TEXT,
                parameters_json TEXT,
                parameter_errors_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grb_name) REFERENCES grbs(grb_name)
            )
        """
        )

        # Light curves table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lightcurves (
                lc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT NOT NULL,
                instrument TEXT,
                energy_band TEXT,
                time_start REAL,
                time_end REAL,
                rate REAL,
                rate_err REAL,
                bg_rate REAL,
                bg_rate_err REAL,
                binsize REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grb_name) REFERENCES grbs(grb_name)
            )
        """
        )

        # Afterglow parameters table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS afterglow_params (
                ag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT NOT NULL,
                band TEXT,
                decay_index REAL,
                decay_index_err REAL,
                break_time REAL,
                break_time_err REAL,
                post_break_index REAL,
                post_break_index_err REAL,
                plateau_flux REAL,
                plateau_duration REAL,
                has_jet_break INTEGER,
                closure_relation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grb_name) REFERENCES grbs(grb_name)
            )
        """
        )

        # GCN circulars table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gcn_circulars (
                gcn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT,
                circular_number INTEGER UNIQUE,
                title TEXT,
                authors TEXT,
                date TEXT,
                raw_text TEXT,
                parsed_data_json TEXT,
                extracted_redshift REAL,
                extracted_t90 REAL,
                extracted_ra REAL,
                extracted_dec REAL,
                ai_interpretation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grb_name) REFERENCES grbs(grb_name)
            )
        """
        )

        # Flux measurements table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS flux_measurements (
                flux_id INTEGER PRIMARY KEY AUTOINCREMENT,
                grb_name TEXT NOT NULL,
                time_mjd REAL,
                wavelength_angstrom REAL,
                energy_kev REAL,
                flux REAL,
                flux_err REAL,
                flux_units TEXT,
                instrument TEXT,
                reference TEXT,
                is_upper_limit INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grb_name) REFERENCES grbs(grb_name)
            )
        """
        )

        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_grbs_name ON grbs(grb_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_grb_name ON observations(grb_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spectral_fits_grb_name ON spectral_fits(grb_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lightcurves_grb_name ON lightcurves(grb_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_afterglow_grb_name ON afterglow_params(grb_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flux_grb_name ON flux_measurements(grb_name)")

        self.connection.commit()
        logger.debug("Database tables initialized")

    # ========================================================================
    # GRB CRUD Operations
    # ========================================================================

    def insert_grb(self, grb: GRBEvent) -> int:
        """
        Insert a new GRB event into the database.

        Args:
            grb: GRBEvent instance or dict with GRB fields to insert

        Returns:
            Row ID of inserted GRB

        Raises:
            sqlite3.IntegrityError: If GRB with same name already exists
        """
        cursor = self.connection.cursor()
        try:
            # Support both GRBEvent objects and plain dicts
            if isinstance(grb, dict):
                _get = lambda key, default=None: grb.get(key, default)
                _cls = grb.get('classification', '')
                _cls_val = _cls.value if hasattr(_cls, 'value') else str(_cls) if _cls else None
                _mission = grb.get('discovery_mission', '')
                _mission_val = _mission.value if hasattr(_mission, 'value') else str(_mission) if _mission else None
            else:
                _get = lambda key, default=None: getattr(grb, key, default)
                _cls_val = grb.classification.value if hasattr(grb.classification, 'value') else str(grb.classification) if grb.classification else None
                _mission_val = grb.discovery_mission.value if hasattr(grb.discovery_mission, 'value') else str(grb.discovery_mission) if grb.discovery_mission else None

            cursor.execute(
                """
                INSERT INTO grbs (
                    grb_name, trigger_time, ra, dec, error_radius,
                    redshift, redshift_err, t90, t90_err, classification,
                    discovery_mission, galactic_lat, galactic_lon, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    _get('grb_name'),
                    _get('trigger_time'),
                    _get('ra'),
                    _get('dec'),
                    _get('error_radius'),
                    _get('redshift'),
                    _get('redshift_err'),
                    _get('t90'),
                    _get('t90_err'),
                    _cls_val,
                    _mission_val,
                    _get('galactic_lat'),
                    _get('galactic_lon'),
                    _get('notes'),
                ),
            )
            self.connection.commit()
            logger.info(f"Inserted GRB {_get('grb_name')}")
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to insert GRB {_get('grb_name')}: {e}")
            raise

    def get_grb(self, grb_name: str) -> Optional[GRBEvent]:
        """
        Retrieve a GRB event by name.

        Args:
            grb_name: GRB name (e.g., "GRB230101A")

        Returns:
            GRBEvent instance or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM grbs WHERE grb_name = ?", (grb_name,))
        row = cursor.fetchone()
        if row:
            return self._row_to_grb(row)
        return None

    def update_grb(self, grb: GRBEvent) -> None:
        """
        Update an existing GRB event.

        Args:
            grb: GRBEvent instance with updated values
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE grbs SET
                trigger_time = ?, ra = ?, dec = ?, error_radius = ?,
                redshift = ?, redshift_err = ?, t90 = ?, t90_err = ?,
                classification = ?, discovery_mission = ?,
                galactic_lat = ?, galactic_lon = ?, notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE grb_name = ?
        """,
            (
                grb.trigger_time,
                grb.ra,
                grb.dec,
                grb.error_radius,
                grb.redshift,
                grb.redshift_err,
                grb.t90,
                grb.t90_err,
                grb.classification.value,
                grb.discovery_mission.value,
                grb.galactic_lat,
                grb.galactic_lon,
                grb.notes,
                grb.grb_name,
            ),
        )
        self.connection.commit()
        logger.info(f"Updated GRB {grb.grb_name}")

    def delete_grb(self, grb_name: str) -> None:
        """
        Delete a GRB and all associated data.

        Warning: This operation is permanent and cascades to related tables.

        Args:
            grb_name: GRB name to delete
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM grbs WHERE grb_name = ?", (grb_name,))
        self.connection.commit()
        logger.warning(f"Deleted GRB {grb_name} and all associated data")

    # ========================================================================
    # Observation Operations
    # ========================================================================

    def insert_observation(self, obs: Observation) -> int:
        """Insert an observation record."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO observations (
                grb_name, mission, instrument, start_time, end_time,
                exposure, energy_min, energy_max, data_path, data_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                obs.grb_name,
                obs.mission.value,
                obs.instrument.value,
                obs.start_time,
                obs.end_time,
                obs.exposure,
                obs.energy_min,
                obs.energy_max,
                obs.data_path,
                obs.data_url,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_observations(self, grb_name: str) -> List[Observation]:
        """Get all observations for a GRB."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM observations WHERE grb_name = ?", (grb_name,))
        return [self._row_to_observation(row) for row in cursor.fetchall()]

    # ========================================================================
    # Spectral Fit Operations
    # ========================================================================

    def insert_spectral_fit(self, fit: SpectralFit) -> int:
        """Insert a spectral fit result."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO spectral_fits (
                grb_name, model_name, epeak, epeak_err, alpha, alpha_err,
                beta, beta_err, norm, norm_err, energy_flux, photon_flux,
                chi_sq, dof, aic, bic, fit_quality,
                parameters_json, parameter_errors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fit.grb_name,
                fit.model_name.value,
                fit.epeak,
                fit.epeak_err,
                fit.alpha,
                fit.alpha_err,
                fit.beta,
                fit.beta_err,
                fit.norm,
                fit.norm_err,
                fit.energy_flux,
                fit.photon_flux,
                fit.chi_sq,
                fit.dof,
                fit.aic,
                fit.bic,
                fit.fit_quality,
                json.dumps(fit.parameters),
                json.dumps(fit.parameter_errors),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_spectral_fits(self, grb_name: str) -> List[SpectralFit]:
        """Get all spectral fits for a GRB."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM spectral_fits WHERE grb_name = ?", (grb_name,))
        return [self._row_to_spectral_fit(row) for row in cursor.fetchall()]

    # ========================================================================
    # Light Curve Operations
    # ========================================================================

    def insert_lightcurve_data(self, lc: LightCurveData) -> int:
        """
        Insert light curve data.

        Note: Stores binned light curve (rate, rate_err) not raw time/rate arrays.
        """
        cursor = self.connection.cursor()
        # Calculate time range from light curve data
        if len(lc.time) > 0:
            time_start = float(lc.time.min())
            time_end = float(lc.time.max())
        else:
            time_start = 0.0
            time_end = 0.0

        # Store average rate for this band
        avg_rate = float(lc.rate.mean()) if len(lc.rate) > 0 else 0.0
        avg_rate_err = float(lc.rate_err.mean()) if len(lc.rate_err) > 0 else 0.0

        cursor.execute(
            """
            INSERT INTO lightcurves (
                grb_name, instrument, energy_band, time_start, time_end,
                rate, rate_err, bg_rate, bg_rate_err, binsize
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                lc.grb_name,
                lc.instrument.value,
                lc.energy_band,
                time_start,
                time_end,
                avg_rate,
                avg_rate_err,
                lc.bg_rate,
                lc.bg_rate_err,
                lc.binsize,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    # ========================================================================
    # Afterglow Parameters Operations
    # ========================================================================

    def insert_afterglow_params(self, ap: AfterglowParams) -> int:
        """Insert afterglow parameters."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO afterglow_params (
                grb_name, band, decay_index, decay_index_err,
                break_time, break_time_err, post_break_index,
                post_break_index_err, plateau_flux, plateau_duration,
                has_jet_break, closure_relation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ap.grb_name,
                ap.band,
                ap.decay_index,
                ap.decay_index_err,
                ap.break_time,
                ap.break_time_err,
                ap.post_break_index,
                ap.post_break_index_err,
                ap.plateau_flux,
                ap.plateau_duration,
                int(ap.has_jet_break),
                ap.closure_relation,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    # ========================================================================
    # GCN Circular Operations
    # ========================================================================

    def insert_gcn_circular(self, gcn: GCNCircular) -> int:
        """Insert a GCN circular."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO gcn_circulars (
                grb_name, circular_number, title, authors, date, raw_text,
                parsed_data_json, extracted_redshift, extracted_t90,
                extracted_ra, extracted_dec, ai_interpretation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                gcn.grb_name,
                gcn.circular_number,
                gcn.title,
                gcn.authors,
                gcn.date,
                gcn.raw_text,
                json.dumps(gcn.parsed_data),
                gcn.extracted_redshift,
                gcn.extracted_t90,
                gcn.extracted_position[0] if gcn.extracted_position else None,
                gcn.extracted_position[1] if gcn.extracted_position else None,
                gcn.ai_interpretation,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    # ========================================================================
    # Flux Measurement Operations
    # ========================================================================

    def insert_flux_measurement(self, flux: FluxMeasurement) -> int:
        """Insert a flux measurement."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO flux_measurements (
                grb_name, time_mjd, wavelength_angstrom, energy_kev,
                flux, flux_err, flux_units, instrument, reference, is_upper_limit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                flux.grb_name,
                flux.time_mjd,
                flux.wavelength_angstrom,
                flux.energy_kev,
                flux.flux,
                flux.flux_err,
                flux.flux_units,
                flux.instrument,
                flux.reference,
                int(flux.is_upper_limit),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_flux_measurements(self, grb_name: str) -> List[FluxMeasurement]:
        """Get all flux measurements for a GRB."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM flux_measurements WHERE grb_name = ?", (grb_name,))
        return [self._row_to_flux_measurement(row) for row in cursor.fetchall()]

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_all_grbs(self) -> List[GRBEvent]:
        """Get all GRBs in database."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM grbs ORDER BY trigger_time DESC")
        return [self._row_to_grb(row) for row in cursor.fetchall()]

    def get_grbs_by_class(self, classification: GRBClass) -> List[GRBEvent]:
        """Get all GRBs of a specific class."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM grbs WHERE classification = ? ORDER BY trigger_time DESC", (classification.value,))
        return [self._row_to_grb(row) for row in cursor.fetchall()]

    def get_grbs_with_redshift(self) -> List[GRBEvent]:
        """Get GRBs with measured redshifts."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM grbs WHERE redshift IS NOT NULL ORDER BY redshift DESC")
        return [self._row_to_grb(row) for row in cursor.fetchall()]

    def search_grbs(self, ra: float, dec: float, radius: float) -> List[GRBEvent]:
        """
        Search for GRBs near a given position.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius: Search radius (degrees)

        Returns:
            List of GRBEvents within search radius (simple cone search)
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM grbs
            WHERE (ra BETWEEN ? AND ?)
              AND (dec BETWEEN ? AND ?)
            ORDER BY trigger_time DESC
        """,
            (ra - radius, ra + radius, dec - radius, dec + radius),
        )
        return [self._row_to_grb(row) for row in cursor.fetchall()]

    def get_grb_count(self) -> int:
        """Get total number of GRBs in database."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM grbs")
        return cursor.fetchone()[0]

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_to_pandas(self, table_name: str) -> pd.DataFrame:
        """
        Export table to pandas DataFrame.

        Args:
            table_name: Name of table to export

        Returns:
            DataFrame with table data
        """
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.connection)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @staticmethod
    def _safe_enum(enum_cls, value, default):
        """Safely convert a string to an enum, trying value then name match."""
        if not value:
            return default
        try:
            return enum_cls(value)
        except ValueError:
            # Try matching by name (case-insensitive)
            for member in enum_cls:
                if member.name.upper() == str(value).upper():
                    return member
            return default

    def _row_to_grb(self, row: sqlite3.Row) -> GRBEvent:
        """Convert database row to GRBEvent."""
        return GRBEvent(
            grb_name=row["grb_name"],
            trigger_time=row["trigger_time"],
            ra=row["ra"],
            dec=row["dec"],
            error_radius=row["error_radius"],
            redshift=row["redshift"],
            redshift_err=row["redshift_err"],
            t90=row["t90"],
            t90_err=row["t90_err"],
            classification=self._safe_enum(GRBClass, row["classification"], GRBClass.LONG),
            discovery_mission=self._safe_enum(Mission, row["discovery_mission"], Mission.SWIFT),
            galactic_lat=row["galactic_lat"],
            galactic_lon=row["galactic_lon"],
            notes=row["notes"],
        )

    def _row_to_observation(self, row: sqlite3.Row) -> Observation:
        """Convert database row to Observation."""
        return Observation(
            obs_id=row["obs_id"],
            grb_name=row["grb_name"],
            mission=self._safe_enum(Mission, row["mission"], Mission.SWIFT),
            instrument=self._safe_enum(Instrument, row["instrument"], Instrument.XRT),
            start_time=row["start_time"],
            end_time=row["end_time"],
            exposure=row["exposure"],
            energy_min=row["energy_min"],
            energy_max=row["energy_max"],
            data_path=row["data_path"],
            data_url=row["data_url"],
        )

    def _row_to_spectral_fit(self, row: sqlite3.Row) -> SpectralFit:
        """Convert database row to SpectralFit."""
        from .constants import SpectralModel

        return SpectralFit(
            grb_name=row["grb_name"],
            model_name=SpectralModel(row["model_name"]) if row["model_name"] else SpectralModel.BAND,
            parameters=json.loads(row["parameters_json"]) if row["parameters_json"] else {},
            parameter_errors=json.loads(row["parameter_errors_json"]) if row["parameter_errors_json"] else {},
            epeak=row["epeak"],
            epeak_err=row["epeak_err"],
            alpha=row["alpha"],
            alpha_err=row["alpha_err"],
            beta=row["beta"],
            beta_err=row["beta_err"],
            norm=row["norm"],
            norm_err=row["norm_err"],
            energy_flux=row["energy_flux"],
            photon_flux=row["photon_flux"],
            chi_sq=row["chi_sq"],
            dof=row["dof"],
            aic=row["aic"],
            bic=row["bic"],
            fit_quality=row["fit_quality"],
        )

    def _row_to_flux_measurement(self, row: sqlite3.Row) -> FluxMeasurement:
        """Convert database row to FluxMeasurement."""
        return FluxMeasurement(
            grb_name=row["grb_name"],
            time_mjd=row["time_mjd"],
            wavelength_angstrom=row["wavelength_angstrom"],
            energy_kev=row["energy_kev"],
            flux=row["flux"],
            flux_err=row["flux_err"],
            flux_units=row["flux_units"],
            instrument=row["instrument"],
            reference=row["reference"],
            is_upper_limit=bool(row["is_upper_limit"]),
        )

    # ========================================================================
    # Context Manager Support
    # ========================================================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
