"""Utilities for reading FITS files from GRB observations."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)


def read_fits_lightcurve(
    fits_file: str,
    hdu_index: int = 1,
    time_col: str = "TIME",
    rate_col: str = "RATE",
    error_col: str = "ERROR",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read light curve data from FITS file.

    Typical Swift XRT structure:
        HDU 0: Primary (header with metadata)
        HDU 1: Lightcurve (table with TIME, RATE, ERROR columns)

    Args:
        fits_file: Path to FITS file
        hdu_index: HDU index to read (default 1 for light curve)
        time_col: Name of TIME column
        rate_col: Name of RATE column
        error_col: Name of ERROR column

    Returns:
        Tuple of (time, rate, rate_error, header_dict)
        - time: np.ndarray of times (seconds)
        - rate: np.ndarray of count rates (counts/s)
        - rate_error: np.ndarray of rate errors
        - header_dict: Dictionary of FITS header keywords

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If expected columns not found
        Exception: For FITS reading errors
    """
    fits_path = Path(fits_file)

    if not fits_path.exists():
        logger.error(f"FITS file not found: {fits_file}")
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    try:
        with fits.open(fits_path) as hdul:
            if hdu_index >= len(hdul):
                logger.error(f"HDU index {hdu_index} out of range for {len(hdul)} HDUs")
                raise IndexError(f"HDU index {hdu_index} out of range")

            hdu = hdul[hdu_index]

            # Check for required columns
            data = hdu.data
            if data is None:
                logger.error(f"No data found in HDU {hdu_index}")
                raise ValueError(f"No data found in HDU {hdu_index}")

            # Extract columns, handling case insensitivity
            col_names = [name.upper() for name in data.names]
            time_idx = col_names.index(time_col.upper()) if time_col.upper() in col_names else None
            rate_idx = col_names.index(rate_col.upper()) if rate_col.upper() in col_names else None
            error_idx = col_names.index(error_col.upper()) if error_col.upper() in col_names else None

            if time_idx is None:
                logger.error(f"TIME column '{time_col}' not found in FITS file")
                raise KeyError(f"TIME column '{time_col}' not found")
            if rate_idx is None:
                logger.error(f"RATE column '{rate_col}' not found in FITS file")
                raise KeyError(f"RATE column '{rate_col}' not found")
            if error_idx is None:
                logger.warning(f"ERROR column '{error_col}' not found, using sqrt(rate)")
                rate_error = np.sqrt(np.abs(data[data.names[rate_idx]]))
            else:
                rate_error = data[data.names[error_idx]]

            time = np.array(data[data.names[time_idx]], dtype=np.float32)
            rate = np.array(data[data.names[rate_idx]], dtype=np.float32)

            # Extract header information
            header = hdu.header
            header_dict = {
                "ORIGIN": header.get("ORIGIN", "Unknown"),
                "INSTRUME": header.get("INSTRUME", "Unknown"),
                "TELESCOP": header.get("TELESCOP", "Unknown"),
                "TIMEREF": header.get("TIMEREF", "Unknown"),
                "TSTART": header.get("TSTART", None),
                "TSTOP": header.get("TSTOP", None),
                "EXPOSURE": header.get("EXPOSURE", None),
            }

            logger.info(f"Read light curve from {fits_path.name}: {len(time)} points")
            return time, rate, rate_error, header_dict

    except fits.HDUException as e:
        logger.error(f"FITS error reading {fits_file}: {e}")
        raise Exception(f"FITS error: {e}")
    except Exception as e:
        logger.error(f"Error reading light curve from {fits_file}: {e}")
        raise


def read_fits_spectrum(
    fits_file: str,
    hdu_index: int = 1,
    energy_col: str = "ENERGY",
    counts_col: str = "COUNTS",
    grouping_col: Optional[str] = "GROUPING",
    quality_col: Optional[str] = "QUALITY",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Read spectrum (energy vs counts) from FITS file.

    Typical Swift XRT/Fermi spectrum structure:
        HDU 0: Primary
        HDU 1: Spectrum (ENERGY, COUNTS, optionally GROUPING, QUALITY)

    Args:
        fits_file: Path to FITS file
        hdu_index: HDU index containing spectrum
        energy_col: Name of ENERGY column
        counts_col: Name of COUNTS column
        grouping_col: Name of GROUPING column (for optimal binning)
        quality_col: Name of QUALITY column (for bad channels)

    Returns:
        Tuple of (energy, energy_err, counts, quality, header_dict)
        - energy: np.ndarray of energies (keV)
        - energy_err: np.ndarray of energy bin widths
        - counts: np.ndarray of photon counts
        - quality: np.ndarray of quality flags or None
        - header_dict: Dictionary of FITS header keywords

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required columns not found
        Exception: For FITS reading errors
    """
    fits_path = Path(fits_file)

    if not fits_path.exists():
        logger.error(f"FITS file not found: {fits_file}")
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    try:
        with fits.open(fits_path) as hdul:
            if hdu_index >= len(hdul):
                logger.error(f"HDU index {hdu_index} out of range")
                raise IndexError(f"HDU index {hdu_index} out of range")

            hdu = hdul[hdu_index]
            data = hdu.data

            if data is None:
                logger.error(f"No data in HDU {hdu_index}")
                raise ValueError(f"No data in HDU {hdu_index}")

            # Extract columns (case insensitive)
            col_names = [name.upper() for name in data.names]

            # Get energy
            if energy_col.upper() not in col_names:
                logger.error(f"ENERGY column not found")
                raise KeyError(f"ENERGY column '{energy_col}' not found")
            energy_idx = col_names.index(energy_col.upper())
            energy = np.array(data[data.names[energy_idx]], dtype=np.float32)

            # Get energy error (bin width)
            energy_err_col = "ENERGY_ERR" if "ENERGY_ERR" in col_names else "DE" if "DE" in col_names else "CHANNEL"
            if energy_err_col in col_names:
                energy_err = np.array(data[data.names[col_names.index(energy_err_col)]], dtype=np.float32)
            else:
                # Assume equal spacing or linear interpolation
                if len(energy) > 1:
                    energy_err = np.diff(np.concatenate([[energy[0]], energy]))
                else:
                    energy_err = np.ones_like(energy)
                logger.warning(f"Energy error/bin width estimated from energy array")

            # Get counts
            if counts_col.upper() not in col_names:
                logger.error(f"COUNTS column not found")
                raise KeyError(f"COUNTS column '{counts_col}' not found")
            counts_idx = col_names.index(counts_col.upper())
            counts = np.array(data[data.names[counts_idx]], dtype=np.float32)

            # Get quality flags if present
            quality = None
            if quality_col and quality_col.upper() in col_names:
                quality_idx = col_names.index(quality_col.upper())
                quality = np.array(data[data.names[quality_idx]], dtype=np.int32)

            # Extract header info
            header = hdu.header
            header_dict = {
                "RESPFILE": header.get("RESPFILE", None),
                "BACKFILE": header.get("BACKFILE", None),
                "ANCRFILE": header.get("ANCRFILE", None),
                "EXPOSURE": header.get("EXPOSURE", None),
                "TELESCOP": header.get("TELESCOP", "Unknown"),
                "INSTRUME": header.get("INSTRUME", "Unknown"),
            }

            logger.info(f"Read spectrum from {fits_path.name}: {len(energy)} channels")
            return energy, energy_err, counts, quality, header_dict

    except fits.HDUException as e:
        logger.error(f"FITS error: {e}")
        raise Exception(f"FITS error: {e}")
    except Exception as e:
        logger.error(f"Error reading spectrum: {e}")
        raise


def read_fits_table(
    fits_file: str,
    hdu_index: int = 1,
    columns: Optional[list] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Read arbitrary table from FITS file.

    Args:
        fits_file: Path to FITS file
        hdu_index: HDU index containing table
        columns: Specific columns to extract (None = all)

    Returns:
        Tuple of (data_dict, header_dict)
        - data_dict: Dictionary mapping column names to arrays
        - header_dict: Dictionary of FITS header keywords

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For FITS reading errors
    """
    fits_path = Path(fits_file)

    if not fits_path.exists():
        logger.error(f"FITS file not found: {fits_file}")
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    try:
        with fits.open(fits_path) as hdul:
            if hdu_index >= len(hdul):
                logger.error(f"HDU index {hdu_index} out of range")
                raise IndexError(f"HDU index {hdu_index} out of range")

            hdu = hdul[hdu_index]
            data = hdu.data

            if data is None:
                logger.error(f"No data in HDU {hdu_index}")
                raise ValueError(f"No data in HDU {hdu_index}")

            # Extract requested columns
            data_dict = {}
            col_names = data.names if hasattr(data, "names") else []

            if columns is None:
                columns = col_names

            for col in columns:
                if col.upper() in [name.upper() for name in col_names]:
                    # Find actual column name (case insensitive)
                    actual_col = next(name for name in col_names if name.upper() == col.upper())
                    data_dict[actual_col] = np.array(data[actual_col])
                else:
                    logger.warning(f"Column '{col}' not found in FITS table")

            # Extract header
            header_dict = dict(hdu.header)

            logger.info(f"Read table from {fits_path.name}: {len(col_names)} columns, {len(data)} rows")
            return data_dict, header_dict

    except Exception as e:
        logger.error(f"Error reading FITS table: {e}")
        raise


def read_fits_coordinates(fits_file: str) -> Optional[Tuple[float, float, float]]:
    """
    Extract GRB coordinates and uncertainty from FITS primary header.

    Args:
        fits_file: Path to FITS file

    Returns:
        Tuple of (ra, dec, error_radius) in degrees or None if not found

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    fits_path = Path(fits_file)

    if not fits_path.exists():
        logger.error(f"FITS file not found: {fits_file}")
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header

            ra = header.get("RA_OBJ", None)
            dec = header.get("DEC_OBJ", None)
            error_radius = header.get("CIRCLE_R", None)

            if ra is not None and dec is not None:
                logger.info(f"Read coordinates from {fits_path.name}: RA={ra}, DEC={dec}")
                return float(ra), float(dec), float(error_radius) if error_radius else 10.0

            return None

    except fits.HDUException as e:
        logger.error(f"FITS error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading coordinates: {e}")
        return None


def convert_coords_galactic(ra: float, dec: float) -> Tuple[float, float]:
    """
    Convert RA/Dec to Galactic coordinates.

    Args:
        ra: Right ascension (degrees)
        dec: Declination (degrees)

    Returns:
        Tuple of (galactic_longitude, galactic_latitude) in degrees
    """
    try:
        sky_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        galactic = sky_coord.galactic
        return float(galactic.l.degree), float(galactic.b.degree)
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return None, None
