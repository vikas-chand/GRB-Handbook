"""Physical constants and mission-specific parameters for GRB analysis."""

from enum import Enum
from typing import Dict, Any

# ============================================================================
# Physical Constants (CGS units where applicable)
# ============================================================================

# Speed of light (cm/s)
SPEED_OF_LIGHT = 2.99792458e10

# Planck constant (erg*s)
PLANCK_CONSTANT = 6.62607015e-27

# Reduced Planck constant (erg*s)
REDUCED_PLANCK = 1.054571817e-27

# Boltzmann constant (erg/K)
BOLTZMANN_CONSTANT = 1.380649e-16

# Electron mass (g)
ELECTRON_MASS = 9.1093837015e-28

# Electron mass energy (keV)
ELECTRON_MASS_KEV = 510.99895

# Proton mass (g)
PROTON_MASS = 1.67262192369e-24

# Proton mass energy (keV)
PROTON_MASS_KEV = 938.27208816

# Gravitational constant (cm^3 g^-1 s^-2)
GRAVITATIONAL_CONSTANT = 6.67430e-8

# Electron charge (esu)
ELECTRON_CHARGE = 4.803204257e-10

# Fine structure constant (dimensionless)
FINE_STRUCTURE_CONSTANT = 7.2973525693e-3

# Thomson cross section (cm^2)
THOMSON_CROSS_SECTION = 6.6524e-25

# ============================================================================
# Cosmological Constants (Planck 2018)
# ============================================================================

# Hubble constant (km/s/Mpc)
HUBBLE_CONSTANT = 67.4

# Matter density parameter
OMEGA_MATTER = 0.315

# Dark energy density parameter
OMEGA_LAMBDA = 0.685

# Curvature density parameter
OMEGA_CURVATURE = 1.0 - OMEGA_MATTER - OMEGA_LAMBDA

# Baryon density parameter
OMEGA_BARYON = 0.049

# Age of universe (Gyr)
UNIVERSE_AGE = 13.801

# ============================================================================
# Energy Conversion Factors
# ============================================================================

# 1 keV in erg
KEV_TO_ERG = 1.60217663e-9

# 1 erg in Joule
ERG_TO_JOULE = 1.0e-7

# 1 keV in Joule
KEV_TO_JOULE = KEV_TO_ERG * ERG_TO_JOULE

# 1 Joule in erg
JOULE_TO_ERG = 1.0e7

# 1 erg in keV
ERG_TO_KEV = 1.0 / KEV_TO_ERG

# ============================================================================
# Distance/Time Conversion Factors
# ============================================================================

# 1 parsec in cm
PARSEC_TO_CM = 3.0857e18

# 1 kiloparsec in cm
KILOPARSEC_TO_CM = PARSEC_TO_CM * 1000

# 1 megaparsec in cm
MEGAPARSEC_TO_CM = PARSEC_TO_CM * 1.0e6

# 1 solar radius in cm
SOLAR_RADIUS = 6.9570e10

# 1 solar mass in g
SOLAR_MASS = 1.98892e33

# 1 day in seconds
DAY_TO_SECONDS = 86400.0

# 1 year in seconds
YEAR_TO_SECONDS = 365.25 * DAY_TO_SECONDS

# ============================================================================
# Mission-Specific Constants
# ============================================================================

# Swift BAT (Burst Alert Telescope)
SWIFT_BAT_ENERGY_MIN = 15.0  # keV
SWIFT_BAT_ENERGY_MAX = 150.0  # keV
SWIFT_BAT_RESOLUTION = 0.064  # degrees

# Swift XRT (X-Ray Telescope)
SWIFT_XRT_ENERGY_MIN = 0.3  # keV
SWIFT_XRT_ENERGY_MAX = 10.0  # keV
SWIFT_XRT_RESOLUTION = 5.4  # arcsec

# Swift UVOT (UV/Optical Telescope)
SWIFT_UVOT_FILTERS = {
    "V": {"wavelength": 5560, "fwhm": 1050},  # Angstrom
    "B": {"wavelength": 4380, "fwhm": 900},
    "U": {"wavelength": 3460, "fwhm": 600},
    "UVW1": {"wavelength": 2600, "fwhm": 690},
    "UVM2": {"wavelength": 2246, "fwhm": 498},
    "UVW2": {"wavelength": 1928, "fwhm": 657},
    "WHITE": {"wavelength": 7000, "fwhm": 4000},
}
SWIFT_UVOT_RESOLUTION = 2.5  # arcsec

# Fermi GBM (Gamma-ray Burst Monitor)
FERMI_GBM_ENERGY_MIN = 8.0e-3  # keV
FERMI_GBM_ENERGY_MAX = 40000.0  # keV
FERMI_GBM_NBI_RESOLUTION = 0.096  # seconds (best resolution)
FERMI_GBM_DETECTORS = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "b0", "b1"]

# Fermi LAT (Large Area Telescope)
FERMI_LAT_ENERGY_MIN = 20.0  # MeV
FERMI_LAT_ENERGY_MAX = 300000.0  # MeV
FERMI_LAT_RESOLUTION = 0.1  # degrees (depends on energy)

# INTEGRAL
INTEGRAL_IBIS_ENERGY_MIN = 15.0  # keV
INTEGRAL_IBIS_ENERGY_MAX = 10000.0  # keV
INTEGRAL_SPI_ENERGY_MIN = 20.0  # keV
INTEGRAL_SPI_ENERGY_MAX = 8000.0  # keV

# MAXI (Monitor of All-sky X-ray Image)
MAXI_ENERGY_MIN = 0.7  # keV
MAXI_ENERGY_MAX = 30.0  # keV

# HXMT (Hard X-ray Modulation Telescope)
HXMT_LE_ENERGY_MIN = 1.0  # keV
HXMT_LE_ENERGY_MAX = 8.0  # keV
HXMT_ME_ENERGY_MIN = 5.0  # keV
HXMT_ME_ENERGY_MAX = 30.0  # keV
HXMT_HE_ENERGY_MIN = 20.0  # keV
HXMT_HE_ENERGY_MAX = 250.0  # keV

# ============================================================================
# Enum Classes
# ============================================================================


class Mission(str, Enum):
    """Supported GRB observation missions."""

    SWIFT = "Swift"
    FERMI = "Fermi"
    INTEGRAL = "INTEGRAL"
    MAXI = "MAXI"
    HXMT = "HXMT"
    AGILE = "AGILE"
    SUZAKU = "Suzaku"
    XMM = "XMM-Newton"
    CHANDRA = "Chandra"
    RXTE = "RXTE"


class Instrument(str, Enum):
    """Supported instruments for GRB observations."""

    # Swift
    BAT = "BAT"
    XRT = "XRT"
    UVOT = "UVOT"

    # Fermi
    GBM = "GBM"
    LAT = "LAT"

    # INTEGRAL
    IBIS = "IBIS"
    SPI = "SPI"
    JEM_X = "JEM-X"

    # Others
    MAXI = "MAXI"
    LE = "LE"  # HXMT Low Energy
    ME = "ME"  # HXMT Medium Energy
    HE = "HE"  # HXMT High Energy
    AGILE_IMAGER = "AGILE-IMAGER"
    AGILE_MCAL = "AGILE-MCAL"


class GRBClass(str, Enum):
    """GRB classification based on T90 duration."""

    SHORT = "Short"  # T90 < 2 seconds
    LONG = "Long"  # T90 >= 2 seconds
    ULTRA_LONG = "Ultra-long"  # T90 > 1000 seconds
    SPECIAL = "Special"  # Unusual GRB (e.g., plateau-dominated)


class SpectralModel(str, Enum):
    """Spectral model types for GRB analysis."""

    BAND = "Band"  # Band et al. model (broken power law)
    CPL = "CPL"  # Cutoff Power Law
    PL = "PL"  # Simple Power Law
    BBODY = "BBody"  # Blackbody
    SBPL = "SBPL"  # Smoothly Broken Power Law
    BKNPOWER = "BknPower"  # Broken Power Law with intermediate slope
    COMPTONIZED = "Comptonized"  # Comptonized spectrum


class FitQuality(str, Enum):
    """Classification of spectral fit quality."""

    EXCELLENT = "Excellent"  # χ²/dof < 0.8, BIC/AIC favorable
    GOOD = "Good"  # 0.8 < χ²/dof < 1.2
    ACCEPTABLE = "Acceptable"  # 1.2 < χ²/dof < 2.0
    POOR = "Poor"  # χ²/dof > 2.0


class WavelengthBand(str, Enum):
    """Wavelength bands for multi-wavelength afterglow observations."""

    RADIO = "Radio"
    SUBMILLIMETER = "Submillimeter"
    INFRARED = "Infrared"
    OPTICAL = "Optical"
    ULTRAVIOLET = "UV"
    X_RAY = "X-Ray"
    GAMMA_RAY = "Gamma-Ray"


# ============================================================================
# Mission-Specific Parameter Dictionaries
# ============================================================================

MISSION_PARAMETERS: Dict[Mission, Dict[str, Any]] = {
    Mission.SWIFT: {
        "instruments": [Instrument.BAT, Instrument.XRT, Instrument.UVOT],
        "slew_time": 60.0,  # seconds to point
        "location_precision": 3.0,  # arcmin, 90% confidence
        "data_archive": "https://www.swift.ac.uk",
    },
    Mission.FERMI: {
        "instruments": [Instrument.GBM, Instrument.LAT],
        "all_sky_coverage": True,
        "location_precision": 10.0,  # degrees (GBM), 0.1 (LAT)
        "data_archive": "https://fermi.gsfc.nasa.gov",
    },
    Mission.INTEGRAL: {
        "instruments": [Instrument.IBIS, Instrument.SPI, Instrument.JEM_X],
        "location_precision": 2.0,  # arcmin
        "data_archive": "https://www.isdc.unige.ch",
    },
    Mission.MAXI: {
        "instruments": [Instrument.MAXI],
        "all_sky_coverage": True,
        "location_precision": 10.0,  # arcmin
        "data_archive": "http://maxi.riken.jp",
    },
    Mission.HXMT: {
        "instruments": [Instrument.LE, Instrument.ME, Instrument.HE],
        "location_precision": 15.0,  # arcmin
        "data_archive": "http://www.ihep.ac.cn/hxmt",
    },
}

# ============================================================================
# Spectral Model Parameter Templates
# ============================================================================

SPECTRAL_MODEL_PARAMETERS: Dict[SpectralModel, Dict[str, str]] = {
    SpectralModel.BAND: {
        "epeak": "Peak energy (keV)",
        "alpha": "Low-energy photon index",
        "beta": "High-energy photon index",
        "norm": "Normalization at 1 keV (photons/cm²/s/keV)",
    },
    SpectralModel.CPL: {
        "gamma": "Photon index",
        "ecut": "Cutoff energy (keV)",
        "norm": "Normalization at 1 keV (photons/cm²/s/keV)",
    },
    SpectralModel.PL: {
        "gamma": "Photon index",
        "norm": "Normalization at 1 keV (photons/cm²/s/keV)",
    },
    SpectralModel.BBODY: {
        "kT": "Temperature (keV)",
        "norm": "Normalization (10⁻³ L₃₉/(d/10kpc)²)",
    },
    SpectralModel.SBPL: {
        "gamma1": "Low-energy photon index",
        "gamma2": "High-energy photon index",
        "epeak": "Break energy (keV)",
        "norm": "Normalization at 1 keV (photons/cm²/s/keV)",
    },
}

# ============================================================================
# Afterglow Closure Relations (Energy-dependent indices)
# ============================================================================

CLOSURE_RELATIONS: Dict[str, Dict[str, float]] = {
    "ISM_FS": {"optical_decay": 0.5, "xray_decay": 0.5, "description": "Fireball, ISM, forward shock"},
    "ISM_RS": {"optical_decay": 1.5, "xray_decay": 1.5, "description": "Reverse shock dominated"},
    "Wind_FS": {"optical_decay": 0.0, "xray_decay": 0.0, "description": "Wind environment, forward shock"},
    "Jet": {"description": "Post-jet-break decay"},
}

# ============================================================================
# Typical GRB Parameter Ranges (for validation/reasonable defaults)
# ============================================================================

TYPICAL_RANGES = {
    "redshift": (0.0, 20.0),
    "t90": (0.01, 10000.0),  # seconds
    "epeak": (10.0, 10000.0),  # keV
    "alpha": (-2.0, 0.0),  # photon index
    "beta": (-5.0, 0.0),
    "decay_index": (-3.0, 2.0),
    "break_time": (0.1, 1e6),  # seconds
}
