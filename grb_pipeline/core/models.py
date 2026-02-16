"""Data models for GRB analysis using dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import numpy as np
from .constants import Mission, Instrument, GRBClass, SpectralModel


@dataclass
class GRBEvent:
    """
    Represents a single gamma-ray burst event.

    Attributes:
        grb_name: Canonical GRB name (e.g., "GRB230101A")
        trigger_time: Trigger time as Modified Julian Date (MJD)
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        error_radius: Localization uncertainty radius (arcmin)
        redshift: Spectroscopic redshift
        redshift_err: Redshift uncertainty
        t90: T90 duration (seconds) - time interval containing 90% of burst counts
        t90_err: T90 uncertainty (seconds)
        classification: GRB classification (SHORT, LONG, ULTRA_LONG, SPECIAL)
        discovery_mission: Mission that detected the GRB
        galactic_lat: Galactic latitude (degrees)
        galactic_lon: Galactic longitude (degrees)
        notes: Additional notes or comments
    """

    grb_name: str
    trigger_time: float = 0.0  # MJD
    ra: float = 0.0  # degrees
    dec: float = 0.0  # degrees
    error_radius: float = 0.0  # arcmin
    classification: Any = GRBClass.LONG  # GRB classification (GRBClass or str)
    discovery_mission: Any = Mission.SWIFT  # Detection mission (Mission or str)
    redshift: Optional[float] = None
    redshift_err: Optional[float] = None
    t90: Optional[float] = None  # seconds
    t90_err: Optional[float] = None
    galactic_lat: Optional[float] = None
    galactic_lon: Optional[float] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime and enum objects."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (GRBClass, Mission, Instrument, SpectralModel)):
                d[k] = v.value
            elif isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d

    @staticmethod
    def _safe_enum(enum_cls, value, default):
        """Convert string to enum safely."""
        if value is None:
            return default
        if isinstance(value, enum_cls):
            return value
        try:
            return enum_cls(value)
        except ValueError:
            for member in enum_cls:
                if member.name.upper() == str(value).upper():
                    return member
            return default

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GRBEvent":
        """Create instance from dictionary."""
        d_copy = d.copy()
        d_copy["classification"] = cls._safe_enum(
            GRBClass, d_copy.get("classification"), GRBClass.LONG
        )
        d_copy["discovery_mission"] = cls._safe_enum(
            Mission, d_copy.get("discovery_mission"), Mission.SWIFT
        )
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        if "updated_at" in d_copy and isinstance(d_copy["updated_at"], str):
            d_copy["updated_at"] = datetime.fromisoformat(d_copy["updated_at"])
        # Remove any unknown keys
        import inspect
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        d_copy = {k: v for k, v in d_copy.items() if k in valid_fields}
        return cls(**d_copy)


@dataclass
class Observation:
    """
    Represents a multi-wavelength observation of a GRB.

    Attributes:
        obs_id: Unique observation identifier
        grb_name: Associated GRB name
        mission: Observing mission
        instrument: Specific instrument used
        start_time: Observation start time (MJD)
        end_time: Observation end time (MJD)
        exposure: Total exposure time (seconds)
        energy_min: Minimum energy of observation (keV)
        energy_max: Maximum energy of observation (keV)
        data_path: Local path to observation data
        data_url: URL to online data archive
    """

    obs_id: str
    grb_name: str
    mission: Mission
    instrument: Instrument
    start_time: float  # MJD
    end_time: float
    exposure: float  # seconds
    energy_min: float  # keV
    energy_max: float
    data_path: Optional[str] = None
    data_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["mission"] = self.mission.value
        d["instrument"] = self.instrument.value
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if isinstance(d_copy.get("mission"), str):
            d_copy["mission"] = Mission(d_copy["mission"])
        if isinstance(d_copy.get("instrument"), str):
            d_copy["instrument"] = Instrument(d_copy["instrument"])
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class LightCurveData:
    """
    Time-resolved light curve data.

    Attributes:
        grb_name: GRB identifier
        instrument: Observing instrument
        time: Time array relative to trigger (seconds, np.ndarray)
        rate: Count rate array (counts/s, np.ndarray)
        rate_err: Count rate uncertainty (counts/s, np.ndarray)
        energy_band: Energy band description (e.g., "0.3-10 keV")
        time_ref: Reference time (MJD) for time array
        bg_rate: Background count rate (counts/s)
        bg_rate_err: Background rate uncertainty
        binsize: Time bin size (seconds)
    """

    grb_name: str
    instrument: Instrument
    time: np.ndarray  # relative to trigger, seconds
    rate: np.ndarray  # counts/s
    rate_err: np.ndarray  # counts/s
    energy_band: str
    time_ref: float  # MJD
    bg_rate: Optional[float] = None
    bg_rate_err: Optional[float] = None
    binsize: float = 1.0  # seconds
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, encoding numpy arrays as lists."""
        return {
            "grb_name": self.grb_name,
            "instrument": self.instrument.value,
            "time": self.time.tolist(),
            "rate": self.rate.tolist(),
            "rate_err": self.rate_err.tolist(),
            "energy_band": self.energy_band,
            "time_ref": self.time_ref,
            "bg_rate": self.bg_rate,
            "bg_rate_err": self.bg_rate_err,
            "binsize": self.binsize,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LightCurveData":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if isinstance(d_copy.get("instrument"), str):
            d_copy["instrument"] = Instrument(d_copy["instrument"])
        d_copy["time"] = np.array(d_copy["time"])
        d_copy["rate"] = np.array(d_copy["rate"])
        d_copy["rate_err"] = np.array(d_copy["rate_err"])
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class SpectralData:
    """
    Energy-resolved spectral data (spectrum).

    Attributes:
        grb_name: GRB identifier
        instrument: Observing instrument
        energy: Energy array (keV, np.ndarray)
        energy_err: Energy bin width or uncertainty (keV)
        counts: Count array (photons, np.ndarray)
        counts_err: Count uncertainty (photons)
        exposure: Effective exposure time (seconds)
        response_file: Path to instrument response matrix
        background_file: Path to background spectrum
    """

    grb_name: str
    instrument: Instrument
    energy: np.ndarray  # keV
    energy_err: np.ndarray  # keV
    counts: np.ndarray  # photons
    counts_err: np.ndarray  # photons
    exposure: float  # seconds
    response_file: Optional[str] = None
    background_file: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grb_name": self.grb_name,
            "instrument": self.instrument.value,
            "energy": self.energy.tolist(),
            "energy_err": self.energy_err.tolist(),
            "counts": self.counts.tolist(),
            "counts_err": self.counts_err.tolist(),
            "exposure": self.exposure,
            "response_file": self.response_file,
            "background_file": self.background_file,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectralData":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if isinstance(d_copy.get("instrument"), str):
            d_copy["instrument"] = Instrument(d_copy["instrument"])
        d_copy["energy"] = np.array(d_copy["energy"])
        d_copy["energy_err"] = np.array(d_copy["energy_err"])
        d_copy["counts"] = np.array(d_copy["counts"])
        d_copy["counts_err"] = np.array(d_copy["counts_err"])
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class SpectralFit:
    """
    Results from spectral fitting.

    Attributes:
        grb_name: GRB identifier
        model_name: Fitted spectral model type
        parameters: Fitted parameter dictionary
        parameter_errors: Parameter uncertainty dictionary
        epeak: Peak energy (keV) - most important GRB parameter
        epeak_err: Peak energy uncertainty
        alpha: Low-energy photon index
        alpha_err: Uncertainty
        beta: High-energy photon index
        beta_err: Uncertainty
        norm: Normalization parameter value
        norm_err: Normalization uncertainty
        energy_flux: Integrated energy flux (erg/cm²/s)
        photon_flux: Integrated photon flux (photons/cm²/s)
        chi_sq: Chi-squared value from fit
        dof: Degrees of freedom
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        fit_quality: Classification of fit quality
    """

    grb_name: str
    model_name: SpectralModel
    parameters: Dict[str, float]
    parameter_errors: Dict[str, float]
    epeak: Optional[float] = None  # keV
    epeak_err: Optional[float] = None
    alpha: Optional[float] = None
    alpha_err: Optional[float] = None
    beta: Optional[float] = None
    beta_err: Optional[float] = None
    norm: Optional[float] = None
    norm_err: Optional[float] = None
    energy_flux: Optional[float] = None  # erg/cm²/s
    photon_flux: Optional[float] = None  # photons/cm²/s
    chi_sq: Optional[float] = None
    dof: Optional[int] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    fit_quality: str = "Unknown"
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["model_name"] = self.model_name.value
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectralFit":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if isinstance(d_copy.get("model_name"), str):
            d_copy["model_name"] = SpectralModel(d_copy["model_name"])
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class AfterglowParams:
    """
    Afterglow temporal and spectral evolution parameters.

    Attributes:
        grb_name: GRB identifier
        band: Wavelength band (e.g., "X-Ray", "Optical")
        decay_index: Power-law decay index (F ∝ t^-α)
        decay_index_err: Uncertainty
        break_time: Time of decay index break (seconds after trigger)
        break_time_err: Uncertainty
        post_break_index: Decay index after break
        post_break_index_err: Uncertainty
        plateau_flux: Flux during plateau phase (erg/cm²/s)
        plateau_duration: Duration of plateau phase (seconds)
        has_jet_break: Whether jet break is detected
        closure_relation: Expected closure relation from GRB physics
    """

    grb_name: str
    band: str
    decay_index: float
    decay_index_err: float
    post_break_index: Optional[float] = None
    post_break_index_err: Optional[float] = None
    break_time: Optional[float] = None  # seconds
    break_time_err: Optional[float] = None
    plateau_flux: Optional[float] = None  # erg/cm²/s
    plateau_duration: Optional[float] = None  # seconds
    has_jet_break: bool = False
    closure_relation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AfterglowParams":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class GCNCircular:
    """
    GRB Coordinates Network (GCN) circular - rapid communication of observations.

    Attributes:
        circular_number: GCN circular number
        grb_name: Associated GRB name
        title: Circular title
        authors: Author list
        date: Publication date
        raw_text: Full circular text
        parsed_data: Extracted structured data
        extracted_redshift: Parsed redshift from text
        extracted_t90: Parsed T90 from text
        extracted_position: Parsed RA, Dec from text (tuple)
        extracted_flux_densities: List of parsed flux measurements
    """

    circular_number: int
    grb_name: Optional[str] = None
    title: str = ""
    authors: str = ""
    date: str = ""
    raw_text: str = ""
    parsed_data: Dict[str, Any] = field(default_factory=dict)
    extracted_redshift: Optional[float] = None
    extracted_t90: Optional[float] = None
    extracted_position: Optional[Tuple[float, float]] = None
    extracted_flux_densities: List[Dict[str, Any]] = field(default_factory=list)
    ai_interpretation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GCNCircular":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class FluxMeasurement:
    """
    Single-wavelength flux measurement for multi-wavelength afterglow study.

    Attributes:
        grb_name: GRB identifier
        time_mjd: Measurement time (MJD)
        wavelength_angstrom: Wavelength (Angstrom) - None for high-energy
        energy_kev: Energy (keV) - None for optical/IR
        flux: Flux value in specified units
        flux_err: Flux uncertainty
        flux_units: Units of flux ("mJy", "erg/cm²/s", "mag", etc.)
        instrument: Observing instrument
        reference: Bibliographic reference
        is_upper_limit: Whether this is an upper limit
    """

    grb_name: str
    time_mjd: float
    flux: float
    flux_err: float
    flux_units: str
    instrument: str
    wavelength_angstrom: Optional[float] = None
    energy_kev: Optional[float] = None
    reference: str = ""
    is_upper_limit: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FluxMeasurement":
        """Create instance from dictionary."""
        d_copy = d.copy()
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)


@dataclass
class AnalysisResult:
    """
    Comprehensive analysis results for a GRB.

    Attributes:
        grb_name: GRB identifier
        lightcurves: List of LightCurveData objects
        spectral_fits: List of SpectralFit objects
        afterglow_params: List of AfterglowParams objects
        correlations: Correlations found (dict of correlation_name: value)
        classification: Final classification based on analysis
        ai_interpretation: Claude AI interpretation of results
        report_path: Path to generated analysis report
    """

    grb_name: str
    lightcurves: List[LightCurveData] = field(default_factory=list)
    spectral_fits: List[SpectralFit] = field(default_factory=list)
    afterglow_params: List[AfterglowParams] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)
    classification: str = "Unknown"
    ai_interpretation: str = ""
    report_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grb_name": self.grb_name,
            "lightcurves": [lc.to_dict() for lc in self.lightcurves],
            "spectral_fits": [sf.to_dict() for sf in self.spectral_fits],
            "afterglow_params": [ap.to_dict() for ap in self.afterglow_params],
            "correlations": self.correlations,
            "classification": self.classification,
            "ai_interpretation": self.ai_interpretation,
            "report_path": self.report_path,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisResult":
        """Create instance from dictionary."""
        d_copy = d.copy()
        d_copy["lightcurves"] = [LightCurveData.from_dict(lc) for lc in d_copy.get("lightcurves", [])]
        d_copy["spectral_fits"] = [SpectralFit.from_dict(sf) for sf in d_copy.get("spectral_fits", [])]
        d_copy["afterglow_params"] = [AfterglowParams.from_dict(ap) for ap in d_copy.get("afterglow_params", [])]
        if "created_at" in d_copy and isinstance(d_copy["created_at"], str):
            d_copy["created_at"] = datetime.fromisoformat(d_copy["created_at"])
        return cls(**d_copy)
