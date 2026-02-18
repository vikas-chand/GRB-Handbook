"""
Light curve analysis module for GRB events.

Provides general-purpose light curve extraction, rebinning, smoothing, and
Bayesian blocks segmentation, as well as Swift BAT-specific multi-band
catalog light curve I/O and visualization.

Classes
-------
LightCurveData
    Dataclass container for a single-band light curve.
LightCurveAnalyzer
    General-purpose light curve analysis (binning, Bayesian blocks, fluence,
    background subtraction, smoothing, multi-band merging).
BATLightCurve
    Swift BAT catalog light curve reader, multi-band Bayesian blocks
    analysis, and stacked plotting.

Notes
-----
Optional dependencies (astropy, matplotlib) are imported lazily so that the
core analysis functionality remains available without them.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy optional-dependency helpers
# ---------------------------------------------------------------------------

_HAS_ASTROPY: Optional[bool] = None
_HAS_MATPLOTLIB: Optional[bool] = None


def _check_astropy() -> bool:
    """Return True if ``astropy.stats.bayesian_blocks`` is importable."""
    global _HAS_ASTROPY
    if _HAS_ASTROPY is None:
        try:
            from astropy.stats import bayesian_blocks as _bb  # noqa: F401
            _HAS_ASTROPY = True
        except ImportError:
            _HAS_ASTROPY = False
    return _HAS_ASTROPY


def _check_matplotlib() -> bool:
    """Return True if ``matplotlib`` is importable."""
    global _HAS_MATPLOTLIB
    if _HAS_MATPLOTLIB is None:
        try:
            import matplotlib  # noqa: F401
            _HAS_MATPLOTLIB = True
        except ImportError:
            _HAS_MATPLOTLIB = False
    return _HAS_MATPLOTLIB


# ---------------------------------------------------------------------------
# BAT energy band definitions
# ---------------------------------------------------------------------------

BAT_BANDS: Dict[str, Tuple[float, float]] = {
    "15-25 keV": (15.0, 25.0),
    "25-50 keV": (25.0, 50.0),
    "50-100 keV": (50.0, 100.0),
    "100-350 keV": (100.0, 350.0),
}
"""Standard Swift BAT energy bands and their boundaries in keV."""

BAT_BAND_NAMES: List[str] = list(BAT_BANDS.keys())
"""Ordered list of BAT band names matching the catalog column layout."""


# ===================================================================
# LightCurveData
# ===================================================================

@dataclass
class LightCurveData:
    """Container for light curve data.

    Parameters
    ----------
    time : np.ndarray
        Bin-centre times in seconds relative to trigger.
    rate : np.ndarray
        Count rate (counts s-1).
    rate_err : np.ndarray
        1-sigma statistical uncertainty on *rate*.
    energy_min : float, optional
        Lower bound of the energy band (keV).
    energy_max : float, optional
        Upper bound of the energy band (keV).
    binsize : float
        Time-bin width in seconds.
    mission : str
        Originating mission (e.g. ``"Swift"``).
    instrument : str
        Originating instrument (e.g. ``"BAT"``).
    """

    time: np.ndarray
    rate: np.ndarray
    rate_err: np.ndarray
    energy_min: float = None
    energy_max: float = None
    binsize: float = 1.0
    mission: str = "Unknown"
    instrument: str = "Unknown"

    def copy(self):
        """Create a deep copy of this light curve."""
        return LightCurveData(
            time=self.time.copy(),
            rate=self.rate.copy(),
            rate_err=self.rate_err.copy(),
            energy_min=self.energy_min,
            energy_max=self.energy_max,
            binsize=self.binsize,
            mission=self.mission,
            instrument=self.instrument,
        )

    @property
    def band_label(self) -> str:
        """Human-readable energy band label, or ``''`` if unset."""
        if self.energy_min is not None and self.energy_max is not None:
            return f"{self.energy_min:.0f}-{self.energy_max:.0f} keV"
        return ""


# ===================================================================
# LightCurveAnalyzer
# ===================================================================

class LightCurveAnalyzer:
    """General-purpose GRB light curve analysis toolkit.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  Recognised keys:

        * ``bayesian_blocks_ncp_prior`` (float) -- penalty term for the
          custom Bayesian blocks algorithm (default 4).
        * ``bayesian_blocks_p0`` (float) -- false-alarm probability for
          astropy ``bayesian_blocks`` (default 0.01).
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.bayesian_blocks_ncp_prior = self.config.get(
            "bayesian_blocks_ncp_prior", 4
        )
        self.bayesian_blocks_p0 = self.config.get("bayesian_blocks_p0", 0.01)

    # ----------------------------------------------------------------
    # Light curve extraction / rebinning
    # ----------------------------------------------------------------

    def extract_lightcurve(
        self,
        event_data: np.ndarray,
        energy_range: Optional[Tuple[float, float]] = None,
        binsize: float = 1.0,
        mission: str = "Unknown",
        instrument: str = "Unknown",
    ) -> LightCurveData:
        """Extract a binned light curve from photon event data.

        Parameters
        ----------
        event_data : np.ndarray
            Event arrival times.  Shape ``(n_events,)`` for times only, or
            ``(n_events, 2)`` where the second column contains energies.
        energy_range : tuple of float, optional
            ``(emin, emax)`` in keV.  If *None*, all events are used.
        binsize : float
            Time bin width in seconds.
        mission : str
            Mission name.
        instrument : str
            Instrument name.

        Returns
        -------
        LightCurveData
        """
        if event_data.ndim == 2:
            times = event_data[:, 0]
            energies = event_data[:, 1]
        else:
            times = event_data
            energies = None

        # Filter by energy range if provided
        if energy_range is not None and energies is not None:
            mask = (energies >= energy_range[0]) & (energies <= energy_range[1])
            times = times[mask]
            emin, emax = energy_range
        else:
            emin, emax = None, None

        # Create time bins
        t_start = np.floor(times.min())
        t_end = np.ceil(times.max())
        bin_edges = np.arange(t_start, t_end + binsize, binsize)

        # Bin the data
        counts, _ = np.histogram(times, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate errors as sqrt(counts) for Poisson statistics
        rate = counts / binsize
        rate_err = np.sqrt(counts) / binsize

        return LightCurveData(
            time=bin_centers,
            rate=rate,
            rate_err=rate_err,
            energy_min=emin,
            energy_max=emax,
            binsize=binsize,
            mission=mission,
            instrument=instrument,
        )

    def rebin(
        self,
        lc: LightCurveData,
        new_binsize: float,
    ) -> LightCurveData:
        """Rebin a light curve to a coarser time resolution.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve.
        new_binsize : float
            Target bin width in seconds (must be >= current bin size).

        Returns
        -------
        LightCurveData

        Raises
        ------
        ValueError
            If *new_binsize* is smaller than the current bin size.
        """
        factor = int(np.round(new_binsize / lc.binsize))
        if factor < 1:
            raise ValueError("Cannot rebin to smaller size")

        n_new = len(lc.time) // factor
        time_new = lc.time[: n_new * factor].reshape(n_new, factor).mean(axis=1)
        rate_new = (
            lc.rate[: n_new * factor].reshape(n_new, factor).sum(axis=1) / factor
        )

        rate_err_squared = lc.rate_err[: n_new * factor].reshape(n_new, factor) ** 2
        rate_err_new = np.sqrt(rate_err_squared.sum(axis=1)) / factor

        return LightCurveData(
            time=time_new,
            rate=rate_new,
            rate_err=rate_err_new,
            energy_min=lc.energy_min,
            energy_max=lc.energy_max,
            binsize=new_binsize,
            mission=lc.mission,
            instrument=lc.instrument,
        )

    # ----------------------------------------------------------------
    # Bayesian blocks
    # ----------------------------------------------------------------

    def bayesian_blocks_measured(
        self,
        lc: LightCurveData,
        p0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bayesian blocks segmentation using ``astropy.stats.bayesian_blocks``.

        Implements the *measures* fitness suitable for pre-binned rate data
        with Gaussian uncertainties, following the approach used for Swift
        BAT catalog light curves.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve.  ``rate_err`` values of zero are replaced by
            the minimum non-zero error to avoid division-by-zero.
        p0 : float, optional
            False-alarm probability for change-point detection.  Lower
            values produce fewer (more conservative) blocks.  Defaults to
            the value set in the analyzer config (0.01).

        Returns
        -------
        edges : np.ndarray
            Block edge positions in seconds (length ``n_blocks + 1``).
        block_rates : np.ndarray
            Mean count rate within each block (length ``n_blocks``).

        Raises
        ------
        ImportError
            If ``astropy`` is not installed.

        Notes
        -----
        Adapted from the BAT GRBCat Bayesian-blocks notebook.  The input
        arrays are sorted by time internally; ``sigma`` is halved to match
        the convention used by the catalog analysis code.
        """
        if not _check_astropy():
            raise ImportError(
                "astropy is required for bayesian_blocks_measured().  "
                "Install it with:  pip install astropy"
            )
        from astropy.stats import bayesian_blocks as astropy_bblocks

        if p0 is None:
            p0 = self.bayesian_blocks_p0

        # Sort by time
        idx = np.argsort(lc.time)
        t = lc.time[idx]
        r = lc.rate[idx]
        e = lc.rate_err[idx].copy()

        # Replace zero-error bins with the minimum non-zero error
        zero_mask = e == 0
        if zero_mask.any():
            nonzero_min = e[~zero_mask].min() if (~zero_mask).any() else 1.0
            e[zero_mask] = nonzero_min

        # Compute block edges via astropy (sigma halved per catalog convention)
        edges = astropy_bblocks(
            t, r, sigma=e / 2.0, fitness="measures", p0=p0
        )

        # Compute mean rate in each block
        block_centres = (edges[1:] + edges[:-1]) / 2.0
        block_rates = np.empty(len(block_centres))
        for i in range(len(block_centres)):
            mask = (t >= edges[i]) & (t <= edges[i + 1])
            block_rates[i] = np.mean(r[mask]) if mask.any() else 0.0

        return np.asarray(edges), block_rates

    def bayesian_blocks_custom(self, lc: LightCurveData) -> LightCurveData:
        """Bayesian blocks optimal binning (built-in implementation).

        This is a simplified, dependency-free Bayesian blocks algorithm
        following Scargle (2013).  It converts the light curve to an
        approximate event list and iteratively refines change-point edges
        using the cash-statistic fitness:

        .. math::

            F = \\sum_k N_k \\ln(N_k / T_k) - \\gamma \\, n_{\\mathrm{cp}}

        Use :meth:`bayesian_blocks_measured` when ``astropy`` is available
        for a more rigorous treatment of pre-binned measured data.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve.

        Returns
        -------
        LightCurveData
            Light curve re-binned onto the optimal block grid.
        """
        # Convert to event list format for algorithm
        times = np.repeat(lc.time, (lc.rate * lc.binsize).astype(int))

        if len(times) < 2:
            return lc.copy()

        times = np.sort(times)
        N = len(times)

        # Set penalty (ncp_prior)
        ncp_prior = self.bayesian_blocks_ncp_prior
        gamma = ncp_prior

        # Initialize
        edges = np.concatenate([[times[0]], times[:: max(1, N // 50)], [times[-1]]])
        edges = np.unique(edges)

        # Refine edges iteratively
        for iteration in range(10):
            n_edges = len(edges)
            if n_edges > 100:
                break

            best_edges = [edges[0]]
            last_edge = edges[0]

            for i in range(1, len(edges)):
                current_edge = edges[i]
                mask = (times >= best_edges[-1]) & (times <= current_edge)
                n_bin = mask.sum()
                t_bin = current_edge - best_edges[-1]

                if n_bin == 0 or t_bin == 0:
                    continue

                fitness_with = n_bin * np.log(n_bin / t_bin) - gamma

                if fitness_with > -gamma * 0.1:
                    best_edges.append(current_edge)

            best_edges = np.array(best_edges)
            if len(best_edges) == len(edges):
                break
            edges = best_edges

        # Bin with final edges
        final_edges = edges
        bin_centers = []
        bin_rates = []
        bin_errors = []

        for i in range(len(final_edges) - 1):
            mask = (lc.time >= final_edges[i]) & (lc.time < final_edges[i + 1])
            if mask.sum() == 0:
                continue

            t_center = (final_edges[i] + final_edges[i + 1]) / 2
            bin_time = final_edges[i + 1] - final_edges[i]

            total_rate = lc.rate[mask].sum()
            rate_binned = total_rate / len(mask)

            err_quad = (lc.rate_err[mask] ** 2).sum()
            err_binned = np.sqrt(err_quad) / len(mask)

            bin_centers.append(t_center)
            bin_rates.append(rate_binned)
            bin_errors.append(err_binned)

        return LightCurveData(
            time=np.array(bin_centers),
            rate=np.array(bin_rates),
            rate_err=np.array(bin_errors),
            energy_min=lc.energy_min,
            energy_max=lc.energy_max,
            binsize=np.diff(final_edges).mean(),
            mission=lc.mission,
            instrument=lc.instrument,
        )

    def bayesian_blocks(self, lc: LightCurveData) -> LightCurveData:
        """Bayesian blocks segmentation with automatic backend selection.

        Tries :meth:`bayesian_blocks_measured` (astropy) first; falls back
        to :meth:`bayesian_blocks_custom` if ``astropy`` is not available.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve.

        Returns
        -------
        LightCurveData
            Light curve re-binned onto the optimal block grid.
        """
        if _check_astropy():
            edges, block_rates = self.bayesian_blocks_measured(lc)
            # Build LightCurveData from the block representation
            block_centres = (edges[1:] + edges[:-1]) / 2.0
            block_widths = np.diff(edges)

            # Compute per-block errors from original data
            block_errs = np.empty(len(block_centres))
            idx = np.argsort(lc.time)
            t_sorted = lc.time[idx]
            e_sorted = lc.rate_err[idx]
            for i in range(len(block_centres)):
                mask = (t_sorted >= edges[i]) & (t_sorted <= edges[i + 1])
                n = mask.sum()
                if n > 0:
                    block_errs[i] = np.sqrt((e_sorted[mask] ** 2).sum()) / n
                else:
                    block_errs[i] = 0.0

            return LightCurveData(
                time=block_centres,
                rate=block_rates,
                rate_err=block_errs,
                energy_min=lc.energy_min,
                energy_max=lc.energy_max,
                binsize=float(np.mean(block_widths)),
                mission=lc.mission,
                instrument=lc.instrument,
            )
        else:
            logger.info(
                "astropy not available; falling back to custom Bayesian blocks"
            )
            return self.bayesian_blocks_custom(lc)

    # ----------------------------------------------------------------
    # Background subtraction
    # ----------------------------------------------------------------

    def background_subtract(
        self,
        lc: LightCurveData,
        bg_interval: Tuple[float, float] = (-100, -10),
    ) -> LightCurveData:
        """Subtract a constant background estimated from a pre-burst interval.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve.
        bg_interval : tuple of float
            ``(t_start, t_end)`` relative to trigger for background
            estimation.  Default ``(-100, -10)``.

        Returns
        -------
        LightCurveData
            Background-subtracted light curve (negative rates clipped to 0).
        """
        lc_out = lc.copy()

        bg_mask = (lc.time >= bg_interval[0]) & (lc.time <= bg_interval[1])
        if bg_mask.sum() == 0:
            return lc_out

        bg_rate = lc.rate[bg_mask].mean()
        bg_err = np.sqrt((lc.rate_err[bg_mask] ** 2).mean())

        lc_out.rate = lc.rate - bg_rate
        lc_out.rate[lc_out.rate < 0] = 0

        lc_out.rate_err = np.sqrt(lc.rate_err ** 2 + bg_err ** 2)
        return lc_out

    # ----------------------------------------------------------------
    # Signal metrics
    # ----------------------------------------------------------------

    def calculate_snr(self, lc: LightCurveData) -> float:
        """Calculate peak signal-to-noise ratio.

        Parameters
        ----------
        lc : LightCurveData

        Returns
        -------
        float
        """
        peak_idx = np.argmax(lc.rate)
        peak_rate = lc.rate[peak_idx]
        peak_err = lc.rate_err[peak_idx]
        if peak_err == 0:
            return np.inf
        return peak_rate / peak_err

    def find_peak(self, lc: LightCurveData) -> Tuple[float, float, float]:
        """Return ``(peak_time, peak_rate, peak_err)``."""
        peak_idx = np.argmax(lc.rate)
        return (lc.time[peak_idx], lc.rate[peak_idx], lc.rate_err[peak_idx])

    def calculate_fluence(
        self,
        lc: LightCurveData,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """Integrate the light curve to obtain fluence.

        Parameters
        ----------
        lc : LightCurveData
        time_range : tuple of float, optional
            ``(t_start, t_end)`` over which to integrate.  If *None* the
            full time range is used.

        Returns
        -------
        tuple of float
            ``(fluence, fluence_err)`` in counts cm-2 (if rates are per
            cm2) or total counts.
        """
        if time_range is not None:
            mask = (lc.time >= time_range[0]) & (lc.time <= time_range[1])
        else:
            mask = np.ones(len(lc.time), dtype=bool)

        if mask.sum() == 0:
            return 0.0, 0.0

        fluence = (lc.rate[mask] * lc.binsize).sum()
        fluence_err = np.sqrt((lc.rate_err[mask] ** 2 * lc.binsize ** 2).sum())
        return fluence, fluence_err

    # ----------------------------------------------------------------
    # Smoothing
    # ----------------------------------------------------------------

    def smooth(
        self,
        lc: LightCurveData,
        window: int = 5,
    ) -> LightCurveData:
        """Apply Savitzky-Golay smoothing to a light curve.

        Parameters
        ----------
        lc : LightCurveData
        window : int
            Smoothing window width in bins (will be forced to an odd
            number).

        Returns
        -------
        LightCurveData
        """
        lc_out = lc.copy()

        window = min(window, len(lc.rate) // 2)
        if window % 2 == 0:
            window += 1
        if window < 3:
            return lc_out

        try:
            lc_out.rate = savgol_filter(lc.rate, window, polyorder=2)
        except ValueError:
            kernel = np.ones(window) / window
            lc_out.rate = np.convolve(lc.rate, kernel, mode="same")

        lc_out.rate_err = savgol_filter(lc.rate_err, window, polyorder=2)
        return lc_out

    # ----------------------------------------------------------------
    # Multi-band merging
    # ----------------------------------------------------------------

    def merge_energy_bands(
        self,
        lightcurves: List[LightCurveData],
    ) -> LightCurveData:
        """Sum multiple energy-band light curves onto a common time grid.

        Parameters
        ----------
        lightcurves : list of LightCurveData
            Individual-band light curves.

        Returns
        -------
        LightCurveData
            Summed broadband light curve.

        Raises
        ------
        ValueError
            If *lightcurves* is empty.
        """
        if not lightcurves:
            raise ValueError("Need at least one light curve")

        ref_lc = lightcurves[0]
        common_time = ref_lc.time.copy()

        total_rate = np.zeros_like(common_time)
        total_err_sq = np.zeros_like(common_time)

        for lc in lightcurves:
            f = interpolate.interp1d(
                lc.time, lc.rate, kind="linear", bounds_error=False, fill_value=0
            )
            f_err = interpolate.interp1d(
                lc.time, lc.rate_err, kind="linear", bounds_error=False, fill_value=0
            )
            rate_interp = f(common_time)
            err_interp = f_err(common_time)

            total_rate += rate_interp
            total_err_sq += err_interp ** 2

        return LightCurveData(
            time=common_time,
            rate=total_rate,
            rate_err=np.sqrt(total_err_sq),
            energy_min=None,
            energy_max=None,
            binsize=ref_lc.binsize,
            mission=", ".join(set(lc.mission for lc in lightcurves)),
            instrument=", ".join(set(lc.instrument for lc in lightcurves)),
        )


# ===================================================================
# BAT catalog light curve I/O
# ===================================================================

def read_bat_catalog_lc(
    filepath: str,
    binsize_ms: Optional[float] = None,
) -> Dict[str, LightCurveData]:
    """Read a Swift BAT catalog ASCII light curve file.

    The file is expected to contain 9 whitespace-delimited columns::

        time  rate1 err1  rate2 err2  rate3 err3  rate4 err4

    corresponding to the four standard BAT energy bands (15-25, 25-50,
    50-100, 100-350 keV).

    Parameters
    ----------
    filepath : str or Path
        Path to the ``*_lc_ascii.dat.txt`` file.
    binsize_ms : float, optional
        Bin size in milliseconds.  If *None*, the function attempts to
        infer it from the file name (pattern ``{grbname}_{binsize}ms_...``)
        or falls back to the median time-step.

    Returns
    -------
    dict of str -> LightCurveData
        Dictionary keyed by band name (e.g. ``"15-25 keV"``), each value a
        :class:`LightCurveData` instance with ``mission="Swift"`` and
        ``instrument="BAT"``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file does not contain exactly 9 columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"BAT catalog file not found: {filepath}")

    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 9:
        raise ValueError(
            f"Expected 9 columns in BAT catalog file, got {data.shape[1]}"
        )

    time = data[:, 0]

    # Infer bin size -------------------------------------------------------
    if binsize_ms is not None:
        binsize_s = binsize_ms / 1000.0
    else:
        # Try to parse from filename  e.g. "GRB050911_64ms_lc_ascii.dat.txt"
        stem = filepath.stem  # may be "…dat" because of double extension
        name = filepath.name
        binsize_s = None
        import re
        m = re.search(r"_(\d+)ms_", name)
        if m:
            binsize_s = int(m.group(1)) / 1000.0
        if binsize_s is None:
            # Fallback: median time step
            dt = np.diff(time)
            binsize_s = float(np.median(dt[dt > 0])) if len(dt) > 0 else 1.0

    # Build per-band LightCurveData objects --------------------------------
    band_col_pairs = [
        ("15-25 keV", 1, 2),
        ("25-50 keV", 3, 4),
        ("50-100 keV", 5, 6),
        ("100-350 keV", 7, 8),
    ]

    result: Dict[str, LightCurveData] = {}
    for band_name, rate_col, err_col in band_col_pairs:
        emin, emax = BAT_BANDS[band_name]
        result[band_name] = LightCurveData(
            time=time.copy(),
            rate=data[:, rate_col].copy(),
            rate_err=data[:, err_col].copy(),
            energy_min=emin,
            energy_max=emax,
            binsize=binsize_s,
            mission="Swift",
            instrument="BAT",
        )

    return result


# ===================================================================
# BAT multi-band plotting
# ===================================================================

def plot_bat_multiband(
    bands: Dict[str, LightCurveData],
    bblocks: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    title: str = "",
    time_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 12),
    rate_label: str = "Rate (counts/s)",
):
    """Create a stacked 4-panel BAT light curve plot.

    Each panel shows one energy band as a step histogram with error bars.
    If Bayesian block representations are supplied they are overlaid as
    coloured step functions.

    Parameters
    ----------
    bands : dict of str -> LightCurveData
        Band light curves keyed by band name.
    bblocks : dict of str -> (edges, rates), optional
        Bayesian block representations per band, as returned by
        :meth:`LightCurveAnalyzer.bayesian_blocks_measured`.
    title : str
        Super-title for the figure.
    time_range : tuple of float, optional
        ``(t_start, t_end)`` to restrict the x-axis.
    figsize : tuple of float
        Figure size in inches ``(width, height)``.
    rate_label : str
        Y-axis label.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plot_bat_multiband().  "
            "Install it with:  pip install matplotlib"
        )
    import matplotlib.pyplot as plt

    band_order = [b for b in BAT_BAND_NAMES if b in bands]
    n_panels = len(band_order)
    if n_panels == 0:
        raise ValueError("No recognised BAT bands found in *bands* dict")

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.05},
    )
    if n_panels == 1:
        axes = np.array([axes])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (band_name, ax) in enumerate(zip(band_order, axes)):
        lc = bands[band_name]
        color = colors[idx % len(colors)]

        # Light curve with error bars
        ax.errorbar(
            lc.time, lc.rate, yerr=lc.rate_err,
            fmt="none", ecolor="gray", elinewidth=0.5, alpha=0.4,
        )
        ax.step(lc.time, lc.rate, where="mid", color=color, linewidth=0.8,
                alpha=0.7, label=band_name)

        # Bayesian blocks overlay
        if bblocks is not None and band_name in bblocks:
            edges, block_rates = bblocks[band_name]
            # Draw step function for blocks
            for i in range(len(block_rates)):
                ax.hlines(
                    block_rates[i], edges[i], edges[i + 1],
                    colors="black", linewidths=1.5,
                )
                # Vertical connectors
                if i > 0:
                    ax.vlines(
                        edges[i], block_rates[i - 1], block_rates[i],
                        colors="black", linewidths=1.5,
                    )

        ax.set_ylabel(rate_label, fontsize=9)
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(labelsize=8)

        if time_range is not None:
            ax.set_xlim(time_range)

    axes[-1].set_xlabel("Time since trigger (s)", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=13, y=0.95)

    fig.align_ylabels(axes)
    plt.tight_layout()
    return fig, axes


# ===================================================================
# BATLightCurve -- high-level BAT-specific wrapper
# ===================================================================

class BATLightCurve:
    """Swift BAT catalog light curve: multi-band I/O, Bayesian blocks, and plotting.

    This class ties together the BAT-specific functionality in a convenient
    object-oriented interface.

    Parameters
    ----------
    bands : dict of str -> LightCurveData
        Per-band light curves, typically produced by :func:`read_bat_catalog_lc`.
    grb_name : str, optional
        GRB identifier (e.g. ``"GRB050911"``).
    config : dict, optional
        Analysis configuration forwarded to :class:`LightCurveAnalyzer`.

    Examples
    --------
    >>> bat = BATLightCurve.from_catalog_file("GRB050911_64ms_lc_ascii.dat.txt")
    >>> bat.run_bayesian_blocks()
    >>> fig, axes = bat.plot()
    """

    def __init__(
        self,
        bands: Dict[str, LightCurveData],
        grb_name: str = "",
        config: Dict = None,
    ):
        self.bands = bands
        self.grb_name = grb_name
        self.analyzer = LightCurveAnalyzer(config=config)
        self.bblocks: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

    # ----------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------

    @classmethod
    def from_catalog_file(
        cls,
        filepath: str,
        grb_name: str = "",
        binsize_ms: Optional[float] = None,
        config: Dict = None,
    ) -> "BATLightCurve":
        """Create a :class:`BATLightCurve` from a BAT catalog ASCII file.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``*_lc_ascii.dat.txt`` file.
        grb_name : str, optional
            GRB name.  If empty, the function tries to infer it from the
            file name.
        binsize_ms : float, optional
            Bin size in milliseconds (auto-detected if *None*).
        config : dict, optional
            Analyzer configuration.

        Returns
        -------
        BATLightCurve
        """
        filepath = Path(filepath)
        bands = read_bat_catalog_lc(filepath, binsize_ms=binsize_ms)

        if not grb_name:
            # Try to extract from filename, e.g. "GRB050911_64ms_..."
            import re
            m = re.match(r"(GRB\d+[A-Za-z]*)", filepath.name)
            if m:
                grb_name = m.group(1)

        return cls(bands=bands, grb_name=grb_name, config=config)

    # ----------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------

    def run_bayesian_blocks(
        self,
        p0: Optional[float] = None,
        band_names: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run Bayesian blocks on each energy band.

        Uses :meth:`LightCurveAnalyzer.bayesian_blocks_measured` when
        ``astropy`` is available; otherwise falls back to the custom
        implementation wrapped via
        :meth:`LightCurveAnalyzer.bayesian_blocks`.

        Parameters
        ----------
        p0 : float, optional
            False-alarm probability (passed to ``bayesian_blocks_measured``).
        band_names : list of str, optional
            Subset of bands to process.  If *None*, all bands are processed.

        Returns
        -------
        dict of str -> (edges, block_rates)
            Results keyed by band name.  Also stored in ``self.bblocks``.
        """
        if band_names is None:
            band_names = list(self.bands.keys())

        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in band_names:
            if name not in self.bands:
                logger.warning("Band '%s' not found; skipping.", name)
                continue

            lc = self.bands[name]

            if _check_astropy():
                edges, rates = self.analyzer.bayesian_blocks_measured(lc, p0=p0)
                results[name] = (edges, rates)
            else:
                # Fallback: use the custom implementation, then extract
                # edges + rates from the returned LightCurveData
                bb_lc = self.analyzer.bayesian_blocks_custom(lc)
                half_bins = np.diff(bb_lc.time)
                if len(half_bins) == 0:
                    edges = np.array([lc.time[0], lc.time[-1]])
                    rates = np.array([lc.rate.mean()])
                else:
                    # Reconstruct approximate edges from bin centres
                    edges = np.empty(len(bb_lc.time) + 1)
                    edges[0] = bb_lc.time[0] - half_bins[0] / 2 if len(half_bins) > 0 else bb_lc.time[0]
                    edges[-1] = bb_lc.time[-1] + half_bins[-1] / 2 if len(half_bins) > 0 else bb_lc.time[-1]
                    for i in range(1, len(bb_lc.time)):
                        edges[i] = (bb_lc.time[i - 1] + bb_lc.time[i]) / 2.0
                    rates = bb_lc.rate
                results[name] = (edges, rates)

            logger.info(
                "Band %s: %d Bayesian blocks", name, len(results[name][1])
            )

        self.bblocks = results
        return results

    def get_broadband(self) -> LightCurveData:
        """Return a summed broadband (15-350 keV) light curve.

        Returns
        -------
        LightCurveData
        """
        return self.analyzer.merge_energy_bands(list(self.bands.values()))

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------

    def plot(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (10, 12),
        show: bool = False,
    ):
        """Plot stacked 4-band BAT light curves with Bayesian blocks overlay.

        Parameters
        ----------
        time_range : tuple of float, optional
            ``(t_start, t_end)`` for x-axis limits.
        figsize : tuple of float
            Figure dimensions.
        show : bool
            If *True*, call ``plt.show()`` before returning.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray of matplotlib.axes.Axes
        """
        title = self.grb_name if self.grb_name else "BAT Multi-Band Light Curve"
        fig, axes = plot_bat_multiband(
            bands=self.bands,
            bblocks=self.bblocks,
            title=title,
            time_range=time_range,
            figsize=figsize,
        )
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig, axes

    # ----------------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------------

    @property
    def band_names(self) -> List[str]:
        """Names of loaded energy bands."""
        return list(self.bands.keys())

    @property
    def time(self) -> np.ndarray:
        """Time array from the first available band."""
        return next(iter(self.bands.values())).time

    def __repr__(self) -> str:
        n_bands = len(self.bands)
        n_bins = len(self.time) if self.bands else 0
        bb_info = f", bblocks={bool(self.bblocks)}" if self.bblocks else ""
        return (
            f"BATLightCurve(grb={self.grb_name!r}, bands={n_bands}, "
            f"bins={n_bins}{bb_info})"
        )
