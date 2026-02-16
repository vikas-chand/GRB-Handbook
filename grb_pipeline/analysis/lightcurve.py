"""Light curve analysis module for GRB events."""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import minimize


@dataclass
class LightCurveData:
    """Container for light curve data."""
    time: np.ndarray  # Time in seconds
    rate: np.ndarray  # Count rate in counts/s
    rate_err: np.ndarray  # Statistical error
    energy_min: float = None  # Energy range minimum (keV)
    energy_max: float = None  # Energy range maximum (keV)
    binsize: float = 1.0  # Time bin size (seconds)
    mission: str = "Unknown"  # Mission name
    instrument: str = "Unknown"  # Instrument name

    def copy(self):
        """Create a copy of the light curve."""
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


class LightCurveAnalyzer:
    """Analyzer for GRB light curves."""

    def __init__(self, config: Dict = None):
        """
        Initialize the light curve analyzer.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with analysis parameters
        """
        self.config = config or {}
        self.bayesian_blocks_ncp_prior = self.config.get('bayesian_blocks_ncp_prior', 4)

    def extract_lightcurve(
        self,
        event_data: np.ndarray,
        energy_range: Optional[Tuple[float, float]] = None,
        binsize: float = 1.0,
        mission: str = "Unknown",
        instrument: str = "Unknown",
    ) -> LightCurveData:
        """
        Extract light curve from event data by binning.

        Parameters
        ----------
        event_data : np.ndarray
            Event arrival times (shape: (n_events,) or (n_events, 2) with energies)
        energy_range : tuple, optional
            Energy range (keV) to include. If None, includes all events.
        binsize : float
            Time bin size in seconds (default: 1.0)
        mission : str
            Mission name
        instrument : str
            Instrument name

        Returns
        -------
        LightCurveData
            Binned light curve data
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
        """
        Rebin light curve to different time resolution.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        new_binsize : float
            New time bin size in seconds

        Returns
        -------
        LightCurveData
            Rebinned light curve
        """
        # Create new bin edges aligned to original
        factor = int(np.round(new_binsize / lc.binsize))
        if factor < 1:
            raise ValueError("Cannot rebin to smaller size")

        # Group and sum
        n_new = len(lc.time) // factor
        time_new = lc.time[:n_new * factor].reshape(n_new, factor).mean(axis=1)
        rate_new = lc.rate[:n_new * factor].reshape(n_new, factor).sum(axis=1) / factor

        # Propagate errors (sum in quadrature, then convert back to rate)
        rate_err_squared = lc.rate_err[:n_new * factor].reshape(n_new, factor) ** 2
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

    def bayesian_blocks(self, lc: LightCurveData) -> LightCurveData:
        """
        Optimal binning using Bayesian blocks algorithm (Scargle 2013).

        Finds optimal change points that maximize the fitness function:
        F = sum(N_k * log(N_k/T_k)) - penalty * ncp

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        LightCurveData
            Light curve with Bayesian blocks binning
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
        edges = np.concatenate([[times[0]], times[::max(1, N // 50)], [times[-1]]])
        edges = np.unique(edges)

        # Refine edges iteratively
        for iteration in range(10):
            n_edges = len(edges)
            if n_edges > 100:  # Limit complexity
                break

            # Calculate fitness for each bin
            best_edges = [edges[0]]
            last_edge = edges[0]

            for i in range(1, len(edges)):
                current_edge = edges[i]
                mask = (times >= best_edges[-1]) & (times <= current_edge)
                n_bin = mask.sum()
                t_bin = current_edge - best_edges[-1]

                if n_bin == 0 or t_bin == 0:
                    continue

                # Fitness without this bin
                fitness_without = 0
                # Fitness with this bin
                if n_bin > 0:
                    fitness_with = n_bin * np.log(n_bin / t_bin) - gamma
                else:
                    fitness_with = -gamma

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

            # Sum rates in this bin
            total_rate = lc.rate[mask].sum()
            rate_binned = total_rate / len(mask)

            # Error propagation
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

    def background_subtract(
        self,
        lc: LightCurveData,
        bg_interval: Tuple[float, float] = (-100, -10),
    ) -> LightCurveData:
        """
        Subtract background estimated from pre-burst interval.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        bg_interval : tuple
            Time interval (start, end) relative to trigger time for background
            estimation. Default (-100, -10) means 100-10 seconds before burst.

        Returns
        -------
        LightCurveData
            Background-subtracted light curve
        """
        lc_out = lc.copy()

        # Find background region
        bg_mask = (lc.time >= bg_interval[0]) & (lc.time <= bg_interval[1])

        if bg_mask.sum() == 0:
            return lc_out

        # Estimate background (mean rate in pre-burst)
        bg_rate = lc.rate[bg_mask].mean()
        bg_err = np.sqrt((lc.rate_err[bg_mask] ** 2).mean())

        # Subtract background
        lc_out.rate = lc.rate - bg_rate
        lc_out.rate[lc_out.rate < 0] = 0  # Clip negative rates

        # Propagate errors
        lc_out.rate_err = np.sqrt(lc.rate_err ** 2 + bg_err ** 2)

        return lc_out

    def calculate_snr(self, lc: LightCurveData) -> float:
        """
        Calculate signal-to-noise ratio of light curve.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        float
            Signal-to-noise ratio
        """
        # Find peak
        peak_idx = np.argmax(lc.rate)
        peak_rate = lc.rate[peak_idx]
        peak_err = lc.rate_err[peak_idx]

        if peak_err == 0:
            return np.inf

        snr = peak_rate / peak_err
        return snr

    def find_peak(self, lc: LightCurveData) -> Tuple[float, float, float]:
        """
        Find light curve peak.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        tuple
            (peak_time, peak_rate, peak_err)
        """
        peak_idx = np.argmax(lc.rate)
        return (lc.time[peak_idx], lc.rate[peak_idx], lc.rate_err[peak_idx])

    def calculate_fluence(
        self,
        lc: LightCurveData,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Calculate fluence by integrating the light curve.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        time_range : tuple, optional
            Time interval to integrate. If None, uses full range.

        Returns
        -------
        tuple
            (fluence, fluence_err) in counts/cm^2
        """
        if time_range is not None:
            mask = (lc.time >= time_range[0]) & (lc.time <= time_range[1])
        else:
            mask = np.ones(len(lc.time), dtype=bool)

        if mask.sum() == 0:
            return 0.0, 0.0

        # Integrate rate * dt
        fluence = (lc.rate[mask] * lc.binsize).sum()
        fluence_err = np.sqrt((lc.rate_err[mask] ** 2 * lc.binsize ** 2).sum())

        return fluence, fluence_err

    def smooth(
        self,
        lc: LightCurveData,
        window: int = 5,
    ) -> LightCurveData:
        """
        Apply moving average smoothing to light curve.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        window : int
            Window size for smoothing (default: 5 bins)

        Returns
        -------
        LightCurveData
            Smoothed light curve
        """
        lc_out = lc.copy()

        # Ensure window is odd and within bounds
        window = min(window, len(lc.rate) // 2)
        if window % 2 == 0:
            window += 1

        if window < 3:
            return lc_out

        # Apply Savitzky-Golay filter for better preservation
        try:
            lc_out.rate = savgol_filter(lc.rate, window, polyorder=2)
        except ValueError:
            # Fall back to simple moving average
            kernel = np.ones(window) / window
            lc_out.rate = np.convolve(lc.rate, kernel, mode='same')

        # Error increases with smoothing
        lc_out.rate_err = savgol_filter(lc.rate_err, window, polyorder=2)

        return lc_out

    def merge_energy_bands(
        self,
        lightcurves: List[LightCurveData],
    ) -> LightCurveData:
        """
        Combine multi-band light curve data.

        Parameters
        ----------
        lightcurves : list of LightCurveData
            Light curves in different energy bands

        Returns
        -------
        LightCurveData
            Combined light curve (all rates summed)
        """
        if not lightcurves:
            raise ValueError("Need at least one light curve")

        # Align all to coarsest binning
        ref_lc = lightcurves[0]
        max_binsize = max(lc.binsize for lc in lightcurves)

        # Interpolate all to common grid
        common_time = ref_lc.time.copy()

        total_rate = np.zeros_like(common_time)
        total_err_sq = np.zeros_like(common_time)

        for lc in lightcurves:
            # Interpolate this band to common time grid
            f = interpolate.interp1d(
                lc.time, lc.rate,
                kind='linear', bounds_error=False, fill_value=0
            )
            f_err = interpolate.interp1d(
                lc.time, lc.rate_err,
                kind='linear', bounds_error=False, fill_value=0
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
