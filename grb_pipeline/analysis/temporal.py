"""Temporal analysis module for GRB light curves.

Includes standard timing analysis (T90, T50, variability, autocorrelation,
pulse fitting) and the Band (1997) Discrete Cross-Correlation Function (DCCF)
for spectral lag measurement with Monte Carlo error estimation.

References
----------
Band, D. L. 1997, ApJ, 486, 928
    "Postpeak Decline of Gamma-Ray Burst Spectra"
    Defines the discrete cross-correlation function for transient sources.
"""

from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
import numpy as np
from scipy import signal, optimize, stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings

from .lightcurve import LightCurveData


@dataclass
class SpectralLagResult:
    """Container for spectral lag analysis results.

    Attributes
    ----------
    lag : float
        Best-fit spectral lag in seconds (peak of asymmetric Gaussian
        fitted to the observed DCCF).
    lag_err : float
        1-sigma uncertainty on the lag, derived from the standard deviation
        of the Monte Carlo lag distribution.
    offsets : np.ndarray
        Time offset array for the DCCF (seconds).
    ccf : np.ndarray
        Observed discrete cross-correlation function values.
    ccf_err : np.ndarray
        1-sigma CCF uncertainties from Monte Carlo noise simulations.
    fit_params : Dict
        Asymmetric Gaussian fit parameters:
        ``{'const', 'amplitude', 'mu', 'sigma1', 'sigma2'}``.
    n_simulations : Dict
        Number of Monte Carlo simulations used:
        ``{'ccf_error': int, 'lag_error': int}``.
    """
    lag: float
    lag_err: float
    offsets: np.ndarray
    ccf: np.ndarray
    ccf_err: np.ndarray
    fit_params: Dict = field(default_factory=dict)
    n_simulations: Dict = field(default_factory=dict)


class TemporalAnalyzer:
    """Analyzer for temporal properties of GRB light curves.

    Provides standard timing metrics (T90, T50, peak flux, variability,
    autocorrelation, FRED pulse fitting) and the Band (1997) discrete
    cross-correlation function for spectral lag measurement.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize temporal analyzer.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}

    # ------------------------------------------------------------------
    # Duration and basic timing
    # ------------------------------------------------------------------

    def calculate_t90(
        self,
        lc: LightCurveData,
        confidence: float = 0.9,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate T90 duration using cumulative distribution.

        Algorithm: Compute cumulative counts, find times at 5% and 95% of total.
        Error estimated via bootstrap resampling.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        confidence : float
            Confidence interval for duration (default: 0.9 = T90)

        Returns
        -------
        tuple
            (t90, t90_err, t90_start, t90_end)
        """
        # Compute cumulative counts (background-independent)
        cumsum = np.cumsum(lc.rate * lc.binsize)
        cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]

        # Find 5% and 95% points
        lower_frac = (1 - confidence) / 2
        upper_frac = 1 - lower_frac

        # Interpolate to find times
        t_lower = np.interp(lower_frac, cumsum, lc.time)
        t_upper = np.interp(upper_frac, cumsum, lc.time)

        t90 = t_upper - t_lower

        # Bootstrap error estimate
        n_bootstrap = 100
        t90_samples = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(len(lc.rate), len(lc.rate), replace=True)
            rate_resamp = lc.rate[idx]

            cumsum_resamp = np.cumsum(rate_resamp * lc.binsize)
            if cumsum_resamp[-1] > 0:
                cumsum_resamp = cumsum_resamp / cumsum_resamp[-1]
                t_l = np.interp(lower_frac, cumsum_resamp, lc.time[idx])
                t_u = np.interp(upper_frac, cumsum_resamp, lc.time[idx])
                t90_samples.append(t_u - t_l)

        if t90_samples:
            t90_err = np.std(t90_samples)
        else:
            t90_err = 0.0

        return t90, t90_err, t_lower, t_upper

    def calculate_t50(self, lc: LightCurveData) -> Tuple[float, float, float, float]:
        """
        Calculate T50 duration (central 50% of emission).

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        tuple
            (t50, t50_err, t50_start, t50_end)
        """
        return self.calculate_duration(lc, fraction=0.5)

    def calculate_duration(
        self,
        lc: LightCurveData,
        fraction: float = 0.9,
    ) -> Tuple[float, float, float, float]:
        """
        Generic Tx duration calculation.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        fraction : float
            Fraction of counts to include (default: 0.9 for T90)

        Returns
        -------
        tuple
            (duration, duration_err, start_time, end_time)
        """
        # Compute cumulative distribution
        counts = lc.rate * lc.binsize
        cumsum = np.cumsum(counts)
        cumsum = cumsum / cumsum[-1]

        # Find times containing 'fraction' of counts
        lower_frac = (1 - fraction) / 2
        upper_frac = 1 - lower_frac

        t_start = np.interp(lower_frac, cumsum, lc.time)
        t_end = np.interp(upper_frac, cumsum, lc.time)
        duration = t_end - t_start

        # Error from counting statistics
        duration_err = duration * np.sqrt(
            2 / (counts.sum())
        )  # Rough estimate

        return duration, duration_err, t_start, t_end

    # ------------------------------------------------------------------
    # Peak flux and variability
    # ------------------------------------------------------------------

    def peak_flux(
        self,
        lc: LightCurveData,
        binsize: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate peak flux in specified binning.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        binsize : float
            Bin size for peak finding (default: 1.0 s)

        Returns
        -------
        tuple
            (peak_flux, peak_flux_err, peak_time)
        """
        if binsize != lc.binsize:
            # Rebin if necessary
            from .lightcurve import LightCurveAnalyzer
            analyzer = LightCurveAnalyzer()
            lc_rebin = analyzer.rebin(lc, binsize)
        else:
            lc_rebin = lc

        peak_idx = np.argmax(lc_rebin.rate)
        peak_rate = lc_rebin.rate[peak_idx]
        peak_err = lc_rebin.rate_err[peak_idx]
        peak_time = lc_rebin.time[peak_idx]

        return peak_rate, peak_err, peak_time

    def variability_index(self, lc: LightCurveData) -> float:
        """
        Calculate variability index V = (F_max - F_min) / (F_max + F_min).

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        float
            Variability index in range [0, 1]
        """
        f_max = lc.rate.max()
        f_min = lc.rate.min()

        if f_max + f_min == 0:
            return 0.0

        v = (f_max - f_min) / (f_max + f_min)
        return np.clip(v, 0, 1)

    def fractional_rms(self, lc: LightCurveData) -> float:
        """
        Calculate fractional RMS variability.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        float
            Fractional RMS = sqrt(variance - mean_error^2) / mean_rate
        """
        mean_rate = lc.rate.mean()
        variance = lc.rate.var()
        mean_err_sq = (lc.rate_err ** 2).mean()

        if variance <= mean_err_sq or mean_rate == 0:
            return 0.0

        rms_var = variance - mean_err_sq
        frms = np.sqrt(rms_var) / mean_rate

        return frms

    # ------------------------------------------------------------------
    # Autocorrelation and variability timescale
    # ------------------------------------------------------------------

    def autocorrelation(
        self,
        lc: LightCurveData,
        max_lag: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate normalized autocorrelation function.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        max_lag : int, optional
            Maximum lag to compute (default: 1/4 of data length)

        Returns
        -------
        tuple
            (lags, acf) - lag times and autocorrelation values
        """
        if max_lag is None:
            max_lag = len(lc.rate) // 4

        # Normalize light curve
        rate_norm = lc.rate - lc.rate.mean()

        # Compute autocorrelation using FFT
        acf = signal.correlate(rate_norm, rate_norm, mode='full')
        acf = acf[len(acf) // 2 :]
        acf = acf / acf[0]  # Normalize to 1 at lag 0

        lags = np.arange(len(acf[:max_lag])) * lc.binsize

        return lags, acf[:max_lag]

    def minimum_variability_timescale(self, lc: LightCurveData) -> float:
        """
        Shortest timescale of significant variability.

        Uses autocorrelation to find when ACF drops below 0.5.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        float
            Minimum variability timescale in seconds
        """
        lags, acf = self.autocorrelation(lc)

        # Find first lag where ACF < 0.5
        below_half = np.where(acf < 0.5)[0]

        if len(below_half) > 0:
            min_timescale = lags[below_half[0]]
        else:
            min_timescale = lags[-1]

        return min_timescale

    # ------------------------------------------------------------------
    # Pulse fitting
    # ------------------------------------------------------------------

    def fit_fred_pulse(
        self,
        lc: LightCurveData,
        n_pulses: int = 1,
    ) -> Dict:
        """
        Fit Fast Rise Exponential Decay (FRED) pulse model.

        Model: F(t) = A * exp(-tau1/(t-ts) - (t-ts)/tau2) for t > ts
        Uses scipy.optimize.curve_fit for parameter estimation.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve
        n_pulses : int
            Number of pulses to fit (default: 1)

        Returns
        -------
        dict
            Parameters for each pulse: {amplitude, tau_rise, tau_decay, t_start, amplitude_err, ...}
        """

        def fred_model(t, *params):
            """FRED model function."""
            if n_pulses == 1:
                A, tau1, tau2, ts = params
                mask = t > ts
                f = np.zeros_like(t)
                if mask.sum() > 0:
                    dt = t[mask] - ts
                    f[mask] = A * np.exp(-tau1 / dt - dt / tau2)
                return f
            else:
                # Multiple pulses
                f = np.zeros_like(t)
                for i in range(n_pulses):
                    A = params[i * 4]
                    tau1 = params[i * 4 + 1]
                    tau2 = params[i * 4 + 2]
                    ts = params[i * 4 + 3]
                    mask = t > ts
                    if mask.sum() > 0:
                        dt = t[mask] - ts
                        f[mask] += A * np.exp(-tau1 / dt - dt / tau2)
                return f

        # Initial guess
        peak_idx = np.argmax(lc.rate)
        peak_rate = lc.rate[peak_idx]
        peak_time = lc.time[peak_idx]

        if n_pulses == 1:
            p0 = [peak_rate, peak_time * 0.1, peak_time * 0.5, peak_time * 0.5]
            bounds = (
                [0, 1e-4, 1e-4, peak_time * 0.1],
                [peak_rate * 10, peak_time, peak_time * 2, peak_time * 0.9],
            )
        else:
            p0 = []
            for _ in range(n_pulses):
                p0.extend([peak_rate / n_pulses, peak_time * 0.1, peak_time * 0.5, peak_time])
            bounds = None

        try:
            popt, pcov = optimize.curve_fit(
                fred_model, lc.time, lc.rate,
                p0=p0, sigma=lc.rate_err, bounds=bounds,
                maxfev=10000
            )

            # Compute residuals
            residuals = lc.rate - fred_model(lc.time, *popt)
            chi_sq = (residuals / lc.rate_err) ** 2
            chi_sq = chi_sq[~np.isnan(chi_sq)].sum()

            result = {
                'chi_sq': chi_sq,
                'dof': len(lc.rate) - len(popt),
                'n_pulses': n_pulses,
            }

            for i in range(n_pulses):
                result[f'amplitude_{i}'] = popt[i * 4]
                result[f'tau_rise_{i}'] = popt[i * 4 + 1]
                result[f'tau_decay_{i}'] = popt[i * 4 + 2]
                result[f't_start_{i}'] = popt[i * 4 + 3]

                if pcov is not None and np.isfinite(pcov).all():
                    result[f'amplitude_{i}_err'] = np.sqrt(pcov[i * 4, i * 4])
                    result[f'tau_rise_{i}_err'] = np.sqrt(pcov[i * 4 + 1, i * 4 + 1])
                    result[f'tau_decay_{i}_err'] = np.sqrt(pcov[i * 4 + 2, i * 4 + 2])
                    result[f't_start_{i}_err'] = np.sqrt(pcov[i * 4 + 3, i * 4 + 3])

            return result

        except Exception as e:
            return {'error': str(e), 'n_pulses': n_pulses}

    def detect_pulses(self, lc: LightCurveData) -> List[Dict]:
        """
        Detect pulse peaks and boundaries in light curve.

        Parameters
        ----------
        lc : LightCurveData
            Input light curve

        Returns
        -------
        list of dict
            Pulse information: {peak_time, peak_rate, start_time, end_time, width}
        """
        # Find local maxima
        peaks, properties = signal.find_peaks(
            lc.rate, height=lc.rate.mean(), distance=1
        )

        pulses = []
        for peak_idx in peaks:
            peak_rate = lc.rate[peak_idx]
            peak_time = lc.time[peak_idx]

            # Find pulse boundaries (where rate drops to half-max on each side)
            half_max = peak_rate / 2

            # Find rising edge
            left_idx = peak_idx
            while left_idx > 0 and lc.rate[left_idx - 1] < half_max:
                left_idx -= 1
            start_time = lc.time[left_idx]

            # Find falling edge
            right_idx = peak_idx
            while right_idx < len(lc.rate) - 1 and lc.rate[right_idx + 1] < half_max:
                right_idx += 1
            end_time = lc.time[right_idx]

            pulses.append({
                'peak_time': peak_time,
                'peak_rate': peak_rate,
                'start_time': start_time,
                'end_time': end_time,
                'width': end_time - start_time,
            })

        return pulses

    # ------------------------------------------------------------------
    # Hardness ratio and simple spectral lag
    # ------------------------------------------------------------------

    def hardness_ratio(
        self,
        lc_hard: LightCurveData,
        lc_soft: LightCurveData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate hardness ratio between energy bands.

        HR = (Hard - Soft) / (Hard + Soft)

        Parameters
        ----------
        lc_hard : LightCurveData
            Hard energy band light curve
        lc_soft : LightCurveData
            Soft energy band light curve

        Returns
        -------
        tuple
            (hardness_ratio, hardness_ratio_err)
        """
        # Interpolate to common time grid
        f_hard = interp1d(
            lc_hard.time, lc_hard.rate,
            kind='linear', bounds_error=False, fill_value=0
        )
        f_soft = interp1d(
            lc_soft.time, lc_soft.rate,
            kind='linear', bounds_error=False, fill_value=0
        )

        f_hard_err = interp1d(
            lc_hard.time, lc_hard.rate_err,
            kind='linear', bounds_error=False, fill_value=0
        )
        f_soft_err = interp1d(
            lc_soft.time, lc_soft.rate_err,
            kind='linear', bounds_error=False, fill_value=0
        )

        # Use coarser time grid
        time_common = lc_hard.time if len(lc_hard.time) < len(lc_soft.time) else lc_soft.time

        hard = f_hard(time_common)
        soft = f_soft(time_common)
        hard_err = f_hard_err(time_common)
        soft_err = f_soft_err(time_common)

        # Calculate HR
        denominator = hard + soft
        mask = denominator > 0

        hr = np.zeros_like(hard)
        hr[mask] = (hard[mask] - soft[mask]) / denominator[mask]

        # Error propagation
        hr_err = np.zeros_like(hard)
        hr_err[mask] = np.sqrt(
            (4 * soft[mask] ** 2 * hard_err[mask] ** 2 +
             4 * hard[mask] ** 2 * soft_err[mask] ** 2) /
            denominator[mask] ** 4
        )

        return hr, hr_err

    def spectral_lag_simple(
        self,
        lc_high: LightCurveData,
        lc_low: LightCurveData,
    ) -> Tuple[float, float]:
        """
        Calculate spectral lag via scipy cross-correlation (simple method).

        This is a quick estimate using scipy.signal.correlate.  For a proper
        Band (1997) DCCF analysis with Monte Carlo errors, use
        ``compute_spectral_lag`` instead.

        Parameters
        ----------
        lc_high : LightCurveData
            High energy band light curve
        lc_low : LightCurveData
            Low energy band light curve

        Returns
        -------
        tuple
            (lag, lag_err) in seconds
        """
        # Align time grids
        t_min = max(lc_high.time.min(), lc_low.time.min())
        t_max = min(lc_high.time.max(), lc_low.time.max())

        mask_h = (lc_high.time >= t_min) & (lc_high.time <= t_max)
        mask_l = (lc_low.time >= t_min) & (lc_low.time <= t_max)

        high = lc_high.rate[mask_h]
        low = lc_low.rate[mask_l]

        if len(high) < 2 or len(low) < 2:
            return 0.0, 0.0

        # Normalize
        high = (high - high.mean()) / high.std()
        low = (low - low.mean()) / low.std()

        # Cross-correlation
        lag_bins = int(min(10, len(high) // 4))
        xcorr = signal.correlate(high, low, mode='same')
        lags = np.arange(len(xcorr)) - len(xcorr) // 2

        # Find peak
        peak_idx = np.argmax(xcorr)
        lag = lags[peak_idx] * lc_high.binsize

        # Estimate error (width of cross-correlation peak)
        xcorr_norm = xcorr / xcorr.max()
        above_half = np.where(xcorr_norm > 0.5)[0]
        if len(above_half) > 0:
            lag_err = (lags[above_half[-1]] - lags[above_half[0]]) * lc_high.binsize / 2
        else:
            lag_err = lc_high.binsize

        return lag, lag_err

    # ==================================================================
    # Band (1997) Discrete Cross-Correlation Function (DCCF) and
    # Monte Carlo spectral lag estimation
    # ==================================================================

    def discrete_cross_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Band (1997) Discrete Cross-Correlation Function for transient sources.

        Computes the un-binned DCCF between two background-subtracted light
        curves sampled on the same uniform time grid.  The normalisation
        follows Band (1997) Eq. 1:

            CCF(k) = sum_i x(i) * y(i+k) / sqrt( sum x^2 * sum y^2 )

        where the summation indices are adjusted for each lag *k* so that
        only overlapping samples contribute.

        Parameters
        ----------
        x : np.ndarray
            Background-subtracted count-rate array for the first energy band.
        y : np.ndarray
            Background-subtracted count-rate array for the second energy band.
            Must have the same length and time-step as *x*.
        t : np.ndarray
            Uniform time array corresponding to both light curves.

        Returns
        -------
        offsets : np.ndarray
            Lag values in the same time units as *t*, from ``-N*dt`` to
            ``+N*dt`` inclusive.
        ccf : np.ndarray
            Normalised cross-correlation values at each offset.

        Notes
        -----
        Both *x* and *y* must already be background-subtracted before calling
        this method.  The time array *t* must be uniformly spaced.
        """
        tstep = t[1] - t[0]
        N = len(t)
        lag_k = np.arange(-N, N + 1, 1)
        offsets = lag_k * tstep

        denominator = np.sqrt(
            np.sum(np.square(x)) * np.sum(np.square(y))
        )

        if denominator == 0.0:
            return offsets, np.zeros_like(offsets, dtype=float)

        ccf = np.empty(len(lag_k), dtype=float)
        for idx, k in enumerate(range(-N, N + 1)):
            # Summation bounds (1-based in Band's notation, 0-based here)
            i_start = max(0, -k)
            i_end = min(N, N - k)
            if i_start >= i_end:
                ccf[idx] = 0.0
            else:
                ccf[idx] = np.sum(
                    x[i_start:i_end] * y[i_start + k:i_end + k]
                ) / denominator

        return offsets, ccf

    @staticmethod
    def asymmetric_gaussian(
        x: np.ndarray,
        const: float,
        amplitude: float,
        mu: float,
        sigma1: float,
        sigma2: float,
    ) -> np.ndarray:
        """
        Asymmetric (split-normal) Gaussian used to fit the DCCF peak.

        .. math::

            f(x) = \\begin{cases}
                C + A \\exp\\!\\bigl(-\\tfrac{(x-\\mu)^2}{2\\sigma_1^2}\\bigr)
                    & x < \\mu \\\\
                C + A \\exp\\!\\bigl(-\\tfrac{(x-\\mu)^2}{2\\sigma_2^2}\\bigr)
                    & x \\ge \\mu
            \\end{cases}

        Parameters
        ----------
        x : array_like
            Independent variable (lag offsets).
        const : float
            Constant baseline offset.
        amplitude : float
            Peak amplitude above the baseline.
        mu : float
            Location of the peak (the spectral lag).
        sigma1 : float
            Width parameter for the left (x < mu) side.
        sigma2 : float
            Width parameter for the right (x >= mu) side.

        Returns
        -------
        np.ndarray
            Model values evaluated at each element of *x*.
        """
        x = np.asarray(x, dtype=float)
        result = np.empty_like(x)
        left = x < mu
        right = ~left
        result[left] = const + amplitude * np.exp(
            -((x[left] - mu) / sigma1) ** 2 / 2.0
        )
        result[right] = const + amplitude * np.exp(
            -((x[right] - mu) / sigma2) ** 2 / 2.0
        )
        return result

    def compute_spectral_lag(
        self,
        lc_low: LightCurveData,
        lc_high: LightCurveData,
        tstart: float,
        tstop: float,
        n_ccf_sims: int = 10000,
        n_lag_sims: int = 1000,
        data_type: str = 'swift',
    ) -> SpectralLagResult:
        """
        Full Band (1997) DCCF spectral-lag pipeline with Monte Carlo errors.

        The procedure has three stages:

        1. **Observed DCCF** -- Trim both light curves to ``[tstart, tstop]``,
           subtract backgrounds (mean of the rate), and compute the discrete
           cross-correlation function.

        2. **CCF error estimation** (``n_ccf_sims`` iterations, default 10 000)
           -- For each realisation, add random noise to the background-subtracted
           light curves, recompute the DCCF, and record it.  The standard
           deviation across realisations gives the 1-sigma CCF uncertainty at
           each lag bin.  Noise model depends on ``data_type``:

           * ``'fermi'`` -- Poisson noise: ``sqrt(|rate|)`` fluctuations.
           * ``'swift'``  -- Gaussian noise drawn from the observed rate errors.

        3. **Lag error estimation** (``n_lag_sims`` iterations, default 1 000)
           -- For each realisation, draw a randomised CCF from within the CCF
           errors (Gaussian scatter), fit an asymmetric Gaussian, and record
           the peak location.  The standard deviation of the resulting lag
           distribution is the 1-sigma lag uncertainty.

        Parameters
        ----------
        lc_low : LightCurveData
            Low-energy band light curve.
        lc_high : LightCurveData
            High-energy band light curve.
        tstart : float
            Start time of the analysis window (seconds, same reference frame
            as ``LightCurveData.time``).
        tstop : float
            End time of the analysis window.
        n_ccf_sims : int, optional
            Number of Monte Carlo realisations for CCF error estimation
            (default: 10 000).
        n_lag_sims : int, optional
            Number of Monte Carlo realisations for lag error estimation
            (default: 1 000).
        data_type : str, optional
            Noise model selector: ``'fermi'`` for Poisson or ``'swift'`` for
            Gaussian (default: ``'swift'``).

        Returns
        -------
        SpectralLagResult
            Dataclass with ``lag``, ``lag_err``, ``offsets``, ``ccf``,
            ``ccf_err``, ``fit_params``, and ``n_simulations``.

        Raises
        ------
        ValueError
            If fewer than 4 bins overlap after trimming, or if both light
            curves have zero variance (making the DCCF undefined).
        """

        # ---------------------------------------------------------------
        # 0. Trim both light curves to [tstart, tstop]
        # ---------------------------------------------------------------
        mask_low = (lc_low.time >= tstart) & (lc_low.time <= tstop)
        mask_high = (lc_high.time >= tstart) & (lc_high.time <= tstop)

        t_low = lc_low.time[mask_low]
        rate_low = lc_low.rate[mask_low]
        err_low = lc_low.rate_err[mask_low]

        t_high = lc_high.time[mask_high]
        rate_high = lc_high.rate[mask_high]
        err_high = lc_high.rate_err[mask_high]

        # Interpolate to common uniform time grid (use the finer sampling)
        dt = min(lc_low.binsize, lc_high.binsize)
        t_common_start = max(t_low[0], t_high[0])
        t_common_stop = min(t_low[-1], t_high[-1])
        t_common = np.arange(t_common_start, t_common_stop + dt / 2.0, dt)

        if len(t_common) < 4:
            raise ValueError(
                "Fewer than 4 overlapping bins after trimming to "
                f"[{tstart}, {tstop}].  Cannot compute DCCF."
            )

        interp_low = interp1d(
            t_low, rate_low, kind='linear',
            bounds_error=False, fill_value=0.0,
        )
        interp_high = interp1d(
            t_high, rate_high, kind='linear',
            bounds_error=False, fill_value=0.0,
        )
        interp_err_low = interp1d(
            t_low, err_low, kind='linear',
            bounds_error=False, fill_value=0.0,
        )
        interp_err_high = interp1d(
            t_high, err_high, kind='linear',
            bounds_error=False, fill_value=0.0,
        )

        rate_low_c = interp_low(t_common)
        rate_high_c = interp_high(t_common)
        err_low_c = interp_err_low(t_common)
        err_high_c = interp_err_high(t_common)

        # Background subtraction (mean rate as background estimate)
        bg_low = np.mean(rate_low_c)
        bg_high = np.mean(rate_high_c)
        x_bs = rate_low_c - bg_low
        y_bs = rate_high_c - bg_high

        # ---------------------------------------------------------------
        # 1. Observed DCCF
        # ---------------------------------------------------------------
        offsets, ccf_observed = self.discrete_cross_correlation(
            x_bs, y_bs, t_common
        )

        # ---------------------------------------------------------------
        # 2. Monte Carlo CCF error estimation
        # ---------------------------------------------------------------
        ccf_ensemble = np.empty((n_ccf_sims, len(offsets)), dtype=float)

        for i in range(n_ccf_sims):
            if data_type == 'fermi':
                # Poisson noise: fluctuation ~ sqrt(|rate|)
                noise_low = np.random.normal(
                    0.0, np.sqrt(np.abs(rate_low_c))
                )
                noise_high = np.random.normal(
                    0.0, np.sqrt(np.abs(rate_high_c))
                )
            else:
                # Gaussian noise from observed errors (Swift-like)
                noise_low = np.random.normal(0.0, np.abs(err_low_c))
                noise_high = np.random.normal(0.0, np.abs(err_high_c))

            x_noisy = x_bs + noise_low
            y_noisy = y_bs + noise_high

            _, ccf_i = self.discrete_cross_correlation(
                x_noisy, y_noisy, t_common
            )
            ccf_ensemble[i, :] = ccf_i

        ccf_err = np.std(ccf_ensemble, axis=0)

        # ---------------------------------------------------------------
        # 3. Fit asymmetric Gaussian to observed CCF and get the lag
        # ---------------------------------------------------------------
        # Initial parameter guesses from the observed CCF
        peak_idx = np.argmax(ccf_observed)
        mu_guess = offsets[peak_idx]
        amp_guess = ccf_observed[peak_idx]
        sigma_guess = (offsets[-1] - offsets[0]) / 10.0

        p0 = [0.0, amp_guess, mu_guess, abs(sigma_guess), abs(sigma_guess)]

        # Reasonable bounds
        offset_range = offsets[-1] - offsets[0]
        bounds_lower = [-1.0, 0.0, offsets[0], 1e-6, 1e-6]
        bounds_upper = [1.0, 2.0, offsets[-1], offset_range, offset_range]

        try:
            popt_obs, _ = curve_fit(
                self.asymmetric_gaussian,
                offsets, ccf_observed,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=20000,
            )
        except Exception:
            # If the full CCF is hard to fit, restrict to central region
            center_mask = (
                (offsets >= mu_guess - offset_range / 4.0) &
                (offsets <= mu_guess + offset_range / 4.0)
            )
            popt_obs, _ = curve_fit(
                self.asymmetric_gaussian,
                offsets[center_mask], ccf_observed[center_mask],
                p0=p0,
                maxfev=20000,
            )

        lag_observed = popt_obs[2]  # mu is the lag

        fit_params = {
            'const': popt_obs[0],
            'amplitude': popt_obs[1],
            'mu': popt_obs[2],
            'sigma1': popt_obs[3],
            'sigma2': popt_obs[4],
        }

        # ---------------------------------------------------------------
        # 4. Monte Carlo lag error estimation
        # ---------------------------------------------------------------
        lag_distribution = []

        for _ in range(n_lag_sims):
            # Randomise CCF within its errors
            ccf_randomised = ccf_observed + np.random.normal(
                0.0, np.where(ccf_err > 0, ccf_err, 1e-10)
            )

            try:
                popt_mc, _ = curve_fit(
                    self.asymmetric_gaussian,
                    offsets, ccf_randomised,
                    p0=popt_obs,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=20000,
                )
                lag_distribution.append(popt_mc[2])
            except Exception:
                # Skip failed fits
                continue

        if len(lag_distribution) > 0:
            lag_err = float(np.std(lag_distribution))
        else:
            warnings.warn(
                "All Monte Carlo lag fits failed; lag_err set to NaN.",
                RuntimeWarning,
            )
            lag_err = float('nan')

        # ---------------------------------------------------------------
        # 5. Assemble result
        # ---------------------------------------------------------------
        return SpectralLagResult(
            lag=float(lag_observed),
            lag_err=lag_err,
            offsets=offsets,
            ccf=ccf_observed,
            ccf_err=ccf_err,
            fit_params=fit_params,
            n_simulations={'ccf_error': n_ccf_sims, 'lag_error': n_lag_sims},
        )
