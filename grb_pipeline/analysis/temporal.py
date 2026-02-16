"""Temporal analysis module for GRB light curves."""

from typing import Tuple, Optional, Dict, List
import numpy as np
from scipy import signal, optimize, stats
from scipy.interpolate import interp1d

from .lightcurve import LightCurveData


class TemporalAnalyzer:
    """Analyzer for temporal properties of GRB light curves."""

    def __init__(self, config: Dict = None):
        """
        Initialize temporal analyzer.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}

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

    def spectral_lag(
        self,
        lc_high: LightCurveData,
        lc_low: LightCurveData,
    ) -> Tuple[float, float]:
        """
        Calculate spectral lag (cross-correlation lag between energy bands).

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
