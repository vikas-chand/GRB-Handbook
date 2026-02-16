"""Correlation analysis module for GRB observables."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats, optimize
from scipy.odr import ODR, Model, Data, RealData


class CorrelationAnalyzer:
    """Analyzer for GRB correlations (Amati, Yonetoku, etc.)."""

    def __init__(self, config: Dict = None):
        """
        Initialize correlation analyzer.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}

    def amati_relation(
        self,
        epeak_rest: float,
        eiso: float,
    ) -> Dict:
        """
        Check Epeak,i - Eiso correlation (Amati 2002).

        Amati relation: log(Epeak) = 0.57 * log(Eiso) - 27.2 (±0.2)
        (in cgs units: keV and erg)

        Parameters
        ----------
        epeak_rest : float
            Rest-frame peak energy in keV
        eiso : float
            Isotropic equivalent energy in erg

        Returns
        -------
        dict
            {expected_epeak, observed_epeak, sigma_offset, is_outlier}
        """
        # Amati parameters
        slope = 0.57
        intercept = -27.2
        scatter = 0.2  # Intrinsic scatter in dex

        log_eiso = np.log10(eiso)
        expected_log_epeak = slope * log_eiso + intercept
        expected_epeak = 10 ** expected_log_epeak

        observed_log_epeak = np.log10(epeak_rest)
        offset_dex = observed_log_epeak - expected_log_epeak
        sigma_offset = offset_dex / scatter

        is_outlier = np.abs(sigma_offset) > 2.0

        return {
            'expected_epeak': expected_epeak,
            'observed_epeak': epeak_rest,
            'offset_dex': offset_dex,
            'sigma_offset': sigma_offset,
            'is_outlier': is_outlier,
            'relation': 'Amati_2002',
        }

    def yonetoku_relation(
        self,
        epeak_rest: float,
        lpeak: float,
    ) -> Dict:
        """
        Check Epeak,i - Lpeak correlation (Yonetoku 2004).

        Yonetoku relation: log(Lpeak) = 2.0 * log(Epeak) + 47.0
        (in cgs: erg/s and keV)

        Parameters
        ----------
        epeak_rest : float
            Rest-frame peak energy in keV
        lpeak : float
            Peak luminosity in erg/s

        Returns
        -------
        dict
            {expected_lpeak, observed_lpeak, sigma_offset, is_outlier}
        """
        # Yonetoku parameters
        slope = 2.0
        intercept = 47.0
        scatter = 0.25  # Intrinsic scatter in dex

        log_epeak = np.log10(epeak_rest)
        expected_log_lpeak = slope * log_epeak + intercept
        expected_lpeak = 10 ** expected_log_lpeak

        observed_log_lpeak = np.log10(lpeak)
        offset_dex = observed_log_lpeak - expected_log_lpeak
        sigma_offset = offset_dex / scatter

        is_outlier = np.abs(sigma_offset) > 2.0

        return {
            'expected_lpeak': expected_lpeak,
            'observed_lpeak': lpeak,
            'offset_dex': offset_dex,
            'sigma_offset': sigma_offset,
            'is_outlier': is_outlier,
            'relation': 'Yonetoku_2004',
        }

    def ghirlanda_relation(
        self,
        epeak_rest: float,
        egamma: float,
    ) -> Dict:
        """
        Check Epeak,i - Egamma correlation (collimation-corrected, Ghirlanda 2004).

        Ghirlanda relation: log(Egamma) = 1.74 * log(Epeak) + 30.38

        Parameters
        ----------
        epeak_rest : float
            Rest-frame peak energy in keV
        egamma : float
            Collimation-corrected energy (Eiso * theta_jet^2) in erg

        Returns
        -------
        dict
            {expected_egamma, observed_egamma, sigma_offset, is_outlier}
        """
        # Ghirlanda parameters
        slope = 1.74
        intercept = 30.38
        scatter = 0.27

        log_epeak = np.log10(epeak_rest)
        expected_log_egamma = slope * log_epeak + intercept
        expected_egamma = 10 ** expected_log_egamma

        observed_log_egamma = np.log10(egamma)
        offset_dex = observed_log_egamma - expected_log_egamma
        sigma_offset = offset_dex / scatter

        is_outlier = np.abs(sigma_offset) > 2.0

        return {
            'expected_egamma': expected_egamma,
            'observed_egamma': egamma,
            'offset_dex': offset_dex,
            'sigma_offset': sigma_offset,
            'is_outlier': is_outlier,
            'relation': 'Ghirlanda_2004',
        }

    def ep_eiso_diagram(self, grbs: List[Dict]) -> Dict:
        """
        Analyze Amati relation for a sample of GRBs.

        Fits the Epeak - Eiso correlation and returns best-fit parameters.

        Parameters
        ----------
        grbs : list of dict
            GRB data, each with {epeak_rest, eiso, epeak_err, eiso_err}

        Returns
        -------
        dict
            {slope, intercept, scatter, r_squared, n_outliers}
        """
        if len(grbs) < 3:
            return {'error': 'Need at least 3 GRBs for correlation analysis'}

        # Extract data
        epeak_vals = np.array([g.get('epeak_rest', 0) for g in grbs])
        eiso_vals = np.array([g.get('eiso', 0) for g in grbs])
        epeak_err = np.array([g.get('epeak_err', 0.1 * e) for g, e in zip(grbs, epeak_vals)])
        eiso_err = np.array([g.get('eiso_err', 0.1 * e) for g, e in zip(grbs, eiso_vals)])

        # Remove invalid data
        mask = (epeak_vals > 0) & (eiso_vals > 0)
        epeak_vals = epeak_vals[mask]
        eiso_vals = eiso_vals[mask]
        epeak_err = epeak_err[mask]
        eiso_err = eiso_err[mask]

        if len(epeak_vals) < 3:
            return {'error': 'Not enough valid data'}

        # Log-log space
        log_epeak = np.log10(epeak_vals)
        log_eiso = np.log10(eiso_vals)
        log_epeak_err = epeak_err / (epeak_vals * np.log(10))
        log_eiso_err = eiso_err / (eiso_vals * np.log(10))

        # Orthogonal distance regression
        def linear_model(p, x):
            return p[0] * x + p[1]

        model = Model(linear_model)
        data = RealData(log_eiso, log_epeak, sx=log_eiso_err, sy=log_epeak_err)

        try:
            odr = ODR(data, model, beta0=[0.5, -25])
            output = odr.run()
            slope = output.beta[0]
            intercept = output.beta[1]
        except Exception:
            # Fall back to simple linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_eiso, log_epeak)
            r_value = r_value ** 2

        # Compute scatter
        residuals = log_epeak - (slope * log_eiso + intercept)
        scatter = np.std(residuals)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_epeak - np.mean(log_epeak)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Count outliers (> 2 sigma)
        sigma_offsets = residuals / scatter if scatter > 0 else np.zeros_like(residuals)
        n_outliers = np.sum(np.abs(sigma_offsets) > 2.0)

        return {
            'slope': slope,
            'intercept': intercept,
            'scatter': scatter,
            'r_squared': r_squared,
            'n_outliers': n_outliers,
            'n_grbs': len(epeak_vals),
            'correlation': 'Amati',
        }

    def ep_lpeak_diagram(self, grbs: List[Dict]) -> Dict:
        """
        Analyze Yonetoku relation for a sample of GRBs.

        Parameters
        ----------
        grbs : list of dict
            GRB data, each with {epeak_rest, lpeak, epeak_err, lpeak_err}

        Returns
        -------
        dict
            {slope, intercept, scatter, r_squared, n_outliers}
        """
        if len(grbs) < 3:
            return {'error': 'Need at least 3 GRBs'}

        # Extract data
        epeak_vals = np.array([g.get('epeak_rest', 0) for g in grbs])
        lpeak_vals = np.array([g.get('lpeak', 0) for g in grbs])
        epeak_err = np.array([g.get('epeak_err', 0.1 * e) for g, e in zip(grbs, epeak_vals)])
        lpeak_err = np.array([g.get('lpeak_err', 0.1 * l) for g, l in zip(grbs, lpeak_vals)])

        # Remove invalid
        mask = (epeak_vals > 0) & (lpeak_vals > 0)
        epeak_vals = epeak_vals[mask]
        lpeak_vals = lpeak_vals[mask]
        epeak_err = epeak_err[mask]
        lpeak_err = lpeak_err[mask]

        if len(epeak_vals) < 3:
            return {'error': 'Not enough valid data'}

        # Log-log space
        log_epeak = np.log10(epeak_vals)
        log_lpeak = np.log10(lpeak_vals)
        log_epeak_err = epeak_err / (epeak_vals * np.log(10))
        log_lpeak_err = lpeak_err / (lpeak_vals * np.log(10))

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_epeak, log_lpeak)
        r_squared = r_value ** 2

        # Scatter
        residuals = log_lpeak - (slope * log_epeak + intercept)
        scatter = np.std(residuals)

        # Outliers
        sigma_offsets = residuals / scatter if scatter > 0 else np.zeros_like(residuals)
        n_outliers = np.sum(np.abs(sigma_offsets) > 2.0)

        return {
            'slope': slope,
            'intercept': intercept,
            'scatter': scatter,
            'r_squared': r_squared,
            'n_outliers': n_outliers,
            'n_grbs': len(epeak_vals),
            'correlation': 'Yonetoku',
        }

    def fit_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_err: Optional[np.ndarray] = None,
        y_err: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Fit linear correlation in log-log space.

        Uses orthogonal distance regression with intrinsic scatter.

        Parameters
        ----------
        x : np.ndarray
            X data (will be log-transformed)
        y : np.ndarray
            Y data (will be log-transformed)
        x_err : np.ndarray, optional
            X errors
        y_err : np.ndarray, optional
            Y errors

        Returns
        -------
        dict
            {slope, intercept, scatter, r_squared}
        """
        # Remove invalid data
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x_err is not None:
            x_err = x_err[mask]
        if y_err is not None:
            y_err = y_err[mask]

        if len(x) < 3:
            return {'error': 'Insufficient data'}

        # Log-log transformation
        log_x = np.log10(x)
        log_y = np.log10(y)

        if x_err is not None:
            log_x_err = x_err / (x * np.log(10))
        else:
            log_x_err = None

        if y_err is not None:
            log_y_err = y_err / (y * np.log(10))
        else:
            log_y_err = None

        # Try ODR fit
        try:
            def linear_model(p, x):
                return p[0] * x + p[1]

            model = Model(linear_model)

            if log_x_err is not None and log_y_err is not None:
                data = RealData(log_x, log_y, sx=log_x_err, sy=log_y_err)
            else:
                data = RealData(log_x, log_y)

            odr = ODR(data, model, beta0=[1.0, 0.0])
            output = odr.run()

            slope = output.beta[0]
            intercept = output.beta[1]

        except Exception:
            # Fall back to simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

        # Compute metrics
        residuals = log_y - (slope * log_x + intercept)
        scatter = np.std(residuals)

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'slope': slope,
            'intercept': intercept,
            'scatter': scatter,
            'r_squared': r_squared,
            'n_points': len(x),
        }

    def lag_luminosity_relation(
        self,
        lag: float,
        lpeak: float,
    ) -> Dict:
        """
        Check spectral lag - luminosity relation.

        Loose anti-correlation: brighter GRBs have smaller lags.

        Parameters
        ----------
        lag : float
            Spectral lag in seconds
        lpeak : float
            Peak luminosity in erg/s

        Returns
        -------
        dict
            Analysis results
        """
        # Empirical relation (Norris et al. 2000)
        # lag ≈ 50 ms * (Lpeak/10^52)^(-0.4)

        expected_lag = 0.05 * (lpeak / 1e52) ** (-0.4)

        ratio = lag / expected_lag if expected_lag > 0 else 0

        return {
            'observed_lag': lag,
            'expected_lag': expected_lag,
            'ratio': ratio,
            'consistent': 0.5 < ratio < 2.0,
        }

    def variability_luminosity_relation(
        self,
        variability: float,
        lpeak: float,
    ) -> Dict:
        """
        Check variability - luminosity anti-correlation.

        Brighter GRBs tend to be less variable.

        Parameters
        ----------
        variability : float
            Variability index (e.g., fractional RMS)
        lpeak : float
            Peak luminosity in erg/s

        Returns
        -------
        dict
            Analysis results
        """
        # Empirical relation: brighter GRBs have smaller V
        # log(V) ≈ -0.5 * log(Lpeak) + 2.0

        log_lpeak = np.log10(lpeak)
        expected_log_v = -0.5 * log_lpeak + 2.0
        expected_v = 10 ** expected_log_v

        ratio = variability / expected_v if expected_v > 0 else 0

        return {
            'observed_variability': variability,
            'expected_variability': expected_v,
            'ratio': ratio,
            'consistent': 0.5 < ratio < 2.0,
        }

    def check_all_correlations(self, grb_data: Dict) -> Dict:
        """
        Run all correlation checks for a single GRB.

        Parameters
        ----------
        grb_data : dict
            GRB parameters including epeak_rest, eiso, lpeak, etc.

        Returns
        -------
        dict
            Summary of all correlation checks
        """
        results = {
            'grb_name': grb_data.get('name', 'Unknown'),
            'z': grb_data.get('redshift', None),
        }

        # Check Amati relation
        if 'epeak_rest' in grb_data and 'eiso' in grb_data:
            results['amati'] = self.amati_relation(
                grb_data['epeak_rest'],
                grb_data['eiso']
            )

        # Check Yonetoku relation
        if 'epeak_rest' in grb_data and 'lpeak' in grb_data:
            results['yonetoku'] = self.yonetoku_relation(
                grb_data['epeak_rest'],
                grb_data['lpeak']
            )

        # Check Ghirlanda relation
        if 'epeak_rest' in grb_data and 'egamma' in grb_data:
            results['ghirlanda'] = self.ghirlanda_relation(
                grb_data['epeak_rest'],
                grb_data['egamma']
            )

        # Check lag-luminosity
        if 'spectral_lag' in grb_data and 'lpeak' in grb_data:
            results['lag_luminosity'] = self.lag_luminosity_relation(
                grb_data['spectral_lag'],
                grb_data['lpeak']
            )

        # Check variability-luminosity
        if 'variability' in grb_data and 'lpeak' in grb_data:
            results['variability_luminosity'] = self.variability_luminosity_relation(
                grb_data['variability'],
                grb_data['lpeak']
            )

        return results
