"""Afterglow analysis module for GRB events."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import optimize, stats, integrate
from scipy.interpolate import interp1d


@dataclass
class AfterglowParams:
    """Container for afterglow parameters."""
    decay_index: float = None
    decay_index_err: float = None
    t_break: float = None
    t_break_err: float = None
    alpha1: float = None  # Initial decay index
    alpha2: float = None  # Post-break decay index
    jet_angle: float = None
    has_plateau: bool = False
    has_jet_break: bool = False


class AfterglowModeler:
    """Modeler for GRB afterglow properties."""

    def __init__(self, config: Dict = None):
        """
        Initialize afterglow modeler.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}

    def fit_powerlaw_decay(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        t0: float = 0,
    ) -> Dict:
        """
        Fit single power-law decay model.

        Model: F(t) = F0 * ((t-t0)/t_ref)^(-alpha)

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors
        t0 : float
            Reference time

        Returns
        -------
        dict
            Fit parameters: {decay_index, decay_index_err, norm, norm_err, chi_sq, dof}
        """

        def model(t, alpha, f0):
            return f0 * ((t - t0) / 1.0) ** (-alpha)

        def chi_squared(params):
            model_pred = model(time, *params)
            chi_sq = ((flux - model_pred) / flux_err) ** 2
            return chi_sq.sum()

        # Initial guess
        p0 = [1.5, flux[0]]

        # Bounds
        bounds = [(0.1, 5.0), (np.min(flux) * 0.5, np.max(flux) * 2)]

        try:
            result = optimize.minimize(
                chi_squared,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha, f0 = result.x
                chi_sq = chi_squared(result.x)
                dof = len(time) - 2

                # Estimate errors from curvature
                eps = 1e-5
                hess = np.zeros((2, 2))
                for i in range(2):
                    for j in range(2):
                        p_pp = result.x.copy()
                        p_pp[i] += eps
                        p_pp[j] += eps

                        p_pm = result.x.copy()
                        p_pm[i] += eps
                        p_pm[j] -= eps

                        p_mp = result.x.copy()
                        p_mp[i] -= eps
                        p_mp[j] += eps

                        p_mm = result.x.copy()
                        p_mm[i] -= eps
                        p_mm[j] -= eps

                        hess[i, j] = (
                            chi_squared(p_pp) - chi_squared(p_pm) -
                            chi_squared(p_mp) + chi_squared(p_mm)
                        ) / (4 * eps ** 2)

                try:
                    cov = np.linalg.inv(hess)
                    alpha_err = np.sqrt(cov[0, 0])
                    f0_err = np.sqrt(cov[1, 1])
                except np.linalg.LinAlgError:
                    alpha_err = np.inf
                    f0_err = np.inf

                return {
                    'decay_index': alpha,
                    'decay_index_err': alpha_err,
                    'norm': f0,
                    'norm_err': f0_err,
                    'chi_sq': chi_sq,
                    'dof': dof,
                    'fit_success': True,
                }
        except Exception as e:
            print(f"Power-law fit error: {e}")

        return {'fit_success': False}

    def fit_broken_powerlaw(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> Dict:
        """
        Fit broken power-law model (smooth break).

        Model: F(t) = F0 * [(t/tb)^(alpha1*s) + (t/tb)^(alpha2*s)]^(-1/s)

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors

        Returns
        -------
        dict
            Fit parameters: {alpha1, alpha2, t_break, smoothness, chi_sq, ...}
        """

        def model(t, alpha1, alpha2, tb, s):
            with np.errstate(divide='ignore', over='ignore'):
                term1 = (t / tb) ** (alpha1 * s)
                term2 = (t / tb) ** (alpha2 * s)
                flux_model = ((term1 + term2) ** (-1.0 / s))
            return flux_model * flux[0]

        def chi_squared(params):
            model_pred = model(time, *params)
            chi_sq = ((flux - model_pred) / flux_err) ** 2
            return np.sum(chi_sq[~np.isnan(chi_sq)])

        # Find likely break time
        break_idx = len(time) // 2
        tb0 = time[break_idx]

        p0 = [0.8, 2.0, tb0, 2.0]
        bounds = [
            (0.1, 3.0),  # alpha1
            (1.0, 5.0),  # alpha2
            (time[1], time[-2]),  # t_break
            (0.5, 5.0),  # smoothness
        ]

        try:
            result = optimize.minimize(
                chi_squared,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha1, alpha2, tb, s = result.x
                chi_sq = chi_squared(result.x)
                dof = len(time) - 4

                return {
                    'alpha1': alpha1,
                    'alpha2': alpha2,
                    't_break': tb,
                    'smoothness': s,
                    'chi_sq': chi_sq,
                    'dof': dof,
                    'fit_success': True,
                }
        except Exception as e:
            print(f"Broken power-law fit error: {e}")

        return {'fit_success': False}

    def fit_double_broken_powerlaw(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> Dict:
        """
        Fit double broken power-law model (triple-segment afterglow).

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors

        Returns
        -------
        dict
            Fit parameters including two break times
        """

        def model(t, alpha1, alpha2, alpha3, tb1, tb2):
            flux_model = np.zeros_like(t, dtype=float)

            # Segment 1: t < tb1
            mask1 = t < tb1
            if np.any(mask1):
                flux_model[mask1] = flux[0] * (t[mask1] / tb1) ** (-alpha1)

            # Segment 2: tb1 < t < tb2
            mask2 = (t >= tb1) & (t < tb2)
            if np.any(mask2):
                f_break1 = flux[0] * (tb1 / tb1) ** (-alpha1)
                flux_model[mask2] = f_break1 * (t[mask2] / tb1) ** (-alpha2)

            # Segment 3: t > tb2
            mask3 = t >= tb2
            if np.any(mask3):
                f_break1 = flux[0] * (tb1 / tb1) ** (-alpha1)
                f_break2 = f_break1 * (tb2 / tb1) ** (-alpha2)
                flux_model[mask3] = f_break2 * (t[mask3] / tb2) ** (-alpha3)

            return flux_model

        def chi_squared(params):
            model_pred = model(time, *params)
            chi_sq = ((flux - model_pred) / flux_err) ** 2
            return np.sum(chi_sq[~np.isnan(chi_sq)])

        # Initial breaks at 1/3 and 2/3
        tb1_0 = time[len(time) // 3]
        tb2_0 = time[2 * len(time) // 3]

        p0 = [0.8, 1.5, 2.5, tb1_0, tb2_0]
        bounds = [
            (0.1, 3.0),  # alpha1
            (0.5, 3.0),  # alpha2
            (1.0, 5.0),  # alpha3
            (time[1], time[-3]),  # tb1
            (time[2], time[-2]),  # tb2
        ]

        try:
            result = optimize.minimize(
                chi_squared,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha1, alpha2, alpha3, tb1, tb2 = result.x
                chi_sq = chi_squared(result.x)
                dof = len(time) - 5

                return {
                    'alpha1': alpha1,
                    'alpha2': alpha2,
                    'alpha3': alpha3,
                    't_break1': tb1,
                    't_break2': tb2,
                    'chi_sq': chi_sq,
                    'dof': dof,
                    'fit_success': True,
                }
        except Exception as e:
            print(f"Double broken power-law fit error: {e}")

        return {'fit_success': False}

    def detect_plateau(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> Dict:
        """
        Detect plateau phase in X-ray afterglow.

        Looks for segment with decay index |alpha| < 0.5.

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors

        Returns
        -------
        dict
            Plateau parameters: {has_plateau, start_time, end_time, plateau_level}
        """
        if len(time) < 10:
            return {'has_plateau': False}

        # Fit local power-laws in sliding windows
        window = max(3, len(time) // 10)

        alpha_values = []
        time_mid = []

        for i in range(len(time) - window):
            t_seg = time[i : i + window]
            f_seg = flux[i : i + window]
            f_err_seg = flux_err[i : i + window]

            if len(t_seg) > 2:
                fit_result = self.fit_powerlaw_decay(t_seg, f_seg, f_err_seg)
                if fit_result.get('fit_success', False):
                    alpha_values.append(fit_result['decay_index'])
                    time_mid.append(t_seg[len(t_seg) // 2])

        if not alpha_values:
            return {'has_plateau': False}

        alpha_values = np.array(alpha_values)
        time_mid = np.array(time_mid)

        # Find where |alpha| < 0.5
        plateau_mask = np.abs(alpha_values) < 0.5

        if plateau_mask.sum() == 0:
            return {'has_plateau': False}

        # Find continuous plateau region
        plateau_times = time_mid[plateau_mask]

        return {
            'has_plateau': True,
            'start_time': plateau_times[0],
            'end_time': plateau_times[-1],
            'plateau_level': np.mean(flux[
                (time >= plateau_times[0]) & (time <= plateau_times[-1])
            ]),
            'mean_alpha': np.mean(alpha_values[plateau_mask]),
        }

    def detect_jet_break(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> Dict:
        """
        Detect jet break (steepening) in decay.

        Looks for point where decay index changes by > 0.5.

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors

        Returns
        -------
        dict
            Jet break parameters: {has_jet_break, break_time, alpha_before, alpha_after}
        """
        if len(time) < 20:
            return {'has_jet_break': False}

        # Split data in half and fit each
        split_idx = len(time) // 2

        t1 = time[:split_idx]
        f1 = flux[:split_idx]
        f_err1 = flux_err[:split_idx]

        t2 = time[split_idx:]
        f2 = flux[split_idx:]
        f_err2 = flux_err[split_idx:]

        fit1 = self.fit_powerlaw_decay(t1, f1, f_err1)
        fit2 = self.fit_powerlaw_decay(t2, f2, f_err2)

        if not fit1.get('fit_success') or not fit2.get('fit_success'):
            return {'has_jet_break': False}

        alpha1 = fit1['decay_index']
        alpha2 = fit2['decay_index']

        # Check for significant steepening
        if abs(alpha2 - alpha1) > 0.5:
            return {
                'has_jet_break': True,
                'break_time': time[split_idx],
                'alpha_before': alpha1,
                'alpha_after': alpha2,
                'steepening': alpha2 - alpha1,
            }
        else:
            return {'has_jet_break': False}

    def closure_relations(
        self,
        decay_index: float,
        spectral_index: float,
        regime: str = "ISM",
    ) -> Dict:
        """
        Test standard afterglow closure relations alpha = f(beta).

        Parameters
        ----------
        decay_index : float
            Temporal decay index
        spectral_index : float
            Spectral photon index
        regime : str
            Environment: "ISM" or "wind"

        Returns
        -------
        dict
            Closure relation results: {relation_name: {expected_alpha, observed_alpha, consistent}}
        """
        beta = spectral_index
        alpha_obs = decay_index

        relations = {}

        if regime == "ISM":
            # Forward shock, ISM environment
            # Synchrotron (slow cooling): alpha = (3*beta - 1)/2
            alpha_syn_slow = (3 * beta - 1) / 2
            relations['FS_ISM_Synchrotron_SlowCool'] = {
                'expected_alpha': alpha_syn_slow,
                'observed_alpha': alpha_obs,
                'consistent': np.abs(alpha_obs - alpha_syn_slow) < 0.3,
            }

            # Inverse Compton (slow cooling): alpha = (3*beta + 1)/2
            alpha_ic_slow = (3 * beta + 1) / 2
            relations['FS_ISM_IC_SlowCool'] = {
                'expected_alpha': alpha_ic_slow,
                'observed_alpha': alpha_obs,
                'consistent': np.abs(alpha_obs - alpha_ic_slow) < 0.3,
            }

            # Synchrotron (fast cooling): alpha = beta/2 - 1/4
            alpha_syn_fast = beta / 2 - 0.25
            relations['FS_ISM_Synchrotron_FastCool'] = {
                'expected_alpha': alpha_syn_fast,
                'observed_alpha': alpha_obs,
                'consistent': np.abs(alpha_obs - alpha_syn_fast) < 0.3,
            }

        elif regime == "wind":
            # Forward shock, wind environment
            # Synchrotron (slow cooling): alpha = (3*beta - 1)/2 - 0.5
            alpha_syn_slow = (3 * beta - 1) / 2 - 0.5
            relations['FS_Wind_Synchrotron_SlowCool'] = {
                'expected_alpha': alpha_syn_slow,
                'observed_alpha': alpha_obs,
                'consistent': np.abs(alpha_obs - alpha_syn_slow) < 0.3,
            }

        return relations

    def fit_reverse_shock(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> Dict:
        """
        Fit reverse shock component (early steep decay).

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors

        Returns
        -------
        dict
            Reverse shock parameters
        """
        # Reverse shock typically has very steep decay (alpha ~ 3-4) at early times
        if len(time) < 10:
            return {'has_reverse_shock': False}

        # Check first few points
        t_early = time[:len(time) // 4]
        f_early = flux[:len(time) // 4]
        f_err_early = flux_err[:len(time) // 4]

        if len(t_early) < 3:
            return {'has_reverse_shock': False}

        fit_result = self.fit_powerlaw_decay(t_early, f_early, f_err_early)

        if fit_result.get('fit_success', False):
            alpha_early = fit_result['decay_index']

            if alpha_early > 2.0:  # Steep decay indicative of reverse shock
                return {
                    'has_reverse_shock': True,
                    'decay_index': alpha_early,
                    'decay_index_err': fit_result.get('decay_index_err', 0),
                }

        return {'has_reverse_shock': False}

    def multiwavelength_fit(self, data_dict: Dict) -> Dict:
        """
        Simultaneous fit across multiple wavelength bands.

        Assumes shared break times but different normalizations.

        Parameters
        ----------
        data_dict : dict
            Dictionary with band names as keys, each containing {time, flux, flux_err}

        Returns
        -------
        dict
            Fit parameters: {t_break, alphas_per_band, norms_per_band, ...}
        """
        bands = list(data_dict.keys())
        n_bands = len(bands)

        if n_bands < 2:
            return {'fit_success': False, 'error': 'Need at least 2 bands'}

        # Use first band to estimate break time
        t_common = data_dict[bands[0]]['time']
        break_idx = len(t_common) // 2
        tb0 = t_common[break_idx]

        def joint_chi_squared(params):
            # params: [tb, alpha1_band1, ..., alpha1_bandN, alpha2_band1, ..., alpha2_bandN,
            #          norm_band1, ..., norm_bandN]
            tb = params[0]
            alphas1 = params[1 : n_bands + 1]
            alphas2 = params[n_bands + 1 : 2 * n_bands + 1]
            norms = params[2 * n_bands + 1 : 3 * n_bands + 1]

            chi_sq = 0

            for i, band in enumerate(bands):
                t = data_dict[band]['time']
                f = data_dict[band]['flux']
                f_err = data_dict[band]['flux_err']

                # Power law before and after break
                model_flux = np.zeros_like(t)
                mask1 = t < tb
                mask2 = t >= tb

                if np.any(mask1):
                    model_flux[mask1] = norms[i] * (t[mask1] / tb) ** (-alphas1[i])

                if np.any(mask2):
                    f_break = norms[i] * (tb / tb) ** (-alphas1[i])
                    model_flux[mask2] = f_break * (t[mask2] / tb) ** (-alphas2[i])

                chi_sq += ((f - model_flux) / f_err) ** 2

            return chi_sq.sum()

        # Initial guess
        p0 = [tb0] + [0.8] * n_bands + [2.0] * n_bands + [1.0] * n_bands

        bounds = [
            (t_common[1], t_common[-2]),  # tb
        ] + [(0.1, 3.0)] * n_bands + [(0.5, 5.0)] * n_bands + [(0.01, 100)] * n_bands

        try:
            result = optimize.minimize(
                joint_chi_squared,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                tb = result.x[0]
                alphas1 = dict(zip(bands, result.x[1 : n_bands + 1]))
                alphas2 = dict(zip(bands, result.x[n_bands + 1 : 2 * n_bands + 1]))
                norms = dict(zip(bands, result.x[2 * n_bands + 1 : 3 * n_bands + 1]))

                chi_sq = joint_chi_squared(result.x)
                dof = sum(len(data_dict[b]['time']) for b in bands) - len(result.x)

                return {
                    'fit_success': True,
                    't_break': tb,
                    'alpha1_per_band': alphas1,
                    'alpha2_per_band': alphas2,
                    'norm_per_band': norms,
                    'chi_sq': chi_sq,
                    'dof': dof,
                }
        except Exception as e:
            print(f"Multiwavelength fit error: {e}")

        return {'fit_success': False}

    def calculate_jet_angle(
        self,
        t_break: float,
        E_iso: float,
        n_ism: float = 1.0,
        redshift: float = 1.0,
    ) -> float:
        """
        Calculate jet opening angle from jet break time.

        Uses equation: theta_jet ~ 0.07 * sqrt(n_ism * E51 / t_break_days)

        Parameters
        ----------
        t_break : float
            Jet break time in seconds
        E_iso : float
            Isotropic equivalent energy in erg
        n_ism : float
            ISM density in cm^-3
        redshift : float
            Source redshift

        Returns
        -------
        float
            Jet opening angle in radians
        """
        E51 = E_iso / 1e51

        # Convert time to days
        t_break_days = t_break / 86400

        # Jet angle in radians
        # theta_jet = 0.07 * sqrt(n_ism * E51 / t_break_days)
        theta_jet = 0.07 * np.sqrt(n_ism * E51 / t_break_days)

        return theta_jet

    def energy_injection_model(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> Dict:
        """
        Fit plateau powered by energy injection.

        Model assumes constant energy injection rate.

        Parameters
        ----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux measurements
        flux_err : np.ndarray
            Flux errors

        Returns
        -------
        dict
            Energy injection parameters
        """

        def model(t, alpha0, alpha1, injection_end, f0):
            # Initial steep decay, then shallower with energy injection
            f = np.zeros_like(t)
            mask1 = t < injection_end
            mask2 = t >= injection_end

            if np.any(mask1):
                # With energy injection: alpha_ej = alpha0 - injection_index
                alpha_ej = alpha0 - 0.5  # Typical injection reduces decay by 0.5
                f[mask1] = f0 * (t[mask1]) ** (-alpha_ej)

            if np.any(mask2):
                # Without energy injection: normal decay
                f_end = f0 * injection_end ** (-alpha0 + 0.5)
                f[mask2] = f_end * (t[mask2] / injection_end) ** (-alpha0)

            return f

        def chi_squared(params):
            model_pred = model(time, *params)
            return ((flux - model_pred) / flux_err) ** 2

        p0 = [2.0, 1.5, time[len(time) // 2], flux[0]]
        bounds = [(0.5, 5.0), (0.5, 3.0), (time[1], time[-2]), (flux.min() * 0.5, flux.max() * 2)]

        try:
            result = optimize.minimize(
                lambda p: chi_squared(p).sum(),
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha0, alpha1, t_inj_end, f0 = result.x
                chi_sq = chi_squared(result.x).sum()

                return {
                    'fit_success': True,
                    'alpha_before_injection': alpha0,
                    'alpha_after_injection': alpha1,
                    'injection_end_time': t_inj_end,
                    'injection_rate': alpha0 - alpha1 - 0.5,
                    'chi_sq': chi_sq,
                }
        except Exception as e:
            print(f"Energy injection fit error: {e}")

        return {'fit_success': False}

    def forward_shock_prediction(
        self,
        params: Dict,
        times: np.ndarray,
        frequency: float,
    ) -> np.ndarray:
        """
        Predict forward shock afterglow flux at given times and frequency.

        Uses standard forward shock model.

        Parameters
        ----------
        params : dict
            Parameters: {E_iso, n_ism, theta_obs, p_index, epsilon_e, epsilon_b, ...}
        times : np.ndarray
            Time array in seconds
        frequency : float
            Frequency in Hz

        Returns
        -------
        np.ndarray
            Predicted flux in mJy
        """
        E_iso = params.get('E_iso', 1e51)  # erg
        n_ism = params.get('n_ism', 1.0)  # cm^-3
        theta_obs = params.get('theta_obs', 0.1)  # rad
        p_index = params.get('p_index', 2.5)
        epsilon_e = params.get('epsilon_e', 0.1)
        epsilon_b = params.get('epsilon_b', 0.01)

        # Simple forward shock prediction
        # Flux ~ E_iso^(1/4) * n^(-1/2) * t^(-3/4) for typical parameters
        c = 3e10  # speed of light
        mpc = 3.086e24  # 1 Mpc in cm
        d_L = 1e28  # Assume 1 Gpc luminosity distance

        flux = (E_iso ** 0.25) * (n_ism ** 0.5) / (d_L ** 2) * (times + 1) ** (-0.75)

        return flux
