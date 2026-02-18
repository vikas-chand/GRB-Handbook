"""Spectral analysis module for GRB events."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import optimize, stats, integrate
from scipy.special import gamma as gamma_function


@dataclass
class SpectralData:
    """Container for spectral data."""
    energy: np.ndarray  # Energy in keV
    counts: np.ndarray  # Spectral counts
    counts_err: np.ndarray  # Count errors (sqrt for Poisson)
    area: float = 1.0  # Detector area in cm^2
    exposure: float = 1.0  # Exposure time in seconds
    backscale: float = 1.0  # Background scaling factor


@dataclass
class SpectralFit:
    """Container for spectral fit results."""
    model: str  # Model name
    params: Dict  # Parameter values
    errors: Dict  # Parameter errors
    cov_matrix: np.ndarray = None  # Covariance matrix
    chi_sq: float = None  # Chi-squared
    dof: int = None  # Degrees of freedom
    aic: float = None  # Akaike information criterion
    bic: float = None  # Bayesian information criterion
    fit_success: bool = True


class SpectralAnalyzer:
    """Analyzer for GRB spectral properties."""

    def __init__(self, config: Dict = None):
        """
        Initialize spectral analyzer.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with cosmology parameters
        """
        self.config = config or {}
        # Cosmological parameters
        self.H0 = self.config.get('H0', 70.0)  # Hubble constant km/s/Mpc
        self.Om0 = self.config.get('Om0', 0.3)  # Matter density
        self.OL0 = self.config.get('OL0', 0.7)  # Dark energy density

    @staticmethod
    def band_function(
        energy: np.ndarray,
        alpha: float,
        beta: float,
        epeak: float,
        norm: float,
    ) -> np.ndarray:
        """
        Band GRB spectral model (Band et al. 1993).

        N(E) = norm * [(E/100)^alpha * exp(-(2+alpha)*E/epeak)] for E < (alpha-beta)*epeak/(2+alpha)
        N(E) = norm * [((alpha-beta)*epeak/(100*(2+alpha)))^(alpha-beta) * exp(beta-alpha) * (E/100)^beta]
               for E >= (alpha-beta)*epeak/(2+alpha)

        Parameters
        ----------
        energy : np.ndarray
            Photon energy in keV
        alpha : float
            Low-energy spectral index
        beta : float
            High-energy spectral index
        epeak : float
            Peak energy in keV
        norm : float
            Normalization

        Returns
        -------
        np.ndarray
            Spectral response at given energies
        """
        energy = np.atleast_1d(energy)
        e0 = 100.0  # Pivot energy

        # Transition energy
        e_trans = (alpha - beta) * epeak / (2 + alpha)

        # Initialize result
        flux = np.zeros_like(energy, dtype=float)

        # Low energy segment
        low = energy < e_trans
        if np.any(low):
            flux[low] = norm * (energy[low] / e0) ** alpha * np.exp(
                -(2 + alpha) * energy[low] / epeak
            )

        # High energy segment
        high = energy >= e_trans
        if np.any(high):
            A = ((alpha - beta) * epeak / (e0 * (2 + alpha))) ** (alpha - beta)
            B = np.exp(beta - alpha)
            flux[high] = norm * A * B * (energy[high] / e0) ** beta

        return flux

    @staticmethod
    def cutoff_powerlaw(
        energy: np.ndarray,
        alpha: float,
        epeak: float,
        norm: float,
    ) -> np.ndarray:
        """
        Cutoff power-law spectral model.

        N(E) = norm * (E/100)^alpha * exp(-(2+alpha)*E/epeak)

        Parameters
        ----------
        energy : np.ndarray
            Photon energy in keV
        alpha : float
            Spectral index
        epeak : float
            Peak energy in keV
        norm : float
            Normalization

        Returns
        -------
        np.ndarray
            Spectral response at given energies
        """
        e0 = 100.0
        return norm * (energy / e0) ** alpha * np.exp(-(2 + alpha) * energy / epeak)

    @staticmethod
    def simple_powerlaw(
        energy: np.ndarray,
        index: float,
        norm: float,
    ) -> np.ndarray:
        """
        Simple power-law spectral model.

        N(E) = norm * (E/100)^index

        Parameters
        ----------
        energy : np.ndarray
            Photon energy in keV
        index : float
            Spectral index
        norm : float
            Normalization

        Returns
        -------
        np.ndarray
            Spectral response at given energies
        """
        e0 = 100.0
        return norm * (energy / e0) ** index

    @staticmethod
    def blackbody(
        energy: np.ndarray,
        kT: float,
        norm: float,
    ) -> np.ndarray:
        """
        Planck (blackbody) function in photon spectrum.

        N(E) = norm * E^2 / (exp(E/kT) - 1)

        Parameters
        ----------
        energy : np.ndarray
            Photon energy in keV
        kT : float
            Temperature in keV
        norm : float
            Normalization

        Returns
        -------
        np.ndarray
            Spectral response at given energies
        """
        e = np.atleast_1d(energy)
        # Avoid overflow
        exp_factor = np.exp(np.clip(e / kT, -100, 100))
        return norm * e ** 2 / (exp_factor - 1)

    @staticmethod
    def smoothly_broken_powerlaw(
        energy: np.ndarray,
        alpha1: float,
        alpha2: float,
        e_break: float,
        norm: float,
        delta: float = 0.3,
    ) -> np.ndarray:
        """
        Smoothly broken power-law model.

        Parameters
        ----------
        energy : np.ndarray
            Photon energy in keV
        alpha1 : float
            Low-energy spectral index
        alpha2 : float
            High-energy spectral index
        e_break : float
            Break energy in keV
        norm : float
            Normalization
        delta : float
            Smoothing parameter (default: 0.3)

        Returns
        -------
        np.ndarray
            Spectral response at given energies
        """
        e0 = 100.0
        x = energy / e_break
        smooth_break = (x ** (-alpha1 * delta) + x ** (-alpha2 * delta)) ** (-1 / delta)
        return norm * (energy / e0) ** (-alpha1) * smooth_break

    def _chi_squared(
        self,
        data: SpectralData,
        model_func,
        params: Tuple,
    ) -> float:
        """
        Calculate chi-squared statistic.

        Parameters
        ----------
        data : SpectralData
            Input spectral data
        model_func : callable
            Model function
        params : tuple
            Model parameters

        Returns
        -------
        float
            Chi-squared value
        """
        model = model_func(data.energy, *params)
        # Avoid division by zero
        expected = model * data.exposure * data.area
        expected = np.maximum(expected, 1e-10)
        chi_sq = ((data.counts - expected) ** 2 / expected).sum()
        return chi_sq

    def _c_statistic(
        self,
        data: SpectralData,
        model_func,
        params: Tuple,
    ) -> float:
        """
        Calculate Cash statistic for low-count data.

        Parameters
        ----------
        data : SpectralData
            Input spectral data
        model_func : callable
            Model function
        params : tuple
            Model parameters

        Returns
        -------
        float
            Cash statistic value
        """
        model = model_func(data.energy, *params)
        expected = model * data.exposure * data.area
        expected = np.maximum(expected, 1e-10)

        # Cash statistic: C = 2 * sum(expected - counts * log(expected))
        c_stat = 2 * (expected - data.counts * np.log(expected)).sum()
        return c_stat

    def fit_spectrum(
        self,
        data: SpectralData,
        model: str = "Band",
    ) -> SpectralFit:
        """
        Fit spectral model to data.

        Tries all models and selects best via AIC/BIC.

        Parameters
        ----------
        data : SpectralData
            Input spectral data
        model : str
            Model to fit: "Band", "CPL", "PL", "BB", "SBPL"

        Returns
        -------
        SpectralFit
            Fit results
        """
        if model == "Band":
            return self.fit_band(data)
        elif model == "CPL":
            return self.fit_cpl(data)
        elif model == "PL":
            return self.fit_powerlaw(data)
        elif model == "BB":
            return self._fit_blackbody(data)
        elif model == "SBPL":
            return self._fit_smoothly_broken_pl(data)
        else:
            raise ValueError(f"Unknown model: {model}")

    def fit_band(self, data: SpectralData) -> SpectralFit:
        """
        Fit Band spectral model specifically.

        Parameters
        ----------
        data : SpectralData
            Input spectral data

        Returns
        -------
        SpectralFit
            Fit results
        """

        def objective(params):
            return self._c_statistic(data, self.band_function, params)

        # Initial guess
        alpha0, beta0, epeak0, norm0 = -1.0, -2.5, 300.0, 1.0

        # Bounds
        bounds = [
            (-3.0, 1.0),  # alpha
            (-5.0, -1.5),  # beta
            (10.0, 10000.0),  # epeak
            (1e-5, 1000.0),  # norm
        ]

        try:
            result = optimize.minimize(
                objective,
                [alpha0, beta0, epeak0, norm0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha, beta, epeak, norm = result.x
                chi_sq = self._c_statistic(data, self.band_function, result.x)
                dof = len(data.energy) - 4

                # Numerical errors
                eps = 1e-5
                hess = np.zeros((4, 4))
                for i in range(4):
                    p_plus = result.x.copy()
                    p_plus[i] += eps
                    p_minus = result.x.copy()
                    p_minus[i] -= eps

                    f_plus = objective(p_plus)
                    f_minus = objective(p_minus)

                    for j in range(4):
                        p2_plus = result.x.copy()
                        p2_plus[j] += eps
                        p2_minus = result.x.copy()
                        p2_minus[j] -= eps

                        if i == j:
                            hess[i, j] = (f_plus + f_minus - 2 * chi_sq) / eps ** 2
                        else:
                            f_pp = self._c_statistic(data, self.band_function, p_plus)
                            f_mm = self._c_statistic(data, self.band_function, p_minus)
                            hess[i, j] = (f_pp - f_plus - f_mm + f_minus) / (4 * eps ** 2)

                try:
                    cov = np.linalg.inv(hess)
                except np.linalg.LinAlgError:
                    cov = np.diag([np.inf] * 4)

                errors = {
                    'alpha': np.sqrt(np.abs(cov[0, 0])),
                    'beta': np.sqrt(np.abs(cov[1, 1])),
                    'epeak': np.sqrt(np.abs(cov[2, 2])),
                    'norm': np.sqrt(np.abs(cov[3, 3])),
                }

                aic = chi_sq + 2 * 4
                bic = chi_sq + 4 * np.log(len(data.energy))

                return SpectralFit(
                    model="Band",
                    params={'alpha': alpha, 'beta': beta, 'epeak': epeak, 'norm': norm},
                    errors=errors,
                    cov_matrix=cov,
                    chi_sq=chi_sq,
                    dof=dof,
                    aic=aic,
                    bic=bic,
                    fit_success=True,
                )
        except Exception as e:
            print(f"Band fit error: {e}")

        return SpectralFit(
            model="Band",
            params={},
            errors={},
            fit_success=False,
        )

    def fit_cpl(self, data: SpectralData) -> SpectralFit:
        """
        Fit cutoff power-law model.

        Parameters
        ----------
        data : SpectralData
            Input spectral data

        Returns
        -------
        SpectralFit
            Fit results
        """

        def objective(params):
            return self._c_statistic(data, self.cutoff_powerlaw, params)

        # Initial guess
        alpha0, epeak0, norm0 = -1.5, 300.0, 1.0

        bounds = [
            (-3.0, 1.0),  # alpha
            (10.0, 10000.0),  # epeak
            (1e-5, 1000.0),  # norm
        ]

        try:
            result = optimize.minimize(
                objective,
                [alpha0, epeak0, norm0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha, epeak, norm = result.x
                chi_sq = objective(result.x)
                dof = len(data.energy) - 3

                # Estimate errors
                eps = 1e-5
                errors = {}
                for i, name in enumerate(['alpha', 'epeak', 'norm']):
                    p_plus = result.x.copy()
                    p_plus[i] += eps
                    p_minus = result.x.copy()
                    p_minus[i] -= eps

                    f_plus = objective(p_plus)
                    f_minus = objective(p_minus)
                    second_deriv = (f_plus - 2 * chi_sq + f_minus) / eps ** 2
                    if second_deriv > 0:
                        errors[name] = 1.0 / np.sqrt(second_deriv)
                    else:
                        errors[name] = np.inf

                aic = chi_sq + 2 * 3
                bic = chi_sq + 3 * np.log(len(data.energy))

                return SpectralFit(
                    model="CPL",
                    params={'alpha': alpha, 'epeak': epeak, 'norm': norm},
                    errors=errors,
                    chi_sq=chi_sq,
                    dof=dof,
                    aic=aic,
                    bic=bic,
                    fit_success=True,
                )
        except Exception as e:
            print(f"CPL fit error: {e}")

        return SpectralFit(
            model="CPL",
            params={},
            errors={},
            fit_success=False,
        )

    def fit_powerlaw(self, data: SpectralData) -> SpectralFit:
        """
        Fit simple power-law model.

        Parameters
        ----------
        data : SpectralData
            Input spectral data

        Returns
        -------
        SpectralFit
            Fit results
        """

        def objective(params):
            return self._c_statistic(data, self.simple_powerlaw, params)

        index0, norm0 = -2.0, 1.0

        bounds = [(-3.0, 1.0), (1e-5, 1000.0)]

        try:
            result = optimize.minimize(
                objective,
                [index0, norm0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                index, norm = result.x
                chi_sq = objective(result.x)
                dof = len(data.energy) - 2

                # Estimate errors
                eps = 1e-5
                errors = {}
                for i, name in enumerate(['index', 'norm']):
                    p_plus = result.x.copy()
                    p_plus[i] += eps
                    p_minus = result.x.copy()
                    p_minus[i] -= eps

                    f_plus = objective(p_plus)
                    f_minus = objective(p_minus)
                    second_deriv = (f_plus - 2 * chi_sq + f_minus) / eps ** 2
                    if second_deriv > 0:
                        errors[name] = 1.0 / np.sqrt(second_deriv)
                    else:
                        errors[name] = np.inf

                aic = chi_sq + 2 * 2
                bic = chi_sq + 2 * np.log(len(data.energy))

                return SpectralFit(
                    model="PL",
                    params={'index': index, 'norm': norm},
                    errors=errors,
                    chi_sq=chi_sq,
                    dof=dof,
                    aic=aic,
                    bic=bic,
                    fit_success=True,
                )
        except Exception as e:
            print(f"PL fit error: {e}")

        return SpectralFit(
            model="PL",
            params={},
            errors={},
            fit_success=False,
        )

    def _fit_blackbody(self, data: SpectralData) -> SpectralFit:
        """Fit blackbody model."""

        def objective(params):
            return self._c_statistic(data, self.blackbody, params)

        kT0, norm0 = 10.0, 1.0

        bounds = [(0.1, 100.0), (1e-5, 1000.0)]

        try:
            result = optimize.minimize(
                objective,
                [kT0, norm0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                kT, norm = result.x
                chi_sq = objective(result.x)
                dof = len(data.energy) - 2

                errors = {'kT': np.inf, 'norm': np.inf}

                aic = chi_sq + 2 * 2
                bic = chi_sq + 2 * np.log(len(data.energy))

                return SpectralFit(
                    model="BB",
                    params={'kT': kT, 'norm': norm},
                    errors=errors,
                    chi_sq=chi_sq,
                    dof=dof,
                    aic=aic,
                    bic=bic,
                    fit_success=True,
                )
        except Exception:
            pass

        return SpectralFit(
            model="BB",
            params={},
            errors={},
            fit_success=False,
        )

    def _fit_smoothly_broken_pl(self, data: SpectralData) -> SpectralFit:
        """Fit smoothly broken power-law model."""

        def objective(params):
            return self._c_statistic(data, self.smoothly_broken_powerlaw, params)

        bounds = [(-3, 1), (-5, -1), (10, 10000), (1e-5, 1000)]

        try:
            result = optimize.minimize(
                objective,
                [-1.5, -2.5, 300, 1.0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                alpha1, alpha2, e_break, norm = result.x
                chi_sq = objective(result.x)
                dof = len(data.energy) - 4

                errors = {
                    'alpha1': np.inf, 'alpha2': np.inf,
                    'e_break': np.inf, 'norm': np.inf
                }

                aic = chi_sq + 2 * 4
                bic = chi_sq + 4 * np.log(len(data.energy))

                return SpectralFit(
                    model="SBPL",
                    params={
                        'alpha1': alpha1, 'alpha2': alpha2,
                        'e_break': e_break, 'norm': norm
                    },
                    errors=errors,
                    chi_sq=chi_sq,
                    dof=dof,
                    aic=aic,
                    bic=bic,
                    fit_success=True,
                )
        except Exception:
            pass

        return SpectralFit(
            model="SBPL",
            params={},
            errors={},
            fit_success=False,
        )

    def calculate_energy_flux(
        self,
        fit: SpectralFit,
        energy_range: Tuple[float, float] = (10, 10000),
    ) -> Tuple[float, float]:
        """
        Calculate energy flux from spectral fit.

        Integrates E*N(E)dE over energy range.

        Parameters
        ----------
        fit : SpectralFit
            Spectral fit results
        energy_range : tuple
            Energy range in keV

        Returns
        -------
        tuple
            (eflux, eflux_err) in erg/cm^2/s
        """
        if not fit.fit_success or not fit.params:
            return 0.0, 0.0

        # Select model function
        if fit.model == "Band":

            def model(e):
                return e * self.band_function(
                    e, fit.params['alpha'], fit.params['beta'],
                    fit.params['epeak'], fit.params['norm']
                )

        elif fit.model == "CPL":

            def model(e):
                return e * self.cutoff_powerlaw(
                    e, fit.params['alpha'],
                    fit.params['epeak'], fit.params['norm']
                )

        elif fit.model == "PL":

            def model(e):
                return e * self.simple_powerlaw(
                    e, fit.params['index'], fit.params['norm']
                )

        else:
            return 0.0, 0.0

        # Integrate (convert keV to erg: 1 keV = 1.602e-9 erg)
        eflux, _ = integrate.quad(
            model, energy_range[0], energy_range[1],
            limit=100
        )

        eflux *= 1.602e-9  # Convert keV to erg

        # Rough error estimate
        eflux_err = eflux * 0.1  # ~10% systematic uncertainty

        return eflux, eflux_err

    def calculate_photon_flux(
        self,
        fit: SpectralFit,
        energy_range: Tuple[float, float] = (10, 10000),
    ) -> Tuple[float, float]:
        """
        Calculate photon flux from spectral fit.

        Parameters
        ----------
        fit : SpectralFit
            Spectral fit results
        energy_range : tuple
            Energy range in keV

        Returns
        -------
        tuple
            (pflux, pflux_err) in photons/cm^2/s
        """
        if not fit.fit_success or not fit.params:
            return 0.0, 0.0

        # Select model
        if fit.model == "Band":

            def model(e):
                return self.band_function(
                    e, fit.params['alpha'], fit.params['beta'],
                    fit.params['epeak'], fit.params['norm']
                )

        elif fit.model == "CPL":

            def model(e):
                return self.cutoff_powerlaw(
                    e, fit.params['alpha'],
                    fit.params['epeak'], fit.params['norm']
                )

        elif fit.model == "PL":

            def model(e):
                return self.simple_powerlaw(
                    e, fit.params['index'], fit.params['norm']
                )

        else:
            return 0.0, 0.0

        pflux, _ = integrate.quad(model, energy_range[0], energy_range[1], limit=100)
        pflux_err = pflux * 0.1

        return pflux, pflux_err

    def calculate_epeak(
        self,
        alpha: float,
        beta_or_ecut: float,
        model: str = "Band",
    ) -> float:
        """
        Calculate Epeak from fit parameters.

        Parameters
        ----------
        alpha : float
            Low-energy index or simple index
        beta_or_ecut : float
            High-energy index (Band) or cutoff energy (CPL)
        model : str
            Model type: "Band" or "CPL"

        Returns
        -------
        float
            Epeak in keV
        """
        if model == "Band":
            # For Band, Epeak is the input parameter
            return (alpha - beta_or_ecut) * beta_or_ecut / (2 + alpha)
        else:
            # For CPL, Epeak is roughly the cutoff
            return beta_or_ecut

    def calculate_eiso(
        self,
        fit: SpectralFit,
        redshift: float,
        luminosity_distance: float = None,
    ) -> float:
        """
        Calculate isotropic equivalent energy (Eiso).

        Parameters
        ----------
        fit : SpectralFit
            Spectral fit results
        redshift : float
            Source redshift
        luminosity_distance : float, optional
            Luminosity distance in cm (auto-calculate if None)

        Returns
        -------
        float
            Eiso in erg
        """
        if luminosity_distance is None:
            luminosity_distance = self._luminosity_distance(redshift)

        eflux, _ = self.calculate_energy_flux(fit, (10, 10000))

        # Need time duration (approximate as 100 seconds for now)
        duration = 100.0

        # Eiso = 4*pi*D_L^2 * eflux * duration
        eiso = 4 * np.pi * luminosity_distance ** 2 * eflux * duration

        return eiso

    def calculate_lpeak(
        self,
        peak_flux: float,
        redshift: float,
        luminosity_distance: float = None,
    ) -> float:
        """
        Calculate peak luminosity.

        Parameters
        ----------
        peak_flux : float
            Peak flux in erg/cm^2/s
        redshift : float
            Source redshift
        luminosity_distance : float, optional
            Luminosity distance in cm

        Returns
        -------
        float
            Peak luminosity in erg/s
        """
        if luminosity_distance is None:
            luminosity_distance = self._luminosity_distance(redshift)

        lpeak = 4 * np.pi * luminosity_distance ** 2 * peak_flux * (1 + redshift)

        return lpeak

    def model_comparison(self, fits: List[SpectralFit]) -> SpectralFit:
        """
        Select best model using AIC/BIC.

        Parameters
        ----------
        fits : list of SpectralFit
            List of fit results

        Returns
        -------
        SpectralFit
            Best fit
        """
        valid_fits = [f for f in fits if f.fit_success and f.aic is not None]

        if not valid_fits:
            return fits[0]

        # Use AIC for model comparison
        best_fit = min(valid_fits, key=lambda f: f.aic)
        return best_fit

    def _luminosity_distance(self, redshift: float) -> float:
        """
        Calculate luminosity distance in cm using flat LambdaCDM.

        Parameters
        ----------
        redshift : float
            Source redshift

        Returns
        -------
        float
            Luminosity distance in cm
        """
        # Comoving distance integral
        def integrand(z):
            ez = np.sqrt(self.Om0 * (1 + z) ** 3 + self.OL0)
            return 1.0 / ez

        if redshift == 0:
            return 0.0

        # Speed of light in cm/s
        c = 2.998e10

        # Hubble constant in cm/s/cm (convert from km/s/Mpc)
        h0_cgs = self.H0 * 1e5 / 3.086e24

        result, _ = integrate.quad(integrand, 0, redshift, limit=100)
        comoving_distance = c * result / h0_cgs

        # Luminosity distance
        luminosity_distance = comoving_distance * (1 + redshift)

        return luminosity_distance

    # ------------------------------------------------------------------
    # Flux density conversion (from SwiftXRT_Fd_atEkeV.ipynb)
    # ------------------------------------------------------------------

    @staticmethod
    def flux_density_convert(
        E0: float,
        E: float,
        photon_index: float,
    ) -> float:
        """
        Convert flux density from one energy to another using a power-law spectrum.

        Given flux density at energy E0, returns the conversion factor to get
        flux density at energy E, assuming a power-law spectrum with the given
        photon index:  F_nu ~ E^{-(Gamma-1)}

        From Vikas Chand's SwiftXRT_Fd_atEkeV notebook.

        Parameters
        ----------
        E0 : float
            Reference energy in keV (where flux density is known)
        E : float
            Target energy in keV (where flux density is wanted)
        photon_index : float
            Photon index Gamma (N(E) ~ E^{-Gamma})

        Returns
        -------
        float
            Conversion factor: multiply flux_density@E0 by this to get flux_density@E

        Examples
        --------
        >>> SpectralAnalyzer.flux_density_convert(10, 1, 2.0)
        10.0
        >>> # Flux density at 1 keV = 10 * flux density at 10 keV for Gamma=2
        """
        beta = photon_index - 1  # energy spectral index
        # Convert @E0 to @1 keV, then @1 keV to @E
        factor_to_1keV = E0 ** beta
        factor_from_1keV = E ** (-beta)
        return factor_to_1keV * factor_from_1keV

    @staticmethod
    def flux_density_at_energy(
        flux_density: float,
        E0: float,
        E: float,
        photon_index: float,
    ) -> float:
        """
        Get flux density at energy E given flux density at energy E0.

        Parameters
        ----------
        flux_density : float
            Known flux density at E0 (in any units, e.g. Jy or mJy)
        E0 : float
            Energy where flux density is known (keV)
        E : float
            Energy where flux density is wanted (keV)
        photon_index : float
            Photon index

        Returns
        -------
        float
            Flux density at energy E (same units as input)
        """
        factor = SpectralAnalyzer.flux_density_convert(E0, E, photon_index)
        return flux_density * factor


@dataclass
class XRTFluxDensityData:
    """Container for Swift XRT flux density data with photon indices.

    Holds time-resolved flux density measurements and photon indices
    from Swift XRT spectral analysis, typically from the UK Swift Science
    Data Centre automated pipeline.
    """
    time: np.ndarray              # Mid-times since trigger (s)
    time_err_pos: np.ndarray      # Positive time errors
    time_err_neg: np.ndarray      # Negative time errors
    flux_density: np.ndarray      # Flux density (Jy or mJy)
    flux_density_err_pos: np.ndarray
    flux_density_err_neg: np.ndarray
    photon_index: np.ndarray      # Time-resolved photon index
    photon_index_err_pos: np.ndarray
    photon_index_err_neg: np.ndarray
    reference_energy: float = 10.0  # Energy at which flux density is measured (keV)


class XRTFluxDensity:
    """
    Swift XRT flux density analysis.

    Reads XRT flux density and photon index files (as produced by the
    UK Swift Science Data Centre), and converts flux density to any
    desired energy using the time-resolved photon index.

    From Vikas Chand's SwiftXRT_Fd_atEkeV notebook.

    Usage
    -----
    >>> xrt = XRTFluxDensity()
    >>> data = xrt.read_data('GRB201015A_Fluxdensity10.txt', 'GRB201015A_phindex.txt')
    >>> fd_1keV = xrt.convert_to_energy(data, target_energy=1.0)
    """

    @staticmethod
    def read_data(
        flux_density_file: str,
        photon_index_file: str,
        reference_energy: float = 10.0,
    ) -> XRTFluxDensityData:
        """
        Read XRT flux density and photon index data files.

        Expected file format (whitespace-separated, no header):
            col1: time
            col2: time_err_pos
            col3: time_err_neg
            col4: value
            col5: value_err_pos
            col6: value_err_neg

        Parameters
        ----------
        flux_density_file : str
            Path to flux density data file
        photon_index_file : str
            Path to photon index data file
        reference_energy : float
            Energy (keV) at which flux density is measured (default: 10)

        Returns
        -------
        XRTFluxDensityData
            Combined flux density and photon index data
        """
        # Read flux density
        fd = np.loadtxt(flux_density_file)
        sort_fd = np.argsort(fd[:, 0])
        fd = fd[sort_fd]

        # Read photon index
        ph = np.loadtxt(photon_index_file)
        sort_ph = np.argsort(ph[:, 0])
        ph = ph[sort_ph]

        return XRTFluxDensityData(
            time=fd[:, 0],
            time_err_pos=fd[:, 1],
            time_err_neg=fd[:, 2],
            flux_density=fd[:, 3],
            flux_density_err_pos=fd[:, 4],
            flux_density_err_neg=fd[:, 5],
            photon_index=ph[:, 3],
            photon_index_err_pos=ph[:, 4],
            photon_index_err_neg=ph[:, 5],
            reference_energy=reference_energy,
        )

    @staticmethod
    def convert_to_energy(
        data: XRTFluxDensityData,
        target_energy: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert flux density from reference energy to target energy.

        Uses time-resolved photon indices for proper conversion at each epoch.

        Parameters
        ----------
        data : XRTFluxDensityData
            Input data with flux density and photon indices
        target_energy : float
            Target energy in keV

        Returns
        -------
        tuple of (flux_density, flux_density_err_pos, flux_density_err_neg)
            Converted flux density and errors at target energy
        """
        factor = np.array([
            SpectralAnalyzer.flux_density_convert(
                data.reference_energy, target_energy, gamma
            )
            for gamma in data.photon_index
        ])

        fd_converted = data.flux_density * factor
        fd_err_pos = data.flux_density_err_pos * factor
        fd_err_neg = data.flux_density_err_neg * factor

        return fd_converted, fd_err_pos, fd_err_neg

    @staticmethod
    def save_converted(
        data: XRTFluxDensityData,
        target_energy: float,
        output_file: str,
    ) -> None:
        """
        Convert flux density and save to a new file.

        Parameters
        ----------
        data : XRTFluxDensityData
            Input data
        target_energy : float
            Target energy in keV
        output_file : str
            Output file path
        """
        fd, fd_ep, fd_en = XRTFluxDensity.convert_to_energy(data, target_energy)

        output = np.column_stack([
            data.time, data.time_err_neg, data.time_err_pos,
            fd, fd_en, fd_ep,
        ])

        header = f"time terr_n terr_p Fd@{target_energy}keV Fd_n Fd_p"
        np.savetxt(output_file, output, header=header)
