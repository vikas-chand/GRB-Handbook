"""GRB classification module.

Provides heuristic classification based on duration and hardness ratio,
as well as a proper statistical Gaussian Mixture Model (GMM) approach for
classifying GRBs in the T90-Hardness Ratio plane following the methodology
commonly used in Fermi/GBM catalog analyses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
from enum import Enum
import logging
import numpy as np

from .lightcurve import LightCurveData
from .spectral import SpectralFit

try:
    from sklearn import mixture as sklearn_mixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class GRBClass(Enum):
    """GRB classification categories."""
    SHORT = "Short"
    LONG = "Long"
    ULTRA_LONG = "Ultra-long"
    UNKNOWN = "Unknown"


@dataclass
class GMMResult:
    """Result container for a Gaussian Mixture Model fit on the T90-HR plane.

    Attributes
    ----------
    gmm_model : object
        The fitted ``sklearn.mixture.GaussianMixture`` instance.
    weights : np.ndarray
        Mixing weights of each component, shape ``(n_components,)``.
    means : np.ndarray
        Mean of each Gaussian component, shape ``(n_components, 2)``.
        Columns correspond to [log_t90, log_hr].
    covariances : np.ndarray
        Covariance matrices, shape ``(n_components, 2, 2)`` for
        ``covariance_type='full'``.
    labels : np.ndarray
        Predicted component labels for the training data.
    n_components : int
        Number of Gaussian components in the mixture.
    """

    gmm_model: Any
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    labels: np.ndarray
    n_components: int


class GRBClassifier:
    """Classifier for GRB events.

    Supports simple heuristic classification by T90 duration and hardness
    ratio, as well as a statistically rigorous Gaussian Mixture Model (GMM)
    classification on the log(T90) -- log(HR) plane.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with classification thresholds.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize GRB classifier.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with classification thresholds
        """
        self.config = config or {}
        self.t90_short_long = self.config.get('t90_short_long', 2.0)  # seconds
        self.t90_long_ultralong = self.config.get('t90_long_ultralong', 10000.0)  # seconds
        self.hardness_threshold = self.config.get('hardness_threshold', 0.3)
        self.extended_emission_threshold = self.config.get('extended_emission_threshold', 0.1)

        # GMM state --------------------------------------------------------
        self._gmm_result: Optional[GMMResult] = None
        # Index of the Gaussian component that corresponds to short GRBs.
        # Determined automatically after fitting by inspecting the component
        # means: the component with the *smaller* mean log(T90) is "short".
        self._short_component_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Heuristic classification methods (unchanged)
    # ------------------------------------------------------------------

    def classify_by_duration(self, t90: float) -> GRBClass:
        """
        Simple classification based on T90 duration.

        Parameters
        ----------
        t90 : float
            T90 duration in seconds

        Returns
        -------
        GRBClass
            Classification: SHORT, LONG, or ULTRA_LONG
        """
        if t90 < self.t90_short_long:
            return GRBClass.SHORT
        elif t90 < self.t90_long_ultralong:
            return GRBClass.LONG
        else:
            return GRBClass.ULTRA_LONG

    def classify_by_hardness(
        self,
        hardness_ratio: float,
        t90: float,
    ) -> GRBClass:
        """
        Classification using hardness-duration plane.

        Uses combined hardness and duration to classify.

        Parameters
        ----------
        hardness_ratio : float
            Hardness ratio (hard - soft) / (hard + soft)
        t90 : float
            T90 duration in seconds

        Returns
        -------
        GRBClass
            Classification
        """
        # Duration classification as baseline
        duration_class = self.classify_by_duration(t90)

        # Hardness correction
        # Longer GRBs tend to be softer; short GRBs harder
        # Use hardness to refine classification
        if duration_class == GRBClass.SHORT:
            # Short GRBs are typically harder (higher HR)
            if hardness_ratio > self.hardness_threshold:
                return GRBClass.SHORT
            else:
                # Soft short GRB - possibly X-ray flash
                return GRBClass.SHORT
        elif duration_class == GRBClass.LONG:
            # Long GRBs are typically softer
            return GRBClass.LONG
        else:
            return GRBClass.ULTRA_LONG

    def classify_extended_emission(self, lc: LightCurveData) -> bool:
        """
        Detect extended emission in short GRBs.

        Extended emission is a short burst followed by softer, extended emission.

        Parameters
        ----------
        lc : LightCurveData
            Light curve data

        Returns
        -------
        bool
            True if extended emission is detected
        """
        if len(lc.rate) < 5:
            return False

        # Divide light curve into early and late phases
        n = len(lc.rate)
        early_idx = n // 3
        late_idx = 2 * n // 3

        if early_idx < 2 or late_idx < 3:
            return False

        # Calculate mean rates
        early_rate = lc.rate[:early_idx].mean()
        middle_rate = lc.rate[early_idx:late_idx].mean()
        late_rate = lc.rate[late_idx:].mean()

        # Check for pattern: peak early, then extended tail
        # Early should be brightest, late should be dim but non-zero
        if early_rate > 0 and late_rate > 0:
            ratio = late_rate / early_rate
            if 0.01 < ratio < 0.3 and middle_rate < early_rate:
                return True

        return False

    def classify_supernova_association(self, grb_data: Dict) -> bool:
        """
        Check for supernova association markers.

        Checks for properties consistent with core-collapse supernova origin.

        Parameters
        ----------
        grb_data : dict
            GRB data including spectral and temporal properties

        Returns
        -------
        bool
            True if SN association markers are present
        """
        markers = 0

        # Long GRBs are associated with SNe
        if grb_data.get('grb_class') == GRBClass.LONG:
            markers += 1

        # X-ray plateau suggests SN-powered afterglow
        if grb_data.get('has_plateau', False):
            markers += 1

        # Soft spectrum (low Epeak) associated with SNe
        epeak = grb_data.get('epeak_rest', 300)
        if epeak < 200:
            markers += 1

        # Low redshift
        z = grb_data.get('redshift', 10)
        if z < 2.0:
            markers += 1

        # At least 2 markers suggest SN association
        return markers >= 2

    def classify_kilonova_association(self, grb_data: Dict) -> bool:
        """
        Check for kilonova association markers.

        Checks for properties consistent with compact object merger.

        Parameters
        ----------
        grb_data : dict
            GRB data

        Returns
        -------
        bool
            True if kilonova association markers present
        """
        markers = 0

        # Short GRBs associated with mergers and kilonovae
        if grb_data.get('grb_class') == GRBClass.SHORT:
            markers += 1

        # Extended emission in short GRB suggests merger
        if grb_data.get('has_extended_emission', False):
            markers += 1

        # Short, soft pulses in spectrum
        epeak = grb_data.get('epeak_rest', 300)
        if epeak < 100:
            markers += 1

        # Weak afterglow (limited jet energy collimation)
        if grb_data.get('weak_afterglow', False):
            markers += 1

        return markers >= 2

    def full_classification(
        self,
        grb_data: Dict,
        lc: Optional[LightCurveData] = None,
        spectral_fit: Optional[SpectralFit] = None,
    ) -> Dict:
        """
        Comprehensive multi-criteria classification.

        Parameters
        ----------
        grb_data : dict
            GRB parameters (including t90, hardness ratio, redshift, etc.)
        lc : LightCurveData, optional
            Light curve data
        spectral_fit : SpectralFit, optional
            Spectral fit results

        Returns
        -------
        dict
            Comprehensive classification results
        """
        results = {
            'grb_name': grb_data.get('name', 'Unknown'),
        }

        # Duration-based classification
        t90 = grb_data.get('t90', None)
        if t90 is not None:
            grb_class = self.classify_by_duration(t90)
            results['duration_class'] = grb_class
            results['t90'] = t90
        else:
            grb_class = GRBClass.UNKNOWN
            results['duration_class'] = GRBClass.UNKNOWN

        # Hardness-duration classification
        hr = grb_data.get('hardness_ratio', None)
        if hr is not None and t90 is not None:
            hr_class = self.classify_by_hardness(hr, t90)
            results['hardness_class'] = hr_class
            results['hardness_ratio'] = hr
        else:
            results['hardness_class'] = grb_class

        # Extended emission detection
        has_ee = False
        if lc is not None:
            has_ee = self.classify_extended_emission(lc)
            results['has_extended_emission'] = has_ee
            grb_data['has_extended_emission'] = has_ee

        # Supernova association
        grb_data['grb_class'] = grb_class
        has_sn = self.classify_supernova_association(grb_data)
        results['supernova_association'] = has_sn

        # Kilonova association
        has_kn = self.classify_kilonova_association(grb_data)
        results['kilonova_association'] = has_kn

        # Magnetar candidate
        is_magnetar = self.is_magnetar_candidate(grb_data)
        results['magnetar_candidate'] = is_magnetar

        # Collapsar check
        is_collapse = self.is_collapsar(grb_data)
        results['collapsar'] = is_collapse

        # Compact merger check
        is_merger = self.is_compact_merger(grb_data)
        results['compact_merger'] = is_merger

        # Final classification confidence
        confidence = self.classification_confidence(results)
        results['confidence'] = confidence

        # Best estimate for progenitor
        if is_merger:
            results['best_progenitor'] = "Compact object merger (NS-NS or NS-BH)"
        elif is_magnetar:
            results['best_progenitor'] = "Magnetar-powered jet"
        elif is_collapse:
            results['best_progenitor'] = "Massive star core collapse"
        else:
            if grb_class == GRBClass.SHORT:
                results['best_progenitor'] = "Compact object merger (likely)"
            elif grb_class == GRBClass.LONG:
                results['best_progenitor'] = "Massive star core collapse (likely)"
            else:
                results['best_progenitor'] = "Unknown"

        return results

    def is_magnetar_candidate(
        self,
        grb_data: Dict,
        afterglow_params: Optional[Dict] = None,
    ) -> bool:
        """
        Check if GRB is candidate for magnetar-powered jet.

        Indicators:
        - X-ray plateau phase
        - Shallow decay index
        - Energy injection

        Parameters
        ----------
        grb_data : dict
            GRB parameters
        afterglow_params : dict, optional
            Afterglow fit parameters

        Returns
        -------
        bool
            True if magnetar candidate
        """
        markers = 0

        # X-ray plateau phase
        if grb_data.get('has_plateau', False):
            markers += 2

        # Shallow decay in X-ray (alpha < 0.5)
        if afterglow_params is not None:
            alpha = afterglow_params.get('decay_index', 1.0)
            if alpha < 0.5:
                markers += 1

        # Energy injection signature
        if grb_data.get('has_energy_injection', False):
            markers += 1

        # Typical for long GRBs
        if grb_data.get('grb_class') == GRBClass.LONG:
            pass  # Neutral
        else:
            markers -= 1

        return markers >= 2

    def is_collapsar(self, grb_data: Dict) -> bool:
        """
        Check if GRB is consistent with massive star core collapse (collapsar).

        Parameters
        ----------
        grb_data : dict
            GRB parameters

        Returns
        -------
        bool
            True if collapsar-like
        """
        markers = 0

        # Long duration
        if grb_data.get('grb_class') == GRBClass.LONG:
            markers += 2

        # Supernova association
        if grb_data.get('supernova_association', False):
            markers += 2

        # Low redshift (nearby)
        z = grb_data.get('redshift', 10)
        if z < 3.0:
            markers += 1

        # Soft spectrum (low Epeak)
        epeak = grb_data.get('epeak_rest', 300)
        if epeak < 150:
            markers += 1

        return markers >= 2

    def is_compact_merger(self, grb_data: Dict) -> bool:
        """
        Check if GRB is consistent with compact object merger (NS-NS or NS-BH).

        Parameters
        ----------
        grb_data : dict
            GRB parameters

        Returns
        -------
        bool
            True if merger-like
        """
        markers = 0

        # Short duration
        if grb_data.get('grb_class') == GRBClass.SHORT:
            markers += 2

        # Kilonova association
        if grb_data.get('kilonova_association', False):
            markers += 2

        # Extended emission
        if grb_data.get('has_extended_emission', False):
            markers += 1

        # Hard spectrum (high Epeak for short GRBs)
        epeak = grb_data.get('epeak_rest', 300)
        if grb_data.get('grb_class') == GRBClass.SHORT and epeak > 200:
            markers += 1

        return markers >= 2

    def classification_confidence(self, results: Dict) -> float:
        """
        Calculate overall confidence in classification.

        Returns value between 0 and 1.

        Parameters
        ----------
        results : dict
            Classification results from full_classification

        Returns
        -------
        float
            Confidence score [0, 1]
        """
        score = 0.0
        weight_sum = 0.0

        # Duration classification confidence (0.3 weight)
        if results.get('duration_class') != GRBClass.UNKNOWN:
            score += 0.3
            weight_sum += 0.3

        # Hardness classification (0.2 weight)
        if results.get('hardness_ratio') is not None:
            score += 0.2
            weight_sum += 0.2

        # Extended emission detection (0.15 weight)
        if 'has_extended_emission' in results:
            score += 0.15
            weight_sum += 0.15

        # Spectral properties (0.15 weight)
        if 'epeak_rest' in results or results.get('spectral_fit') is not None:
            score += 0.15
            weight_sum += 0.15

        # Redshift information (0.1 weight)
        if 'redshift' in results:
            score += 0.1
            weight_sum += 0.1

        # Afterglow properties (0.1 weight)
        if 'has_plateau' in results or 'has_jet_break' in results:
            score += 0.1
            weight_sum += 0.1

        if weight_sum == 0:
            return 0.0

        confidence = score / weight_sum

        # Adjust for consistency
        # Check if progenitor assignments are consistent
        n_markers = sum([
            results.get('supernova_association', False),
            results.get('kilonova_association', False),
            results.get('magnetar_candidate', False),
            results.get('collapsar', False),
            results.get('compact_merger', False),
        ])

        if n_markers > 1:
            confidence *= 0.8  # Reduce confidence if multiple markers
        elif n_markers == 1:
            confidence *= 1.1  # Increase if single clear marker
        else:
            confidence *= 0.6  # Lower if no markers

        return np.clip(confidence, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Gaussian Mixture Model classification on the T90-HR plane
    # ------------------------------------------------------------------

    def fit_gmm(
        self,
        log_t90: np.ndarray,
        log_hr: np.ndarray,
        n_components: int = 2,
        covariance_type: str = "full",
    ) -> GMMResult:
        """Fit a Gaussian Mixture Model to the log(T90) vs log(HR) catalog data.

        This implements the standard two-component GMM separation of short and
        long GRBs on the T90--Hardness Ratio plane, following the approach
        widely used in Fermi/GBM catalog analyses.

        Parameters
        ----------
        log_t90 : np.ndarray
            1-D array of log10(T90) values for the full catalog.
        log_hr : np.ndarray
            1-D array of log10(Hardness Ratio) values for the full catalog.
            Must have the same length as *log_t90*.
        n_components : int, optional
            Number of Gaussian components (default 2: short + long).
        covariance_type : str, optional
            Covariance parameterisation passed to
            ``sklearn.mixture.GaussianMixture``.  One of ``'full'``,
            ``'tied'``, ``'diag'``, or ``'spherical'`` (default ``'full'``).

        Returns
        -------
        GMMResult
            Dataclass holding the fitted model and its parameters.

        Raises
        ------
        ImportError
            If scikit-learn is not installed.
        ValueError
            If *log_t90* and *log_hr* have mismatched lengths.
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for GMM classification. "
                "Install it with: pip install scikit-learn"
            )

        log_t90 = np.asarray(log_t90, dtype=np.float64)
        log_hr = np.asarray(log_hr, dtype=np.float64)

        if log_t90.shape[0] != log_hr.shape[0]:
            raise ValueError(
                f"log_t90 and log_hr must have the same length, "
                f"got {log_t90.shape[0]} and {log_hr.shape[0]}"
            )

        # Stack into (N, 2) feature matrix: columns are [log_t90, log_hr]
        X = np.column_stack([log_t90, log_hr])

        gmm = sklearn_mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
        )
        gmm.fit(X)

        labels = gmm.predict(X)

        result = GMMResult(
            gmm_model=gmm,
            weights=gmm.weights_.copy(),
            means=gmm.means_.copy(),
            covariances=gmm.covariances_.copy(),
            labels=labels,
            n_components=n_components,
        )

        self._gmm_result = result

        # Identify which component is "short": the one with the smaller
        # mean log(T90).  Column 0 of means is log_t90.
        self._short_component_idx = int(np.argmin(result.means[:, 0]))

        logger.info(
            "GMM fit complete: %d components, short-GRB component index = %d",
            n_components,
            self._short_component_idx,
        )

        return result

    def predict_gmm(
        self,
        log_t90: np.ndarray,
        log_hr: np.ndarray,
    ) -> np.ndarray:
        """Predict GMM component labels for given log(T90) and log(HR) values.

        Parameters
        ----------
        log_t90 : np.ndarray
            1-D array of log10(T90) values.
        log_hr : np.ndarray
            1-D array of log10(Hardness Ratio) values.

        Returns
        -------
        np.ndarray
            Integer component labels (same shape as *log_t90*).

        Raises
        ------
        RuntimeError
            If the GMM has not been fitted yet (call ``fit_gmm`` first).
        """
        if self._gmm_result is None:
            raise RuntimeError(
                "No fitted GMM available. Call fit_gmm() first."
            )

        log_t90 = np.asarray(log_t90, dtype=np.float64)
        log_hr = np.asarray(log_hr, dtype=np.float64)
        X = np.column_stack([log_t90, log_hr])

        return self._gmm_result.gmm_model.predict(X)

    def short_grb_probability(
        self,
        log_t90: np.ndarray,
        log_hr: np.ndarray,
    ) -> np.ndarray:
        """Return the probability that each data point belongs to the short-GRB component.

        Uses the fitted GMM's ``predict_proba`` to obtain posterior
        membership probabilities and returns the column corresponding to the
        short-GRB Gaussian component.

        Parameters
        ----------
        log_t90 : np.ndarray
            1-D array (or scalar) of log10(T90).
        log_hr : np.ndarray
            1-D array (or scalar) of log10(Hardness Ratio).

        Returns
        -------
        np.ndarray
            Probability of belonging to the short-GRB component, same shape
            as the input arrays.

        Raises
        ------
        RuntimeError
            If the GMM has not been fitted yet.
        """
        if self._gmm_result is None:
            raise RuntimeError(
                "No fitted GMM available. Call fit_gmm() first."
            )

        log_t90 = np.atleast_1d(np.asarray(log_t90, dtype=np.float64))
        log_hr = np.atleast_1d(np.asarray(log_hr, dtype=np.float64))
        X = np.column_stack([log_t90, log_hr])

        proba = self._gmm_result.gmm_model.predict_proba(X)
        return proba[:, self._short_component_idx]

    def classify_gmm(
        self,
        t90: float,
        hardness_ratio: float,
    ) -> GRBClass:
        """Classify a single GRB as SHORT or LONG using the fitted GMM.

        The classification is based on the posterior probability from the GMM.
        If the probability of the short-GRB component exceeds 0.5, the burst
        is classified as SHORT; otherwise LONG.

        Parameters
        ----------
        t90 : float
            T90 duration in seconds (will be log10-transformed internally).
        hardness_ratio : float
            Hardness ratio (will be log10-transformed internally).

        Returns
        -------
        GRBClass
            ``GRBClass.SHORT`` or ``GRBClass.LONG``.

        Raises
        ------
        RuntimeError
            If the GMM has not been fitted yet.
        ValueError
            If *t90* or *hardness_ratio* is non-positive (cannot take log).
        """
        if self._gmm_result is None:
            raise RuntimeError(
                "No fitted GMM available. Call fit_gmm() first."
            )

        if t90 <= 0:
            raise ValueError(f"t90 must be positive, got {t90}")
        if hardness_ratio <= 0:
            raise ValueError(
                f"hardness_ratio must be positive, got {hardness_ratio}"
            )

        log_t90 = np.log10(t90)
        log_hr = np.log10(hardness_ratio)

        prob_short = self.short_grb_probability(
            np.array([log_t90]),
            np.array([log_hr]),
        )[0]

        if prob_short >= 0.5:
            return GRBClass.SHORT
        else:
            return GRBClass.LONG

    def plot_gmm_classification(
        self,
        log_t90_catalog: np.ndarray,
        log_hr_catalog: np.ndarray,
        target_grb: Optional[Tuple[float, float]] = None,
        grid_resolution: int = 200,
        contour_levels: Optional[List[float]] = None,
        ax: Optional[Any] = None,
    ):
        """Create a T90-HR scatter plot with GMM classification contours.

        Produces a publication-quality figure showing:
        - catalog bursts coloured by their GMM cluster assignment,
        - probability contours at specified levels (default 0.25, 0.5, 0.75),
        - an optional marker for a target GRB of interest.

        Parameters
        ----------
        log_t90_catalog : np.ndarray
            1-D array of log10(T90) for the catalog.
        log_hr_catalog : np.ndarray
            1-D array of log10(HR) for the catalog.
        target_grb : tuple of (float, float), optional
            ``(log_t90, log_hr)`` of a GRB to highlight on the plot.
        grid_resolution : int, optional
            Number of grid points along each axis for the probability
            surface (default 200).
        contour_levels : list of float, optional
            Short-GRB probability levels at which to draw contours
            (default ``[0.25, 0.5, 0.75]``).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If *None*, a new figure and axes are created.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        RuntimeError
            If the GMM has not been fitted yet.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        if self._gmm_result is None:
            raise RuntimeError(
                "No fitted GMM available. Call fit_gmm() first."
            )

        if contour_levels is None:
            contour_levels = [0.25, 0.50, 0.75]

        log_t90_catalog = np.asarray(log_t90_catalog, dtype=np.float64)
        log_hr_catalog = np.asarray(log_hr_catalog, dtype=np.float64)

        # Predict labels for colouring the scatter plot
        labels = self.predict_gmm(log_t90_catalog, log_hr_catalog)

        # Build probability grid
        t90_min, t90_max = log_t90_catalog.min(), log_t90_catalog.max()
        hr_min, hr_max = log_hr_catalog.min(), log_hr_catalog.max()

        # Pad the grid slightly beyond the data range
        t90_pad = 0.05 * (t90_max - t90_min)
        hr_pad = 0.05 * (hr_max - hr_min)

        xx = np.linspace(t90_min - t90_pad, t90_max + t90_pad, grid_resolution)
        yy = np.linspace(hr_min - hr_pad, hr_max + hr_pad, grid_resolution)
        XX, YY = np.meshgrid(xx, yy)
        XY = np.column_stack([XX.ravel(), YY.ravel()])

        # Short-GRB probability on the grid
        proba = self._gmm_result.gmm_model.predict_proba(XY)
        ZZ_short = proba[:, self._short_component_idx].reshape(XX.shape)

        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        # Probability colour mesh (background)
        mesh = ax.pcolormesh(
            XX, YY, ZZ_short,
            cmap="RdYlBu",
            shading="auto",
            alpha=0.30,
        )

        # Contour lines at specified probability levels
        cs = ax.contour(
            XX, YY, ZZ_short,
            levels=sorted(contour_levels),
            colors="k",
            linewidths=1.0,
            linestyles="--",
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")

        # Scatter: colour by GMM label
        short_mask = labels == self._short_component_idx
        long_mask = ~short_mask

        ax.scatter(
            log_t90_catalog[short_mask],
            log_hr_catalog[short_mask],
            c="royalblue",
            s=12,
            alpha=0.6,
            label="Short GRBs (GMM)",
            edgecolors="none",
        )
        ax.scatter(
            log_t90_catalog[long_mask],
            log_hr_catalog[long_mask],
            c="firebrick",
            s=12,
            alpha=0.6,
            label="Long GRBs (GMM)",
            edgecolors="none",
        )

        # Optionally mark target GRB
        if target_grb is not None:
            tgt_log_t90, tgt_log_hr = target_grb
            prob = self.short_grb_probability(
                np.array([tgt_log_t90]),
                np.array([tgt_log_hr]),
            )[0]
            label_str = (
                f"Target GRB (P_short={prob:.2f})"
            )
            ax.scatter(
                [tgt_log_t90],
                [tgt_log_hr],
                marker="*",
                s=300,
                c="gold",
                edgecolors="black",
                linewidths=1.2,
                zorder=10,
                label=label_str,
            )

        # Labels and legend
        ax.set_xlabel(r"$\log_{10}(T_{90}\ /\ \mathrm{s})$", fontsize=13)
        ax.set_ylabel(r"$\log_{10}(\mathrm{Hardness\ Ratio})$", fontsize=13)
        ax.set_title("GMM Classification on the T90 -- HR Plane", fontsize=14)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("P(Short GRB)", fontsize=12)

        fig.tight_layout()
        return fig
