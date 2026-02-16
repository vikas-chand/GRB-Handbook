"""GRB classification module."""

from typing import Dict, Optional, List
from enum import Enum
import numpy as np

from .lightcurve import LightCurveData
from .spectral import SpectralFit


class GRBClass(Enum):
    """GRB classification categories."""
    SHORT = "Short"
    LONG = "Long"
    ULTRA_LONG = "Ultra-long"
    UNKNOWN = "Unknown"


class GRBClassifier:
    """Classifier for GRB events."""

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
