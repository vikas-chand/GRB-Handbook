"""
Automated Fermi GBM analysis pipeline.

Provides end-to-end GBM burst analysis:
    1. Data discovery — find TTE/CSPEC/trigdat files
    2. Trigger info — extract from trigdat header
    3. Detector selection — auto-select best NaI + BGO detectors
    4. Light curve extraction — from TTE/CSPEC with energy selection
    5. Burst interval detection — 4 algorithms for Tstart/Tstop
    6. Background modeling — polynomial fitting with BIC-based order selection
    7. T90/T50 calculation — from cumulative counts
    8. Interactive review — show results, ask user to confirm/adjust
    9. Spectral fitting setup — 3ML (threeML) script generation with Band/CPL/PL models

Improvements over gtburst:
    - BIC-optimized polynomial order (gtburst uses fixed order 2)
    - 4 burst detection algorithms (Bayesian blocks, S/N, cumulative, combined)
    - Interactive review step with user confirmation
    - 3ML-based spectral fitting (supports Bayesian + MLE, pyXSPEC backend)
    - Pipeline integration with database logging

Usage:
    from grb_pipeline.analysis.gbm_analysis import GBMAnalyzer

    analyzer = GBMAnalyzer(data_dir='./DATA/GRB240825A/GBM')
    result = analyzer.run(interactive=True)
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class BurstInterval:
    """Detected burst time interval."""
    tstart: float           # Start time relative to trigger (s)
    tstop: float            # Stop time relative to trigger (s)
    method: str             # Detection method used
    snr: float = 0.0        # Signal-to-noise ratio of detection
    confidence: float = 0.0 # Confidence level (0-1)

    @property
    def duration(self) -> float:
        return self.tstop - self.tstart


@dataclass
class BackgroundModel:
    """Polynomial background model for a detector."""
    detector: str
    poly_order: int         # Polynomial order (0-4)
    coefficients: List[float] = field(default_factory=list)
    bic: float = 0.0       # Bayesian Information Criterion
    chi2_red: float = 0.0  # Reduced chi-squared
    pre_interval: Tuple[float, float] = (-100, -10)  # Pre-burst fit interval
    post_interval: Tuple[float, float] = (100, 300)   # Post-burst fit interval

    def evaluate(self, times):
        """Evaluate background model at given times."""
        try:
            import numpy as np
            return np.polyval(self.coefficients, times)
        except ImportError:
            # Fallback without numpy
            result = []
            for t in times:
                val = sum(c * t**(len(self.coefficients) - 1 - i)
                          for i, c in enumerate(self.coefficients))
                result.append(val)
            return result


@dataclass
class DetectorSelection:
    """Selected detectors for analysis."""
    nai_detectors: List[str] = field(default_factory=list)
    bgo_detectors: List[str] = field(default_factory=list)
    angles: Dict[str, float] = field(default_factory=dict)  # det -> angle
    method: str = 'manual'  # 'trigdat', 'approximate', 'manual'


@dataclass
class GBMAnalysisResult:
    """Complete result from GBM analysis."""
    grb_name: str = ''
    trigger_id: str = ''
    trigger_time: float = 0.0
    ra: float = 0.0
    dec: float = 0.0

    # Detector selection
    detectors: Optional[DetectorSelection] = None

    # Burst interval
    burst_interval: Optional[BurstInterval] = None
    all_intervals: List[BurstInterval] = field(default_factory=list)

    # Background
    background_models: Dict[str, BackgroundModel] = field(default_factory=dict)

    # Timing
    t90: Optional[float] = None
    t90_start: Optional[float] = None
    t50: Optional[float] = None
    t50_start: Optional[float] = None

    # Status
    status: str = 'pending'  # pending, running, completed, failed
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {}
        for key, val in self.__dict__.items():
            if val is None:
                d[key] = None
            elif isinstance(val, (str, int, float, bool)):
                d[key] = val
            elif isinstance(val, list):
                d[key] = [
                    item.__dict__ if hasattr(item, '__dict__') and not isinstance(item, str)
                    else item
                    for item in val
                ]
            elif isinstance(val, dict):
                d[key] = {
                    k: v.__dict__ if hasattr(v, '__dict__') and not isinstance(v, str) else v
                    for k, v in val.items()
                }
            elif hasattr(val, '__dict__'):
                d[key] = val.__dict__
            else:
                d[key] = str(val)
        return d

    def save(self, path: str):
        """Save result to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved analysis result to {path}")


# ------------------------------------------------------------------
# GBM detector configuration
# ------------------------------------------------------------------

NaI_DETECTORS = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                 'n6', 'n7', 'n8', 'n9', 'na', 'nb']
BGO_DETECTORS = ['b0', 'b1']
ALL_DETECTORS = NaI_DETECTORS + BGO_DETECTORS

# Energy ranges (keV)
NaI_ENERGY_RANGE = (8.0, 900.0)
BGO_ENERGY_RANGE = (200.0, 40000.0)

# Default energy channels for NaI light curves
DEFAULT_ENERGY_CHANNELS = {
    'soft':   (8, 50),     # keV
    'medium': (50, 300),   # keV
    'hard':   (300, 900),  # keV
    'total':  (8, 900),    # keV
}


class GBMAnalyzer:
    """
    Automated Fermi GBM burst analysis.

    Parameters
    ----------
    data_dir : str
        Directory containing GBM FITS files (TTE, CSPEC, trigdat)
    output_dir : str, optional
        Directory for output files. Defaults to data_dir/analysis/
    """

    def __init__(self, data_dir: str, output_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.result = GBMAnalysisResult()

        # Will be populated during analysis
        self._trigdat_data = None
        self._tte_data = {}     # det -> event data
        self._cspec_data = {}   # det -> spectral data
        self._lightcurves = {}  # det -> (time, counts)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, interactive: bool = True) -> GBMAnalysisResult:
        """
        Run the full automated GBM analysis pipeline.

        Parameters
        ----------
        interactive : bool
            If True, pause for user review after burst detection and
            background fitting.

        Returns
        -------
        GBMAnalysisResult
        """
        self.result.status = 'running'

        try:
            # Step 1: Discover data files
            self._discover_data()

            # Step 2: Read trigger info
            self._read_trigger_info()

            # Step 3: Select detectors
            self._select_detectors()

            # Step 4: Extract light curves
            self._extract_lightcurves()

            # Step 5: Detect burst interval
            self._detect_burst()

            # Step 6: Fit background
            self._fit_background()

            # Step 7: Calculate T90/T50
            self._calculate_durations()

            # Step 8: Interactive review
            if interactive:
                self._interactive_review()

            # Step 9: Generate spectral files
            self._generate_spectral_files()

            self.result.status = 'completed'

        except Exception as e:
            self.result.status = 'failed'
            self.result.notes.append(f"Analysis failed: {e}")
            logger.error(f"GBM analysis failed: {e}", exc_info=True)

        # Save result
        result_path = self.output_dir / 'gbm_analysis_result.json'
        self.result.save(str(result_path))

        return self.result

    # ------------------------------------------------------------------
    # Step 1: Data discovery
    # ------------------------------------------------------------------

    def _discover_data(self):
        """Find TTE, CSPEC, and trigdat files in data_dir."""
        logger.info(f"Discovering GBM data in {self.data_dir}")

        self._tte_files = sorted(self.data_dir.rglob('*_tte_*.fit*'))
        self._cspec_files = sorted(self.data_dir.rglob('*_cspec_*.fit*'))
        self._trigdat_files = sorted(self.data_dir.rglob('*trigdat*.fit*'))
        self._rsp_files = sorted(self.data_dir.rglob('*.rsp*'))

        logger.info(f"Found: {len(self._tte_files)} TTE, {len(self._cspec_files)} CSPEC, "
                     f"{len(self._trigdat_files)} trigdat, {len(self._rsp_files)} RSP files")

        if not self._tte_files and not self._cspec_files:
            raise FileNotFoundError(f"No TTE or CSPEC files found in {self.data_dir}")

    # ------------------------------------------------------------------
    # Step 2: Trigger info
    # ------------------------------------------------------------------

    def _read_trigger_info(self):
        """Extract trigger information from trigdat file."""
        if not self._trigdat_files:
            logger.warning("No trigdat file found, skipping trigger info extraction")
            return

        try:
            from astropy.io import fits
        except ImportError:
            logger.warning("astropy not available, skipping trigdat reading")
            return

        trigdat_path = self._trigdat_files[0]
        logger.info(f"Reading trigger info from {trigdat_path}")

        with fits.open(trigdat_path) as hdul:
            primary = hdul[0].header

            self.result.trigger_id = primary.get('TRIGTIME', '')
            self.result.trigger_time = float(primary.get('TRIGTIME', 0))

            # Try to get position from trigdat
            if 'TRIGRATE' in [ext.name for ext in hdul]:
                trigrate = hdul['TRIGRATE'].data
                if 'RA_OBJ' in trigrate.dtype.names:
                    self.result.ra = float(trigrate['RA_OBJ'][0])
                    self.result.dec = float(trigrate['DEC_OBJ'][0])

            # Extract trigger name from filename
            fname = trigdat_path.stem
            import re
            match = re.search(r'bn(\d{9})', fname)
            if match:
                self.result.trigger_id = match.group(1)
                self.result.grb_name = f"bn{match.group(1)}"

        logger.info(f"Trigger: {self.result.trigger_id}, "
                     f"RA={self.result.ra:.2f}, Dec={self.result.dec:.2f}")

    # ------------------------------------------------------------------
    # Step 3: Detector selection
    # ------------------------------------------------------------------

    def _select_detectors(self):
        """Auto-select the best NaI and BGO detectors."""
        from ..utils.gbm_geometry import select_gbm_detectors

        if self._trigdat_files and self.result.ra != 0:
            selection = select_gbm_detectors(
                ra=self.result.ra,
                dec=self.result.dec,
                trigdat_file=str(self._trigdat_files[0]),
            )
        else:
            # Fallback: use all detectors that have data files
            available = set()
            for f in self._tte_files + self._cspec_files:
                for det in ALL_DETECTORS:
                    if f'_{det}_' in f.name:
                        available.add(det)

            nai = sorted([d for d in available if d.startswith('n')])[:3]
            bgo = sorted([d for d in available if d.startswith('b')])[:1]

            selection = {
                'nai_detectors': [(d, 0.0) for d in nai],
                'bgo_detectors': [(d, 0.0) for d in bgo],
                'all_angles': [],
                'method': 'available_files',
            }

        self.result.detectors = DetectorSelection(
            nai_detectors=[d for d, a in selection['nai_detectors']],
            bgo_detectors=[d for d, a in selection['bgo_detectors']],
            angles={d: a for d, a in selection.get('all_angles', [])},
            method=selection['method'],
        )

        logger.info(f"Selected detectors: NaI={self.result.detectors.nai_detectors}, "
                     f"BGO={self.result.detectors.bgo_detectors}")

    # ------------------------------------------------------------------
    # Step 4: Light curve extraction
    # ------------------------------------------------------------------

    def _extract_lightcurves(self):
        """Extract light curves from TTE or CSPEC data for selected detectors."""
        try:
            import numpy as np
            from astropy.io import fits
        except ImportError:
            logger.error("numpy and astropy required for light curve extraction")
            return

        all_dets = (self.result.detectors.nai_detectors +
                     self.result.detectors.bgo_detectors)

        for det in all_dets:
            # Prefer TTE over CSPEC
            tte_file = self._find_file(self._tte_files, det)
            cspec_file = self._find_file(self._cspec_files, det)

            if tte_file:
                lc = self._extract_tte_lightcurve(tte_file, det)
            elif cspec_file:
                lc = self._extract_cspec_lightcurve(cspec_file, det)
            else:
                logger.warning(f"No data file found for detector {det}")
                continue

            if lc is not None:
                self._lightcurves[det] = lc

        logger.info(f"Extracted light curves for {len(self._lightcurves)} detectors")

    def _find_file(self, file_list: List[Path], detector: str) -> Optional[Path]:
        """Find the file for a specific detector."""
        for f in file_list:
            if f'_{detector}_' in f.name:
                return f
        return None

    def _extract_tte_lightcurve(
        self,
        fits_path: Path,
        detector: str,
        bin_size: float = 0.064,
        trange: Tuple[float, float] = (-50, 300),
    ) -> Optional[Dict[str, Any]]:
        """
        Extract binned light curve from TTE data.

        Parameters
        ----------
        fits_path : Path
            Path to TTE FITS file
        detector : str
            Detector name
        bin_size : float
            Time bin size in seconds
        trange : tuple
            Time range relative to trigger (tstart, tstop) in seconds

        Returns
        -------
        dict
            {'time': array, 'counts': array, 'rate': array, 'bin_size': float}
        """
        try:
            import numpy as np
            from astropy.io import fits
        except ImportError:
            return None

        with fits.open(fits_path) as hdul:
            events = hdul['EVENTS'].data
            times = events['TIME']  # MET seconds
            trigger_time = hdul['PRIMARY'].header.get('TRIGTIME', 0)

            # Relative to trigger
            rel_times = times - trigger_time

            # Filter to requested range
            mask = (rel_times >= trange[0]) & (rel_times <= trange[1])
            rel_times = rel_times[mask]

            if len(rel_times) == 0:
                return None

            # Bin the events
            edges = np.arange(trange[0], trange[1] + bin_size, bin_size)
            counts, _ = np.histogram(rel_times, bins=edges)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            rate = counts / bin_size

        return {
            'time': bin_centers,
            'counts': counts,
            'rate': rate,
            'rate_err': np.sqrt(counts) / bin_size,
            'bin_size': bin_size,
            'detector': detector,
            'data_type': 'tte',
        }

    def _extract_cspec_lightcurve(
        self,
        fits_path: Path,
        detector: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract light curve from CSPEC data."""
        try:
            import numpy as np
            from astropy.io import fits
        except ImportError:
            return None

        with fits.open(fits_path) as hdul:
            spectrum = hdul['SPECTRUM'].data
            times = spectrum['TIME']
            endtimes = spectrum['ENDTIME']
            counts_matrix = spectrum['COUNTS']  # (n_times, n_channels)

            trigger_time = hdul['PRIMARY'].header.get('TRIGTIME', 0)

            # Sum over all energy channels for total light curve
            total_counts = counts_matrix.sum(axis=1)
            dt = endtimes - times
            rate = total_counts / np.where(dt > 0, dt, 1)

            bin_centers = 0.5 * (times + endtimes) - trigger_time

        return {
            'time': bin_centers,
            'counts': total_counts,
            'rate': rate,
            'rate_err': np.sqrt(total_counts) / np.where(dt > 0, dt, 1),
            'bin_size': np.median(dt),
            'detector': detector,
            'data_type': 'cspec',
        }

    # ------------------------------------------------------------------
    # Step 5: Burst detection
    # ------------------------------------------------------------------

    def _detect_burst(self):
        """
        Detect burst interval (Tstart, Tstop) using multiple methods.

        Methods:
            1. bayesian_blocks: Astropy Bayesian blocks on event times
            2. snr_threshold: Signal-to-noise ratio above threshold
            3. cumulative: Change point in cumulative counts
            4. combined: Consensus of all methods
        """
        if not self._lightcurves:
            logger.warning("No light curves available for burst detection")
            return

        # Use the first NaI detector's light curve
        det = self.result.detectors.nai_detectors[0] if self.result.detectors else None
        if det not in self._lightcurves:
            det = list(self._lightcurves.keys())[0]

        lc = self._lightcurves[det]

        intervals = []

        # Method 1: S/N threshold
        try:
            interval = self._detect_snr(lc)
            if interval:
                intervals.append(interval)
        except Exception as e:
            logger.warning(f"S/N detection failed: {e}")

        # Method 2: Bayesian blocks
        try:
            interval = self._detect_bayesian_blocks(lc)
            if interval:
                intervals.append(interval)
        except Exception as e:
            logger.warning(f"Bayesian blocks detection failed: {e}")

        # Method 3: Cumulative counts
        try:
            interval = self._detect_cumulative(lc)
            if interval:
                intervals.append(interval)
        except Exception as e:
            logger.warning(f"Cumulative detection failed: {e}")

        self.result.all_intervals = intervals

        # Combined: take the consensus (median Tstart/Tstop)
        if intervals:
            import statistics
            tstarts = [iv.tstart for iv in intervals]
            tstops = [iv.tstop for iv in intervals]
            combined = BurstInterval(
                tstart=statistics.median(tstarts),
                tstop=statistics.median(tstops),
                method='combined',
                confidence=len(intervals) / 3.0,
            )
            self.result.burst_interval = combined
            self.result.all_intervals.append(combined)

            logger.info(f"Burst interval: [{combined.tstart:.3f}, {combined.tstop:.3f}] s "
                         f"(duration {combined.duration:.3f} s, confidence {combined.confidence:.1f})")

    def _detect_snr(self, lc: dict, threshold: float = 3.0) -> Optional[BurstInterval]:
        """
        Detect burst using signal-to-noise threshold crossing.

        Uses pre-burst background to estimate noise level, then finds
        the time range where count rate exceeds threshold * sigma.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        time = lc['time']
        rate = lc['rate']

        # Estimate background from pre-burst region (t < -10s)
        pre_mask = time < -10
        if np.sum(pre_mask) < 10:
            return None

        bg_mean = np.mean(rate[pre_mask])
        bg_std = np.std(rate[pre_mask])

        if bg_std <= 0:
            return None

        # Find where S/N exceeds threshold
        snr = (rate - bg_mean) / bg_std
        burst_mask = snr > threshold

        if not np.any(burst_mask):
            return None

        # Find first and last threshold crossing
        burst_indices = np.where(burst_mask)[0]
        tstart = time[burst_indices[0]]
        tstop = time[burst_indices[-1]]

        peak_snr = np.max(snr)

        return BurstInterval(
            tstart=tstart,
            tstop=tstop,
            method='snr_threshold',
            snr=peak_snr,
            confidence=min(peak_snr / 10.0, 1.0),
        )

    def _detect_bayesian_blocks(self, lc: dict) -> Optional[BurstInterval]:
        """
        Detect burst using Bayesian blocks algorithm.

        Identifies change points in the count rate using astropy's
        Bayesian blocks implementation.
        """
        try:
            import numpy as np
            from astropy.stats import bayesian_blocks
        except ImportError:
            logger.debug("astropy not available for Bayesian blocks")
            return None

        time = lc['time']
        counts = lc['counts']

        # Run Bayesian blocks
        try:
            edges = bayesian_blocks(time, counts, fitness='events', p0=0.01)
        except Exception as e:
            logger.warning(f"Bayesian blocks failed: {e}")
            return None

        if len(edges) < 3:
            return None

        # Find the block with the highest rate
        block_rates = []
        for i in range(len(edges) - 1):
            mask = (time >= edges[i]) & (time < edges[i + 1])
            if np.any(mask):
                block_rate = np.sum(counts[mask]) / (edges[i + 1] - edges[i])
                block_rates.append((i, block_rate, edges[i], edges[i + 1]))

        if not block_rates:
            return None

        # Sort by rate, find peak block
        block_rates.sort(key=lambda x: x[1], reverse=True)
        peak_idx, peak_rate, peak_start, peak_end = block_rates[0]

        # Background rate estimate from first and last blocks
        bg_rate = (block_rates[-1][1] + block_rates[-2][1]) / 2 if len(block_rates) > 1 else 0

        # Extend to include all blocks significantly above background
        sig_threshold = bg_rate + 2 * math.sqrt(bg_rate + 1)

        burst_blocks = [(s, e) for _, r, s, e in block_rates if r > sig_threshold]
        if burst_blocks:
            tstart = min(s for s, e in burst_blocks)
            tstop = max(e for s, e in burst_blocks)
        else:
            tstart = peak_start
            tstop = peak_end

        snr = (peak_rate - bg_rate) / max(math.sqrt(bg_rate), 1)

        return BurstInterval(
            tstart=tstart,
            tstop=tstop,
            method='bayesian_blocks',
            snr=snr,
            confidence=min(snr / 10.0, 1.0),
        )

    def _detect_cumulative(self, lc: dict) -> Optional[BurstInterval]:
        """
        Detect burst using cumulative counts method.

        Fits a line to the cumulative counts (background) and finds
        where the actual cumulative counts deviate significantly.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        time = lc['time']
        counts = lc['counts']

        cumsum = np.cumsum(counts)

        # Fit line to pre-burst cumulative counts
        pre_mask = time < -10
        if np.sum(pre_mask) < 10:
            return None

        # Linear fit to pre-burst region
        pre_time = time[pre_mask]
        pre_cum = cumsum[pre_mask]
        coeffs = np.polyfit(pre_time, pre_cum, 1)
        bg_line = np.polyval(coeffs, time)

        # Deviation from background
        deviation = cumsum - bg_line

        # Normalize
        if np.std(deviation[pre_mask]) > 0:
            norm_dev = deviation / np.std(deviation[pre_mask])
        else:
            return None

        # Find where deviation exceeds threshold
        threshold = 5.0
        burst_mask = norm_dev > threshold

        if not np.any(burst_mask):
            return None

        burst_indices = np.where(burst_mask)[0]
        tstart = time[burst_indices[0]]

        # Find where deviation stops growing (end of burst)
        # Look for where the slope of the deviation returns to ~0
        if len(burst_indices) > 10:
            slopes = np.diff(deviation[burst_indices])
            flat_mask = np.abs(slopes) < np.std(slopes) * 0.5
            if np.any(flat_mask):
                end_idx = burst_indices[np.where(flat_mask)[0][0]]
                tstop = time[end_idx]
            else:
                tstop = time[burst_indices[-1]]
        else:
            tstop = time[burst_indices[-1]]

        return BurstInterval(
            tstart=tstart,
            tstop=tstop,
            method='cumulative',
            snr=float(np.max(norm_dev)),
            confidence=0.7,
        )

    # ------------------------------------------------------------------
    # Step 6: Background fitting
    # ------------------------------------------------------------------

    def _fit_background(self):
        """
        Fit polynomial background model for each detector.

        Uses BIC (Bayesian Information Criterion) to select optimal
        polynomial order from 0 to 4. This is an improvement over
        gtburst which uses a fixed order of 2.
        """
        if not self.result.burst_interval:
            logger.warning("No burst interval set, cannot fit background")
            return

        try:
            import numpy as np
        except ImportError:
            logger.error("numpy required for background fitting")
            return

        burst = self.result.burst_interval
        buffer = max(5.0, burst.duration * 0.2)

        for det, lc in self._lightcurves.items():
            bg_model = self._fit_detector_background(
                lc,
                burst_start=burst.tstart - buffer,
                burst_stop=burst.tstop + buffer,
                max_order=4,
            )
            if bg_model:
                bg_model.detector = det
                self.result.background_models[det] = bg_model

        logger.info(f"Fitted background for {len(self.result.background_models)} detectors")

    def _fit_detector_background(
        self,
        lc: dict,
        burst_start: float,
        burst_stop: float,
        max_order: int = 4,
    ) -> Optional[BackgroundModel]:
        """
        Fit polynomial background excluding the burst interval.

        Selects the polynomial order that minimizes BIC.

        Parameters
        ----------
        lc : dict
            Light curve data
        burst_start, burst_stop : float
            Burst interval to exclude (seconds)
        max_order : int
            Maximum polynomial order to try

        Returns
        -------
        BackgroundModel or None
        """
        try:
            import numpy as np
        except ImportError:
            return None

        time = lc['time']
        rate = lc['rate']
        rate_err = lc.get('rate_err', np.sqrt(np.abs(rate)))

        # Select background intervals (exclude burst)
        bg_mask = (time < burst_start) | (time > burst_stop)
        bg_time = time[bg_mask]
        bg_rate = rate[bg_mask]
        bg_err = rate_err[bg_mask]

        if len(bg_time) < 10:
            return None

        # Weights for fitting (inverse variance)
        weights = 1.0 / np.where(bg_err > 0, bg_err**2, 1.0)

        # Try polynomial orders 0 through max_order, select by BIC
        best_bic = np.inf
        best_order = 0
        best_coeffs = [np.mean(bg_rate)]

        n = len(bg_time)

        for order in range(max_order + 1):
            k = order + 1  # Number of parameters

            try:
                coeffs = np.polyfit(bg_time, bg_rate, order, w=np.sqrt(weights))
                model = np.polyval(coeffs, bg_time)
                residuals = bg_rate - model
                chi2 = np.sum((residuals / np.where(bg_err > 0, bg_err, 1.0))**2)
                bic = chi2 + k * np.log(n)

                if bic < best_bic:
                    best_bic = bic
                    best_order = order
                    best_coeffs = coeffs.tolist()

            except Exception as e:
                logger.debug(f"Polynomial order {order} fit failed: {e}")
                continue

        chi2_red = best_bic / max(n - best_order - 1, 1)

        # Determine fit intervals used
        pre_times = bg_time[bg_time < burst_start]
        post_times = bg_time[bg_time > burst_stop]

        pre_interval = (float(pre_times[0]), float(pre_times[-1])) if len(pre_times) > 0 else (-100, -10)
        post_interval = (float(post_times[0]), float(post_times[-1])) if len(post_times) > 0 else (100, 300)

        logger.debug(f"Background fit: order={best_order}, BIC={best_bic:.1f}")

        return BackgroundModel(
            detector='',
            poly_order=best_order,
            coefficients=best_coeffs,
            bic=best_bic,
            chi2_red=chi2_red,
            pre_interval=pre_interval,
            post_interval=post_interval,
        )

    # ------------------------------------------------------------------
    # Step 7: T90/T50 calculation
    # ------------------------------------------------------------------

    def _calculate_durations(self):
        """
        Calculate T90 and T50 from background-subtracted light curve.

        T90: time interval containing 5% to 95% of total counts
        T50: time interval containing 25% to 75% of total counts
        """
        if not self.result.burst_interval or not self._lightcurves:
            return

        try:
            import numpy as np
        except ImportError:
            return

        # Use first NaI detector
        det = self.result.detectors.nai_detectors[0] if self.result.detectors else None
        if det not in self._lightcurves:
            det = list(self._lightcurves.keys())[0]

        lc = self._lightcurves[det]
        burst = self.result.burst_interval

        time = lc['time']
        counts = lc['counts']
        bin_size = lc['bin_size']

        # Subtract background
        if det in self.result.background_models:
            bg = self.result.background_models[det]
            bg_counts = np.array(bg.evaluate(time)) * bin_size
        else:
            # Simple background estimate
            pre_mask = time < burst.tstart - 5
            if np.sum(pre_mask) > 0:
                bg_rate = np.mean(counts[pre_mask])
            else:
                bg_rate = 0
            bg_counts = np.full_like(counts, bg_rate, dtype=float)

        net_counts = counts - bg_counts

        # Select burst region with some buffer
        buffer = max(5, burst.duration * 0.3)
        mask = (time >= burst.tstart - buffer) & (time <= burst.tstop + buffer)
        burst_time = time[mask]
        burst_counts = net_counts[mask]

        # Cumulative sum
        cumsum = np.cumsum(np.maximum(burst_counts, 0))

        if cumsum[-1] <= 0:
            logger.warning("No significant counts in burst region")
            return

        frac = cumsum / cumsum[-1]

        # T90: 5% to 95%
        t05_idx = np.searchsorted(frac, 0.05)
        t95_idx = np.searchsorted(frac, 0.95)
        if t05_idx < len(burst_time) and t95_idx < len(burst_time):
            self.result.t90 = burst_time[t95_idx] - burst_time[t05_idx]
            self.result.t90_start = burst_time[t05_idx]

        # T50: 25% to 75%
        t25_idx = np.searchsorted(frac, 0.25)
        t75_idx = np.searchsorted(frac, 0.75)
        if t25_idx < len(burst_time) and t75_idx < len(burst_time):
            self.result.t50 = burst_time[t75_idx] - burst_time[t25_idx]
            self.result.t50_start = burst_time[t25_idx]

        logger.info(f"T90 = {self.result.t90:.3f} s, T50 = {self.result.t50:.3f} s")

    # ------------------------------------------------------------------
    # Step 8: Interactive review
    # ------------------------------------------------------------------

    def _interactive_review(self):
        """
        Display analysis results and ask user for confirmation.

        Shows:
            - Light curve with detected burst interval
            - Background model
            - T90/T50 values
            - Asks user to accept, adjust, or redo
        """
        print("\n" + "=" * 60)
        print("GBM ANALYSIS REVIEW")
        print("=" * 60)
        print(f"GRB: {self.result.grb_name}")
        print(f"Trigger ID: {self.result.trigger_id}")

        if self.result.detectors:
            print(f"NaI detectors: {self.result.detectors.nai_detectors}")
            print(f"BGO detectors: {self.result.detectors.bgo_detectors}")

        if self.result.burst_interval:
            bi = self.result.burst_interval
            print(f"\nBurst interval: [{bi.tstart:.3f}, {bi.tstop:.3f}] s  "
                  f"(duration: {bi.duration:.3f} s)")
            print(f"Detection method: {bi.method}")

        if self.result.all_intervals:
            print("\nAll detection methods:")
            for iv in self.result.all_intervals:
                print(f"  {iv.method:20s}: [{iv.tstart:.3f}, {iv.tstop:.3f}] s  "
                      f"(S/N={iv.snr:.1f})")

        for det, bg in self.result.background_models.items():
            print(f"\nBackground ({det}): polynomial order {bg.poly_order}, BIC={bg.bic:.1f}")

        if self.result.t90 is not None:
            print(f"\nT90 = {self.result.t90:.3f} s (start: {self.result.t90_start:.3f} s)")
        if self.result.t50 is not None:
            print(f"T50 = {self.result.t50:.3f} s (start: {self.result.t50_start:.3f} s)")

        print("\n" + "-" * 60)

        try:
            response = input("Accept these results? [y/n/adjust] (y): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = 'y'

        if response == 'n':
            self.result.notes.append("User rejected automatic results")
            logger.info("User rejected results")
        elif response.startswith('adj'):
            self._adjust_parameters()
        else:
            self.result.notes.append("User accepted automatic results")
            logger.info("User accepted results")

    def _adjust_parameters(self):
        """Allow user to manually adjust burst interval."""
        try:
            tstart_str = input(f"Tstart [{self.result.burst_interval.tstart:.3f}]: ").strip()
            tstop_str = input(f"Tstop [{self.result.burst_interval.tstop:.3f}]: ").strip()

            if tstart_str:
                self.result.burst_interval.tstart = float(tstart_str)
            if tstop_str:
                self.result.burst_interval.tstop = float(tstop_str)

            self.result.burst_interval.method = 'user_adjusted'
            self.result.notes.append(f"User adjusted interval to "
                                     f"[{self.result.burst_interval.tstart}, "
                                     f"{self.result.burst_interval.tstop}]")

            # Re-fit background and re-calculate T90 with new interval
            self._fit_background()
            self._calculate_durations()

        except (ValueError, EOFError, KeyboardInterrupt):
            logger.warning("Invalid input, keeping original values")

    # ------------------------------------------------------------------
    # Step 9: Spectral fitting via 3ML
    # ------------------------------------------------------------------

    def _generate_spectral_files(self):
        """
        Set up spectral analysis using 3ML (threeML).

        3ML provides a unified framework for GBM spectral fitting with:
            - TimeSeriesBuilder for GBM TTE/CSPEC data
            - Built-in background fitting (polynomial)
            - Band, CPL, PL, and other spectral models
            - Bayesian and frequentist fitting (pyXSPEC as backend)
            - Multi-detector joint fitting

        Generates a 3ML fitting script and optionally runs it.
        """
        if not self.result.burst_interval:
            logger.warning("No burst interval, skipping spectral setup")
            return

        logger.info("Setting up spectral analysis via 3ML...")

        # Write a 3ML fitting script
        self._write_3ml_script()

        self.result.notes.append("3ML spectral fitting script generated")

    def _write_3ml_script(self):
        """Write a 3ML (threeML) spectral fitting script."""
        if not self.result.detectors:
            return

        script_path = self.output_dir / 'fit_spectrum_3ml.py'
        bi = self.result.burst_interval
        dets = self.result.detectors.nai_detectors + self.result.detectors.bgo_detectors

        lines = [
            '"""',
            f'3ML spectral fitting script for {self.result.grb_name}',
            f'Trigger: {self.result.trigger_id}',
            f'Burst interval: [{bi.tstart:.3f}, {bi.tstop:.3f}] s',
            f'Detectors: {dets}',
            '"""',
            '',
            'from threeML import *',
            'from threeML.utils.OGIP.response import OGIPResponse',
            '',
            '# --- Configuration ---',
            f'trigger_name = "{self.result.grb_name}"',
            f'tstart = {bi.tstart:.3f}',
            f'tstop = {bi.tstop:.3f}',
            f'nai_detectors = {self.result.detectors.nai_detectors}',
            f'bgo_detectors = {self.result.detectors.bgo_detectors}',
            '',
            '# --- Build time series for each detector ---',
            'plugins = []',
            '',
            'for det in nai_detectors + bgo_detectors:',
            '    # Find TTE file for this detector',
            f'    tte_file = f"glg_tte_{{det}}_{self.result.grb_name}_v00.fit"',
            f'    rsp_file = f"glg_cspec_{{det}}_{self.result.grb_name}_v00.rsp2"',
            '',
            '    ts = TimeSeriesBuilder.from_gbm_tte(',
            '        det,',
            '        tte_file=tte_file,',
            '        rsp_file=rsp_file,',
            '    )',
            '',
            '    # Set background intervals (before and after burst)',
        ]

        # Use background model intervals if available
        for det, bg in self.result.background_models.items():
            lines.append(f'    # {det}: pre={bg.pre_interval}, post={bg.post_interval}, order={bg.poly_order}')
            break  # Just show one as example

        lines.extend([
            f'    ts.set_background_interval("{bi.tstart - 50:.1f}-{bi.tstart - 5:.1f}",',
            f'                               "{bi.tstop + 10:.1f}-{bi.tstop + 100:.1f}")',
            '',
            '    # Set source interval',
            f'    ts.set_active_time_interval("{tstart:.3f}-{tstop:.3f}")',
            '',
            '    # Create plugin',
            '    plugin = ts.to_spectrumlike()',
            '',
            '    # Set energy range',
            '    if det.startswith("n"):',
            '        plugin.set_active_measurements("8-900")',
            '    else:  # BGO',
            '        plugin.set_active_measurements("250-30000")',
            '',
            '    plugins.append(plugin)',
            '',
            '# --- Define spectral models ---',
            '',
            '# Band function (GRB standard)',
            'band = Band()',
            'band.alpha.value = -1.0',
            'band.alpha.bounds = (-10.0, 2.0)',
            'band.beta.value = -2.3',
            'band.beta.bounds = (-10.0, -1.0)',
            'band.xp.value = 300.0',
            'band.xp.bounds = (10.0, 10000.0)',
            'band.K.value = 0.01',
            'band.K.bounds = (1e-10, 1e3)',
            '',
            '# Cutoff power law',
            'cpl = Cutoff_powerlaw()',
            'cpl.index.value = -1.0',
            'cpl.xc.value = 300.0',
            'cpl.K.value = 0.01',
            '',
            '# Simple power law',
            'pl = Powerlaw()',
            'pl.index.value = -1.5',
            'pl.K.value = 0.01',
            '',
            '# --- Fit with Band function ---',
            'ps = PointSource("GRB", 0, 0, spectral_shape=band)',
            'model_band = Model(ps)',
            'data = DataList(*plugins)',
            '',
            '# Maximum likelihood fit',
            'jl_band = JointLikelihood(model_band, data)',
            'best_fit_band, likelihood_band = jl_band.fit()',
            'print("\\n--- Band function fit ---")',
            'print(best_fit_band)',
            '',
            '# --- Fit with Cutoff power law ---',
            'ps_cpl = PointSource("GRB", 0, 0, spectral_shape=cpl)',
            'model_cpl = Model(ps_cpl)',
            'jl_cpl = JointLikelihood(model_cpl, data)',
            'best_fit_cpl, likelihood_cpl = jl_cpl.fit()',
            'print("\\n--- Cutoff power law fit ---")',
            'print(best_fit_cpl)',
            '',
            '# --- Fit with power law ---',
            'ps_pl = PointSource("GRB", 0, 0, spectral_shape=pl)',
            'model_pl = Model(ps_pl)',
            'jl_pl = JointLikelihood(model_pl, data)',
            'best_fit_pl, likelihood_pl = jl_pl.fit()',
            'print("\\n--- Power law fit ---")',
            'print(best_fit_pl)',
            '',
            '# --- Model comparison ---',
            'print("\\n--- Model comparison (log-likelihood) ---")',
            'print(f"Band:     {likelihood_band}")',
            'print(f"CPL:      {likelihood_cpl}")',
            'print(f"PL:       {likelihood_pl}")',
            '',
            '# --- Optional: Bayesian analysis ---',
            '# Uncomment to run Bayesian fitting with emcee',
            '# bs_band = BayesianAnalysis(model_band, data)',
            '# bs_band.set_sampler("emcee")',
            '# bs_band.sampler.setup(n_walkers=50, n_burn_in=500, n_iterations=1000)',
            '# samples_band = bs_band.sample()',
            '# bs_band.results.corner_plot()',
            '',
            '# --- Plot results ---',
            '# jl_band.results.plot_model()',
            '# display_spectrum_model_counts(jl_band)',
        ])

        script_path.write_text('\n'.join(lines))
        logger.info(f"3ML fitting script written to {script_path}")
