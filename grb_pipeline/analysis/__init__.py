"""
GRB analysis pipeline modules.

Modules:
    lightcurve: Light curve analysis with Bayesian blocks
    spectral: Spectral fitting and flux density conversion
    temporal: Cross-correlation and spectral lag
    afterglow: Afterglow decay modeling
    correlations: Amati, Yonetoku, Ghirlanda
    classification: T90-HR GMM classification
    gbm_analysis: Automated Fermi GBM analysis pipeline
"""

__all__ = [
    'lightcurve',
    'spectral',
    'temporal',
    'afterglow',
    'correlations',
    'classification',
    'gbm_analysis',
]
