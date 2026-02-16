"""
Data acquisition package for GRB pipeline.

Modules:
    gcn_parser: Fetch and parse GCN circulars to extract trigger IDs, positions, etc.
    fermi_fetcher: Download Fermi GBM/LAT data from HEASARC archives.
    swift_fetcher: Download Swift BAT/XRT/UVOT data (placeholder).
    downloaders: Low-level download utilities.
    multimessenger: GW/neutrino event cross-matching (placeholder).
"""

__all__ = ['gcn_parser', 'fermi_fetcher', 'swift_fetcher', 'downloaders', 'multimessenger']
