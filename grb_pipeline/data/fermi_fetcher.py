"""
Fermi GBM and LAT data downloader.

Downloads GBM (TTE, CSPEC, trigdat, RSP) and LAT data from the HEASARC archive.
Uses GCN circulars as the primary method to resolve GRB names to Fermi trigger IDs,
with astroquery catalog as a fallback.

Key class:
    FermiFetcher: Main interface for downloading Fermi data.

Usage:
    fetcher = FermiFetcher(data_dir='./DATA')
    result = fetcher.download('GRB240825A')
    # -> resolves trigger via GCN -> bn240825667
    # -> wget recursive download from HEASARC
    # -> files in DATA/GRB240825A/GBM/

Based on Khushboo's original download script with improvements:
    - GCN-first trigger resolution (handles A/B suffixes correctly)
    - Selective download by data type (tte, cspec, trigdat, rsp)
    - Integration with pipeline database
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# HEASARC archive URL patterns
GBM_TRIGGER_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers"
LAT_DATA_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat"
GBM_DAILY_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily"


@dataclass
class DownloadResult:
    """Result of a data download operation."""
    grb_name: str
    trigger_id: Optional[str] = None
    bn_id: Optional[str] = None
    data_dir: Optional[str] = None
    files_downloaded: List[str] = field(default_factory=list)
    gbm_url: Optional[str] = None
    lat_url: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class FermiFetcher:
    """
    Download Fermi GBM and LAT data from HEASARC archives.

    Resolution chain for trigger IDs:
        1. GCN circulars (primary) — handles A/B suffixes correctly
        2. astroquery HEASARC catalog (fallback)
        3. User prompt (last resort)

    Usage:
        fetcher = FermiFetcher(data_dir='./DATA')
        result = fetcher.download('GRB240825A')
        result = fetcher.download('GRB240825A', data_types=['tte', 'trigdat'])
    """

    def __init__(self, data_dir: str = './DATA', timeout: int = 30):
        """
        Parameters
        ----------
        data_dir : str
            Base directory for downloaded data
        timeout : int
            HTTP timeout for metadata queries (seconds)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._gcn_parser = None

    @property
    def gcn_parser(self):
        """Lazy-load GCN parser."""
        if self._gcn_parser is None:
            from .gcn_parser import GCNParser
            cache_dir = self.data_dir / '.gcn_cache'
            self._gcn_parser = GCNParser(cache_dir=str(cache_dir), timeout=self.timeout)
        return self._gcn_parser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        grb_name: str,
        data_types: Optional[List[str]] = None,
        download_lat: bool = True,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Download Fermi data for a GRB.

        Parameters
        ----------
        grb_name : str
            GRB name, e.g., 'GRB240825A'
        data_types : list of str, optional
            Which GBM data types to download. Options: 'tte', 'cspec', 'ctime',
            'trigdat', 'rsp', 'rsp2'. If None, downloads everything.
        download_lat : bool
            Whether to also download LAT data (default: True)
        dry_run : bool
            If True, only resolve trigger ID and print URLs without downloading

        Returns
        -------
        DownloadResult
            Result with paths to downloaded files
        """
        result = DownloadResult(grb_name=grb_name)

        # Step 1: Resolve trigger ID
        trigger_id = self.resolve_trigger_id(grb_name)
        if not trigger_id:
            result.error = f"Could not resolve Fermi trigger ID for {grb_name}"
            logger.error(result.error)
            return result

        result.trigger_id = trigger_id
        result.bn_id = f"bn{trigger_id}"
        logger.info(f"Resolved {grb_name} -> {result.bn_id}")

        # Step 2: Build download URLs
        year = '20' + trigger_id[:2]
        gbm_url = f"{GBM_TRIGGER_URL}/{year}/{result.bn_id}/current/"
        lat_url = f"{LAT_DATA_URL}/triggers/{year}/{result.bn_id}/"

        result.gbm_url = gbm_url
        result.lat_url = lat_url

        if dry_run:
            logger.info(f"Dry run — GBM URL: {gbm_url}")
            logger.info(f"Dry run — LAT URL: {lat_url}")
            result.success = True
            return result

        # Step 3: Create output directory
        grb_dir = self.data_dir / grb_name.replace(' ', '')
        gbm_dir = grb_dir / 'GBM'
        gbm_dir.mkdir(parents=True, exist_ok=True)

        result.data_dir = str(grb_dir)

        # Step 4: Download GBM data
        logger.info(f"Downloading GBM data to {gbm_dir}")
        gbm_files = self._download_gbm(
            gbm_url, str(gbm_dir), result.bn_id, data_types
        )
        result.files_downloaded.extend(gbm_files)

        # Step 5: Download LAT data (optional)
        if download_lat:
            lat_dir = grb_dir / 'LAT'
            lat_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading LAT data to {lat_dir}")
            lat_files = self._download_lat(lat_url, str(lat_dir))
            result.files_downloaded.extend(lat_files)

        result.success = len(result.files_downloaded) > 0
        logger.info(f"Downloaded {len(result.files_downloaded)} files for {grb_name}")
        return result

    def resolve_trigger_id(self, grb_name: str) -> Optional[str]:
        """
        Resolve GRB name to Fermi GBM 9-digit trigger ID.

        Tries in order:
            1. GCN circulars (handles A/B suffixes)
            2. astroquery HEASARC catalog
            3. Returns None if both fail

        Parameters
        ----------
        grb_name : str
            GRB name, e.g., 'GRB240825A'

        Returns
        -------
        str or None
            9-digit trigger ID, e.g., '240825667'
        """
        # Method 1: GCN circulars
        logger.info(f"Resolving trigger ID for {grb_name} via GCN...")
        try:
            info = self.gcn_parser.get_grb_info(grb_name)
            if info.fermi_trigger_id:
                logger.info(f"GCN: {grb_name} -> trigger {info.fermi_trigger_id}")
                return info.fermi_trigger_id
        except Exception as e:
            logger.warning(f"GCN lookup failed: {e}")

        # Method 2: astroquery catalog
        logger.info(f"GCN lookup failed, trying astroquery catalog...")
        try:
            trigger_id = self._resolve_via_catalog(grb_name)
            if trigger_id:
                logger.info(f"Catalog: {grb_name} -> trigger {trigger_id}")
                return trigger_id
        except Exception as e:
            logger.warning(f"Catalog lookup failed: {e}")

        logger.error(f"Could not resolve Fermi trigger ID for {grb_name}")
        return None

    def find_data_files(
        self,
        grb_name: str,
        data_type: Optional[str] = None,
        detector: Optional[str] = None,
    ) -> List[Path]:
        """
        Find downloaded data files for a GRB.

        Parameters
        ----------
        grb_name : str
            GRB name
        data_type : str, optional
            Filter by type: 'tte', 'cspec', 'trigdat', 'rsp', etc.
        detector : str, optional
            Filter by detector: 'n0'-'n9', 'na', 'nb', 'b0', 'b1'

        Returns
        -------
        list of Path
            Matching file paths
        """
        grb_dir = self.data_dir / grb_name.replace(' ', '') / 'GBM'
        if not grb_dir.exists():
            return []

        pattern = '*.fits*'
        files = list(grb_dir.rglob(pattern))

        if data_type:
            files = [f for f in files if data_type in f.name]

        if detector:
            files = [f for f in files if f'_{detector}_' in f.name]

        return sorted(files)

    def get_trigger_info(self, grb_name: str) -> Dict[str, Any]:
        """
        Get trigger metadata without downloading data.

        Parameters
        ----------
        grb_name : str
            GRB name

        Returns
        -------
        dict
            Trigger info from GCN circulars
        """
        info = self.gcn_parser.get_grb_info(grb_name)
        return {
            'grb_name': info.grb_name,
            'fermi_trigger_id': info.fermi_trigger_id,
            'fermi_bn_id': info.fermi_bn_id,
            'swift_trigger_id': info.swift_trigger_id,
            'ra': info.ra,
            'dec': info.dec,
            'redshift': info.redshift,
            't90': info.t90,
            'classification': info.classification,
            'missions_detected': info.missions_detected,
            'num_circulars': info.num_circulars,
        }

    # ------------------------------------------------------------------
    # Private download methods
    # ------------------------------------------------------------------

    def _download_gbm(
        self,
        url: str,
        output_dir: str,
        bn_id: str,
        data_types: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Download GBM data using wget recursive download.

        Parameters
        ----------
        url : str
            HEASARC URL for the trigger
        output_dir : str
            Local directory to save files
        bn_id : str
            Fermi trigger name (e.g., 'bn240825667')
        data_types : list of str, optional
            Filter by data type. If None, download all.

        Returns
        -------
        list of str
            Paths to downloaded files
        """
        if data_types:
            # Download only requested types
            downloaded = []
            for dtype in data_types:
                accept = self._get_accept_pattern(dtype, bn_id)
                files = self._wget_download(url, output_dir, accept=accept)
                downloaded.extend(files)
            return downloaded
        else:
            # Download everything
            return self._wget_download(url, output_dir)

    def _download_lat(self, url: str, output_dir: str) -> List[str]:
        """Download LAT data using wget."""
        return self._wget_download(url, output_dir)

    def _wget_download(
        self,
        url: str,
        output_dir: str,
        accept: Optional[str] = None,
    ) -> List[str]:
        """
        Recursive wget download.

        Parameters
        ----------
        url : str
            URL to download from
        output_dir : str
            Local directory
        accept : str, optional
            Accept pattern for wget (e.g., '*_tte_*')

        Returns
        -------
        list of str
            Downloaded file paths
        """
        cmd = [
            'wget', '-r', '-l1', '-nH', '--no-parent',
            '--cut-dirs=6',
            '-P', output_dir,
            '-q',  # Quiet mode
            '--no-check-certificate',
        ]

        if accept:
            cmd.extend(['-A', accept])

        cmd.append(url)

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for large downloads
            )

            if result.returncode != 0 and result.returncode != 8:
                # wget returns 8 for "server error" on some missing files, not fatal
                logger.warning(f"wget returned {result.returncode}: {result.stderr[:200]}")

        except FileNotFoundError:
            logger.error("wget not found. Install wget or use pip install wget")
            return []
        except subprocess.TimeoutExpired:
            logger.error("Download timed out after 10 minutes")
            return []
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return []

        # List downloaded files
        output_path = Path(output_dir)
        files = [str(f) for f in output_path.rglob('*') if f.is_file()]
        return files

    def _get_accept_pattern(self, data_type: str, bn_id: str) -> str:
        """
        Get wget accept pattern for a data type.

        Parameters
        ----------
        data_type : str
            One of: 'tte', 'cspec', 'ctime', 'trigdat', 'rsp', 'rsp2'
        bn_id : str
            Fermi trigger name

        Returns
        -------
        str
            Wget accept pattern
        """
        patterns = {
            'tte': f'glg_tte_*_{bn_id}_*.fit*',
            'cspec': f'glg_cspec_*_{bn_id}_*.fit*',
            'ctime': f'glg_ctime_*_{bn_id}_*.fit*',
            'trigdat': f'glg_trigdat_all_{bn_id}_*.fit*',
            'rsp': f'glg_cspec_*_{bn_id}_*.rsp*',
            'rsp2': f'glg_cspec_*_{bn_id}_*.rsp2*',
        }
        return patterns.get(data_type, f'*{data_type}*{bn_id}*')

    # ------------------------------------------------------------------
    # Catalog fallback (from Khushboo's approach)
    # ------------------------------------------------------------------

    def _resolve_via_catalog(self, grb_name: str) -> Optional[str]:
        """
        Resolve trigger ID using astroquery HEASARC catalog.

        This is Khushboo's original approach: query the fermigbrst table
        for the GBM trigger name matching the GRB date.

        Parameters
        ----------
        grb_name : str
            GRB name

        Returns
        -------
        str or None
            9-digit trigger ID
        """
        try:
            from astroquery.heasarc import Heasarc
        except ImportError:
            logger.warning("astroquery not installed, catalog lookup unavailable")
            return None

        # Extract date portion from GRB name
        name = grb_name.replace(' ', '').replace('GRB', '')
        # Get just the YYMMDD part (strip any letter suffix)
        date_part = re.match(r'(\d{6})', name)
        if not date_part:
            return None

        search_name = f"GRB{date_part.group(1)}*"

        try:
            heasarc = Heasarc()
            table = heasarc.query_object(
                search_name,
                mission='fermigbrst',
                fields='TRIGGER_NAME,NAME',
            )

            if table is not None and len(table) > 0:
                trigger_name = str(table['TRIGGER_NAME'][0])
                # Extract 9-digit ID from trigger name like 'bn240825667'
                match = re.search(r'(\d{9})', trigger_name)
                if match:
                    return match.group(1)

        except Exception as e:
            logger.warning(f"astroquery catalog query failed: {e}")

        return None
