"""
Low-level download utilities for the GRB pipeline.

Provides HTTP downloading, file caching, and retry logic.
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DataDownloader:
    """Generic file downloader with caching and retry."""

    def __init__(self, cache_dir: str = './cache', timeout: int = 30, retries: int = 3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries

    def download_file(self, url: str, output_path: Optional[str] = None) -> Path:
        """
        Download a single file with retry logic.

        Parameters
        ----------
        url : str
            URL to download
        output_path : str, optional
            Where to save. If None, uses cache directory.

        Returns
        -------
        Path
            Path to downloaded file

        Raises
        ------
        RuntimeError
            If download fails after all retries
        """
        import requests

        if output_path:
            dest = Path(output_path)
        else:
            # Use URL hash as filename in cache
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = url.split('/')[-1] or url_hash
            dest = self.cache_dir / filename

        dest.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, self.retries + 1):
            try:
                logger.debug(f"Downloading {url} (attempt {attempt}/{self.retries})")
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                with open(dest, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Downloaded {dest} ({dest.stat().st_size} bytes)")
                return dest

            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed: {e}")
                if attempt == self.retries:
                    raise RuntimeError(f"Failed to download {url} after {self.retries} attempts") from e

        return dest  # Should not reach here

    def download_fits(self, url: str, output_dir: str) -> Path:
        """Download a FITS file."""
        filename = url.split('/')[-1]
        output_path = Path(output_dir) / filename
        return self.download_file(url, str(output_path))
