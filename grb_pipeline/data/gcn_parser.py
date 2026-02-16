"""
GCN Circular parser for GRB trigger ID resolution and metadata extraction.

The primary purpose is to resolve GRB names (e.g., GRB240825A) to mission-specific
trigger IDs by fetching and parsing GCN circulars from the NASA GCN archive.

Key functions:
    get_fermi_trigger(grb_name) -> str: Get the Fermi GBM trigger ID (e.g., 'bn240825667')
    get_grb_info(grb_name) -> dict: Get all extracted info (trigger IDs, position, T90, z, etc.)
    fetch_event_circulars(grb_name) -> str: Fetch raw circular text from GCN archive

The GCN archive URL pattern for an event:
    https://gcn.gsfc.nasa.gov/other/{date_code}.gcn3
    e.g., https://gcn.gsfc.nasa.gov/other/240825A.gcn3
"""

import logging
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedCircular:
    """A single parsed GCN circular."""
    number: int = 0
    subject: str = ''
    date: str = ''
    body: str = ''
    from_line: str = ''


@dataclass
class GRBInfo:
    """Aggregated GRB information extracted from GCN circulars."""
    grb_name: str = ''
    fermi_trigger_id: Optional[str] = None  # e.g., '240825667'
    fermi_bn_id: Optional[str] = None       # e.g., 'bn240825667'
    swift_trigger_id: Optional[str] = None  # e.g., '1245876'
    ra: Optional[float] = None
    dec: Optional[float] = None
    ra_str: Optional[str] = None            # Original string, e.g., '12:34:56.7'
    dec_str: Optional[str] = None           # Original string, e.g., '+45:23:12.3'
    redshift: Optional[float] = None
    t90: Optional[float] = None
    classification: Optional[str] = None    # 'short' or 'long'
    missions_detected: List[str] = field(default_factory=list)
    num_circulars: int = 0
    circulars: List[ParsedCircular] = field(default_factory=list)


class GCNParser:
    """
    Fetch and parse GCN circulars for GRB event information.

    Usage:
        parser = GCNParser()
        info = parser.get_grb_info('GRB240825A')
        print(info.fermi_bn_id)  # 'bn240825667'

        # Or just get the Fermi trigger ID:
        trigger = parser.get_fermi_trigger('GRB240825A')
    """

    GCN_EVENT_URL = "https://gcn.gsfc.nasa.gov/other/{event_code}.gcn3"
    GCN_CIRCULAR_URL = "https://gcn.gsfc.nasa.gov/gcn3/{number}.gcn3"

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        self.timeout = timeout
        self._cache: Dict[str, str] = {}  # event_code -> raw text

        if cache_dir:
            from pathlib import Path
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fermi_trigger(self, grb_name: str) -> Optional[str]:
        """
        Get the Fermi GBM trigger ID for a GRB.

        Parameters
        ----------
        grb_name : str
            GRB name, e.g., 'GRB240825A' or '240825A'

        Returns
        -------
        str or None
            Fermi trigger ID in 'bn{YYMMDD}{frac}' format, e.g., 'bn240825667'
            Returns None if not found.
        """
        info = self.get_grb_info(grb_name)
        return info.fermi_bn_id

    def get_grb_info(self, grb_name: str) -> GRBInfo:
        """
        Fetch GCN circulars for a GRB and extract all available information.

        Parameters
        ----------
        grb_name : str
            GRB name, e.g., 'GRB240825A' or '240825A'

        Returns
        -------
        GRBInfo
            Extracted information including trigger IDs, position, redshift, etc.
        """
        event_code = self._normalize_event_code(grb_name)
        grb_name_clean = 'GRB' + event_code if not event_code.startswith('GRB') else event_code

        info = GRBInfo(grb_name=grb_name_clean)

        # Fetch raw circular text
        raw_text = self.fetch_event_circulars(grb_name)
        if not raw_text:
            logger.warning(f"No GCN circulars found for {grb_name}")
            return info

        # Parse into individual circulars
        circulars = self.parse_circulars(raw_text)
        info.circulars = circulars
        info.num_circulars = len(circulars)

        # Extract information from all circulars
        full_text = raw_text  # Search the entire text

        info.fermi_trigger_id = self.extract_fermi_trigger(full_text)
        if info.fermi_trigger_id:
            info.fermi_bn_id = f"bn{info.fermi_trigger_id}"

        info.swift_trigger_id = self.extract_swift_trigger(full_text)

        pos = self.extract_position(full_text)
        if pos:
            info.ra = pos.get('ra')
            info.dec = pos.get('dec')
            info.ra_str = pos.get('ra_str')
            info.dec_str = pos.get('dec_str')

        info.redshift = self.extract_redshift(full_text)
        info.t90 = self.extract_t90(full_text)
        info.classification = self.extract_classification(full_text)
        info.missions_detected = self.extract_missions(full_text)

        logger.info(f"Parsed {info.num_circulars} circulars for {grb_name_clean}: "
                     f"Fermi={info.fermi_bn_id}, Swift={info.swift_trigger_id}, "
                     f"z={info.redshift}, T90={info.t90}")

        return info

    def fetch_event_circulars(self, grb_name: str) -> Optional[str]:
        """
        Fetch all GCN circulars for a GRB event from the NASA archive.

        Parameters
        ----------
        grb_name : str
            GRB name, e.g., 'GRB240825A'

        Returns
        -------
        str or None
            Raw text of all circulars, or None if fetch failed.
        """
        event_code = self._normalize_event_code(grb_name)

        # Check cache first
        if event_code in self._cache:
            logger.debug(f"Using cached circulars for {event_code}")
            return self._cache[event_code]

        # Check file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{event_code}.gcn3"
            if cache_file.exists():
                text = cache_file.read_text(errors='replace')
                self._cache[event_code] = text
                return text

        # Fetch from GCN archive
        url = self.GCN_EVENT_URL.format(event_code=event_code)
        logger.info(f"Fetching GCN circulars from {url}")

        try:
            import requests
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            text = response.text

            # Cache it
            self._cache[event_code] = text
            if self.cache_dir:
                cache_file = self.cache_dir / f"{event_code}.gcn3"
                cache_file.write_text(text)

            logger.info(f"Fetched {len(text)} bytes of GCN circulars for {event_code}")
            return text

        except Exception as e:
            logger.error(f"Failed to fetch GCN circulars for {event_code}: {e}")
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_circulars(self, raw_text: str) -> List[ParsedCircular]:
        """
        Split raw GCN archive text into individual circulars.

        The GCN event page contains multiple circulars separated by
        divider lines (typically rows of '=' or similar).

        Parameters
        ----------
        raw_text : str
            Raw text from GCN event page

        Returns
        -------
        list of ParsedCircular
        """
        circulars = []

        # Split on dividers — GCN uses lines of ////// or ====== or similar
        # Common pattern: lines of repeated characters
        sections = re.split(r'\n[/=]{20,}\n', raw_text)

        for section in sections:
            section = section.strip()
            if len(section) < 50:
                continue  # Skip tiny fragments

            circ = ParsedCircular()

            # Extract circular number
            num_match = re.search(r'NUMBER:\s*(\d+)', section)
            if num_match:
                circ.number = int(num_match.group(1))

            # Extract subject
            subj_match = re.search(r'SUBJECT:\s*(.+?)(?:\n|$)', section)
            if subj_match:
                circ.subject = subj_match.group(1).strip()

            # Extract date
            date_match = re.search(r'DATE:\s*(.+?)(?:\n|$)', section)
            if date_match:
                circ.date = date_match.group(1).strip()

            # Extract from
            from_match = re.search(r'FROM:\s*(.+?)(?:\n|$)', section)
            if from_match:
                circ.from_line = from_match.group(1).strip()

            circ.body = section
            circulars.append(circ)

        return circulars

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def extract_fermi_trigger(self, text: str) -> Optional[str]:
        """
        Extract Fermi GBM trigger ID from GCN text.

        The trigger ID is a 9-digit number (YYMMDD + fractional day), e.g., 240825667.
        Common patterns in GCN circulars:
            "trigger 240825667"
            "trigger 240825667 / 769494427"  (trigger / MET)
            "GBM triggered on GRB 240825A (trigger 240825667)"
            "bn240825667"

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        str or None
            9-digit trigger ID string, e.g., '240825667'
        """
        patterns = [
            # "trigger 240825667" or "trigger 240825667 / 769494427"
            r'trigger\s+(\d{9})',
            # "bn240825667" (the standard GBM naming)
            r'bn(\d{9})',
            # "Trigger No: 240825667"
            r'[Tt]rigger\s*(?:[Nn]o\.?|#|ID)\s*:?\s*(\d{9})',
            # "GBM trigger ID 240825667"
            r'GBM\s+trigger\s+(?:ID\s+)?(\d{9})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                trigger_id = match.group(1)
                logger.debug(f"Found Fermi trigger ID: {trigger_id}")
                return trigger_id

        return None

    def extract_swift_trigger(self, text: str) -> Optional[str]:
        """
        Extract Swift BAT trigger number from GCN text.

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        str or None
            Swift trigger number
        """
        patterns = [
            r'Swift(?:/BAT)?\s+trigger\s+(\d{5,7})',
            r'BAT\s+trigger\s+(\d{5,7})',
            r'trigger\s*=\s*(\d{5,7})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                trigger_id = match.group(1)
                logger.debug(f"Found Swift trigger ID: {trigger_id}")
                return trigger_id

        return None

    def extract_position(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract RA/Dec position from GCN text.

        Handles formats:
            RA = 123.456, Dec = -45.678 (decimal degrees)
            RA(J2000) = 08h 23m 12.3s, Dec(J2000) = +45d 12' 34.5"
            RA: 123.456 Dec: -45.678

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        dict or None
            {'ra': float, 'dec': float, 'ra_str': str, 'dec_str': str}
        """
        # Try decimal degrees first
        dec_pattern = r'(?:RA|R\.A\.)\s*(?:\(J2000\))?\s*[=:]\s*([\d.]+)\s*(?:deg|d)?\s*,?\s*(?:Dec|DEC)\s*(?:\(J2000\))?\s*[=:]\s*([-+]?[\d.]+)'
        match = re.search(dec_pattern, text, re.IGNORECASE)
        if match:
            try:
                ra = float(match.group(1))
                dec = float(match.group(2))
                return {'ra': ra, 'dec': dec,
                        'ra_str': match.group(1), 'dec_str': match.group(2)}
            except ValueError:
                pass

        # Try HMS/DMS format: 08h 23m 12.3s, +45d 12' 34.5"
        hms_pattern = (
            r'(?:RA|R\.A\.)\s*(?:\(J2000\))?\s*[=:]\s*'
            r'(\d{1,2})[h:]\s*(\d{1,2})[m:]\s*([\d.]+)s?\s*,?\s*'
            r'(?:Dec|DEC)\s*(?:\(J2000\))?\s*[=:]\s*'
            r'([-+]?\d{1,3})[d:]\s*(\d{1,2})[\'m:]\s*([\d.]+)[\"s]?'
        )
        match = re.search(hms_pattern, text, re.IGNORECASE)
        if match:
            try:
                h, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
                ra = (h + m / 60 + s / 3600) * 15  # hours to degrees

                d, dm, ds = float(match.group(4)), float(match.group(5)), float(match.group(6))
                sign = -1 if d < 0 or match.group(4).startswith('-') else 1
                dec = sign * (abs(d) + dm / 60 + ds / 3600)

                ra_str = f"{int(h):02d}h{int(m):02d}m{s:.1f}s"
                dec_str = f"{match.group(4)}d{int(dm):02d}'{ds:.1f}\""

                return {'ra': ra, 'dec': dec, 'ra_str': ra_str, 'dec_str': dec_str}
            except ValueError:
                pass

        return None

    def extract_redshift(self, text: str) -> Optional[float]:
        """
        Extract redshift from GCN text.

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        float or None
        """
        patterns = [
            # "z = 1.23" or "z=1.23"
            r'(?<![a-zA-Z])z\s*[=~]\s*([\d.]+)',
            # "redshift of 1.23" or "redshift z = 1.23"
            r'redshift\s+(?:of\s+)?(?:z\s*[=~]\s*)?([\d.]+)',
            # "a redshift of z=1.23"
            r'redshift\s+of\s+z\s*[=~]\s*([\d.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    z = float(match.group(1))
                    if 0 < z < 20:  # Reasonable range for GRBs
                        logger.debug(f"Found redshift: {z}")
                        return z
                except ValueError:
                    continue

        return None

    def extract_t90(self, text: str) -> Optional[float]:
        """
        Extract T90 duration from GCN text.

        Handles patterns like:
            "T90 = 23.4 s"
            "T90 ~ 23.4 +/- 1.2 s"
            "T90 (50-300 keV) is 23.4 +/- 1.2 s"
            "T90 duration is about 23.4 s"
            "T90) of about 19.2 s"

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        float or None
            T90 in seconds
        """
        patterns = [
            # T90 = 23.4 s or T90 ~ 23.4 s
            r'T90\s*[=~]\s*([\d.]+)\s*(?:\+/?-?\s*[\d.]+\s*)?(?:s|sec)',
            # T90 (50-300 keV) is 23.4 +/- 1.2 s
            r'T90\s*(?:\([^)]*\))?\s*(?:is|was|=)\s*(?:about\s+)?([\d.]+)\s*(?:\+/?-?\s*[\d.]+\s*)?(?:s|sec)',
            # T90) of about 19.2 s
            r'T90\)?\s+(?:of\s+)?(?:about\s+)?([\d.]+)\s*(?:\+/?-?\s*[\d.]+\s*)?(?:s|sec)',
            # "duration of about 23.4 s" or "duration is about 23.4 s"
            r'duration\s+(?:is\s+|of\s+)?(?:about\s+|approximately\s+)?([\d.]+)\s*(?:\+/?-?\s*[\d.]+\s*)?(?:s|sec)',
            # T90 in ms
            r'T90\s*[=~]\s*([\d.]+)\s*ms',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    t90 = float(match.group(1))
                    # Convert ms to s if needed
                    if 'ms' in pattern:
                        t90 /= 1000.0
                    logger.debug(f"Found T90: {t90} s")
                    return t90
                except ValueError:
                    continue

        return None

    def extract_classification(self, text: str) -> Optional[str]:
        """
        Extract GRB classification (short/long) from GCN text.

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        str or None
            'short' or 'long'
        """
        text_lower = text.lower()

        short_indicators = [
            r'short[\s-]*(?:duration\s+)?(?:grb|burst|gamma)',
            r'(?:grb|burst)\s+(?:is|was|appears)\s+(?:a\s+)?short',
            r'T90\s*[=~<]\s*[\d.]+\s*(?:s|sec)\s*.*short',
        ]
        long_indicators = [
            r'long[\s-]*(?:duration\s+)?(?:grb|burst|gamma)',
            r'(?:grb|burst)\s+(?:is|was|appears)\s+(?:a\s+)?long',
        ]

        for pattern in short_indicators:
            if re.search(pattern, text_lower):
                return 'short'

        for pattern in long_indicators:
            if re.search(pattern, text_lower):
                return 'long'

        return None

    def extract_missions(self, text: str) -> List[str]:
        """
        Identify which missions/instruments detected this GRB.

        Parameters
        ----------
        text : str
            GCN circular text

        Returns
        -------
        list of str
            Mission/instrument names
        """
        missions = []
        mission_patterns = {
            'Fermi GBM': r'(?:Fermi|GBM)\s+(?:GBM\s+)?(?:trigger|detect|observ)',
            'Fermi LAT': r'(?:Fermi\s+)?LAT\s+(?:detect|observ|trigger)',
            'Swift BAT': r'Swift(?:/BAT)?\s+(?:trigger|detect|observ)',
            'Swift XRT': r'(?:Swift\s+)?XRT\s+(?:detect|observ|began)',
            'Swift UVOT': r'(?:Swift\s+)?UVOT\s+(?:detect|observ|began)',
            'INTEGRAL': r'INTEGRAL\s+(?:detect|observ|trigger)',
            'MAXI': r'MAXI\s+(?:detect|observ|trigger)',
            'Konus-Wind': r'Konus[\s-]Wind',
            'CALET': r'CALET\s+(?:detect|observ|trigger)',
        }

        for mission, pattern in mission_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                missions.append(mission)

        return missions

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _normalize_event_code(self, grb_name: str) -> str:
        """
        Normalize GRB name to event code used in GCN URLs.

        'GRB240825A' -> '240825A'
        'GRB 240825A' -> '240825A'
        '240825A' -> '240825A'

        Parameters
        ----------
        grb_name : str
            GRB name in various formats

        Returns
        -------
        str
            Normalized event code
        """
        name = grb_name.strip().replace(' ', '')
        if name.upper().startswith('GRB'):
            name = name[3:]
        return name
