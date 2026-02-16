"""GCN circular parsing and flux extraction using Claude AI."""
import json
import logging
import time
from typing import Dict, List, Optional
import re

import pandas as pd

try:
    from anthropic import Anthropic, RateLimitError, APIError
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None
    RateLimitError = Exception
    APIError = Exception

from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class GCNIntelligence:
    """Intelligent parsing of GCN circulars using Claude AI."""

    def __init__(self, config: Dict):
        """Initialize GCN Intelligence module.

        Args:
            config: Configuration dictionary containing:
                - ai_api_key: Anthropic API key
                - ai_model: Model name (default: claude-3-5-sonnet-20241022)
                - max_retries: Maximum API retry attempts (default: 3)
                - retry_delay: Seconds to wait between retries (default: 1)
        """
        self.api_key = config.get("ai_api_key")
        if not self.api_key:
            raise ValueError("ai_api_key required in config")

        self.client = Anthropic(api_key=self.api_key)
        self.model = config.get("ai_model", "claude-3-5-sonnet-20241022")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)

        logger.info(f"Initialized GCNIntelligence with model: {self.model}")

    def parse_circular(self, circular_text: str) -> Dict:
        """Parse a single GCN circular and extract structured information.

        Args:
            circular_text: Raw text of GCN circular

        Returns:
            Dictionary with extracted fields (GRB name, coordinates, spectral params, etc.)
        """
        logger.info("Parsing GCN circular")

        response = self._call_claude(
            system_prompt=PromptTemplates.GCN_PARSE_PROMPT,
            user_message=f"Please parse the following GCN circular:\n\n{circular_text}",
        )

        parsed = self._parse_json_response(response)
        logger.info(
            f"Successfully parsed GCN circular: {parsed.get('grb_name', 'unknown')}"
        )
        return parsed

    def extract_flux_densities(self, circular_text: str) -> List[Dict]:
        """Extract all flux density measurements from GCN circular.

        Args:
            circular_text: Raw text of GCN circular

        Returns:
            List of flux measurement dictionaries with time, wavelength, flux, units, etc.
        """
        logger.info("Extracting flux densities from GCN circular")

        response = self._call_claude(
            system_prompt=PromptTemplates.GCN_FLUX_EXTRACTION_PROMPT,
            user_message=f"Please extract all flux measurements from this GCN circular:\n\n{circular_text}",
        )

        flux_list = self._parse_json_response(response)
        if not isinstance(flux_list, list):
            flux_list = [flux_list]

        logger.info(f"Extracted {len(flux_list)} flux measurements")
        return flux_list

    def batch_parse_circulars(self, circulars: List[str]) -> List[Dict]:
        """Process multiple GCN circulars and parse each.

        Args:
            circulars: List of GCN circular texts

        Returns:
            List of parsed GCN dictionaries
        """
        logger.info(f"Batch parsing {len(circulars)} GCN circulars")
        parsed_circulars = []

        for i, circular in enumerate(circulars):
            try:
                parsed = self.parse_circular(circular)
                parsed_circulars.append(parsed)
                logger.debug(f"Parsed circular {i+1}/{len(circulars)}")
            except Exception as e:
                logger.error(f"Failed to parse circular {i+1}: {e}")
                continue

        logger.info(f"Successfully parsed {len(parsed_circulars)}/{len(circulars)} circulars")
        return parsed_circulars

    def build_grb_summary(self, parsed_circulars: List[Dict]) -> Dict:
        """Merge information from multiple GCN circulars into single GRB summary.

        Args:
            parsed_circulars: List of parsed GCN dictionaries (may be from multiple circulars)

        Returns:
            Merged dictionary with comprehensive GRB information
        """
        if not parsed_circulars:
            logger.warning("No parsed circulars provided")
            return {}

        logger.info(f"Building GRB summary from {len(parsed_circulars)} parsed circulars")

        # Initialize summary with first circular
        summary = {
            "grb_name": parsed_circulars[0].get("grb_name"),
            "trigger_time": parsed_circulars[0].get("trigger_time"),
            "coordinates": parsed_circulars[0].get("coordinates"),
            "redshift": parsed_circulars[0].get("redshift"),
            "t90": parsed_circulars[0].get("t90"),
            "spectral_parameters": parsed_circulars[0].get("spectral_parameters"),
            "flux_measurements": [],
            "instruments": set(),
            "key_findings": [],
            "sources_count": len(parsed_circulars),
        }

        # Aggregate flux measurements and other data
        for circular in parsed_circulars:
            summary["flux_measurements"].extend(circular.get("flux_measurements", []))
            summary["instruments"].update(circular.get("instruments", []))
            summary["key_findings"].extend(circular.get("key_findings", []))

        # Convert instrument set to list
        summary["instruments"] = sorted(list(summary["instruments"]))

        # Remove duplicates from key findings
        summary["key_findings"] = list(set(summary["key_findings"]))

        logger.info(f"Built GRB summary with {len(summary['flux_measurements'])} total flux measurements")
        return summary

    def create_flux_table(
        self, grb_name: str, parsed_circulars: List[Dict]
    ) -> pd.DataFrame:
        """Build multi-wavelength flux density table from parsed GCN data.

        Args:
            grb_name: GRB designation
            parsed_circulars: List of parsed GCN dictionaries

        Returns:
            DataFrame with columns: time, energy/wavelength, flux, flux_err, units, instrument, reference
        """
        logger.info(f"Creating flux table for {grb_name}")

        # Collect all flux measurements
        all_measurements = []
        for circular in parsed_circulars:
            all_measurements.extend(circular.get("flux_measurements", []))

        if not all_measurements:
            logger.warning(f"No flux measurements found for {grb_name}")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(all_measurements)

        # Ensure key columns exist
        required_cols = [
            "time_seconds_since_trigger",
            "flux_value",
            "flux_error",
            "units",
            "instrument",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Sort by time
        if "time_seconds_since_trigger" in df.columns:
            df = df.sort_values("time_seconds_since_trigger")

        logger.info(f"Created flux table with {len(df)} measurements")
        return df.reset_index(drop=True)

    def _call_claude(
        self, system_prompt: str, user_message: str, max_tokens: int = 4096
    ) -> str:
        """Low-level API call to Claude with error handling, retries, and rate limiting.

        Args:
            system_prompt: System prompt to set behavior
            user_message: User message/query
            max_tokens: Maximum tokens in response

        Returns:
            Raw response text from Claude

        Raises:
            APIError: If API call fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling Claude API (attempt {attempt + 1}/{self.max_retries})")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

                logger.debug("Claude API call successful")
                return response.content[0].text

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded after all retries")
                    raise

            except APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API error: {e}. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error after {self.max_retries} retries: {e}")
                    raise

        raise APIError("Failed to call Claude API after all retries")

    def _parse_json_response(self, response: str) -> Dict:
        """Robustly extract and parse JSON from Claude response.

        Handles markdown code blocks, partial JSON, and other formatting variations.

        Args:
            response: Raw response text from Claude

        Returns:
            Parsed JSON as dictionary or list

        Raises:
            ValueError: If JSON cannot be extracted
        """
        logger.debug("Parsing JSON from response")

        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                json_text = json_match.group(1)
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")

        # Try finding JSON object or array in response
        # Find first '{' or '[' and try to parse from there
        for start_char in ["{", "["]:
            idx = response.find(start_char)
            if idx != -1:
                # Try progressively shorter substrings
                for end_idx in range(len(response), idx, -1):
                    try:
                        json_text = response[idx:end_idx]
                        result = json.loads(json_text)
                        logger.debug("Successfully parsed JSON from extracted substring")
                        return result
                    except json.JSONDecodeError:
                        continue

        # Last resort: try to fix common JSON issues
        try:
            # Handle trailing commas
            fixed_response = re.sub(r",(\s*[}\]])", r"\1", response)
            # Handle unquoted keys (naive approach)
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            pass

        logger.error(f"Could not parse JSON from response: {response[:500]}")
        raise ValueError(
            f"Failed to extract valid JSON from Claude response. "
            f"Response preview: {response[:200]}"
        )
