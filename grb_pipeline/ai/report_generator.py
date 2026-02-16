"""AI-powered scientific report generation for GRB analysis."""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from anthropic import Anthropic, APIError
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None
    APIError = Exception

from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive scientific reports on GRB analysis."""

    def __init__(self, config: Dict):
        """Initialize Report Generator.

        Args:
            config: Configuration dictionary containing:
                - ai_api_key: Anthropic API key
                - ai_model: Model name (default: claude-3-5-sonnet-20241022)
                - output_dir: Directory to save reports (default: ./reports)
                - max_retries: Maximum API retry attempts (default: 3)
        """
        self.api_key = config.get("ai_api_key")
        if not self.api_key:
            raise ValueError("ai_api_key required in config")

        self.client = Anthropic(api_key=self.api_key)
        self.model = config.get("ai_model", "claude-3-5-sonnet-20241022")
        self.output_dir = Path(config.get("output_dir", "./reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = config.get("max_retries", 3)

        logger.info(f"Initialized ReportGenerator with model: {self.model}")
        logger.info(f"Report output directory: {self.output_dir}")

    def generate_full_report(
        self, grb_name: str, analysis_result: Dict, output_format: str = "markdown"
    ) -> str:
        """Generate comprehensive analysis report.

        Args:
            grb_name: GRB designation
            analysis_result: Dictionary with complete analysis results
            output_format: Output format ('markdown' or 'html')

        Returns:
            Path to saved report file
        """
        logger.info(f"Generating full report for {grb_name} (format: {output_format})")

        # Generate individual sections
        sections = {}
        sections["executive_summary"] = self.generate_executive_summary(analysis_result)
        sections["spectral"] = self.generate_spectral_section(
            analysis_result.get("spectral_fits", [])
        )
        sections["temporal"] = self.generate_temporal_section(
            analysis_result.get("temporal_properties", {})
        )
        sections["afterglow"] = self.generate_afterglow_section(
            analysis_result.get("afterglow_parameters", {})
        )
        sections["correlation"] = self.generate_correlation_section(
            analysis_result.get("correlation_data", {})
        )
        sections["multimessenger"] = self.generate_multimessenger_section(
            analysis_result.get("multimessenger_data", {})
        )

        # Assemble full report
        report_content = self._assemble_report(sections, grb_name, output_format)

        # Save report
        output_path = self._save_report(report_content, grb_name, output_format)

        logger.info(f"Report saved to {output_path}")
        return str(output_path)

    def generate_executive_summary(self, analysis_result: Dict) -> str:
        """Generate concise 1-paragraph executive summary.

        Args:
            analysis_result: Dictionary with analysis results

        Returns:
            Summary text (1-2 sentences)
        """
        logger.info("Generating executive summary")

        data_str = json.dumps(analysis_result, indent=2, default=str)

        prompt = f"""Given this GRB analysis, write a concise 1-2 sentence executive summary
suitable for the beginning of a scientific report. The summary should capture the key
scientific findings and the GRB's significance.

Analysis:
{data_str}"""

        response = self._call_claude(
            system_prompt="You are a scientific writer specializing in gamma-ray burst astronomy.",
            user_message=prompt,
            max_tokens=256,
        )

        logger.info("Executive summary generated")
        return response.strip()

    def generate_spectral_section(self, spectral_fits: List[Dict]) -> str:
        """Generate detailed spectral analysis narrative.

        Args:
            spectral_fits: List of spectral fit results

        Returns:
            Narrative text on spectral properties
        """
        logger.info("Generating spectral analysis section")

        if not spectral_fits:
            logger.warning("No spectral fits provided")
            return "No spectral analysis data available."

        data_str = json.dumps(spectral_fits, indent=2, default=str)

        prompt = f"""Write a detailed scientific section on spectral analysis based on these spectral fit results:

{data_str}

The section should include:
1. Description of spectral properties and evolution
2. Physical interpretation of spectral indices (alpha, beta)
3. Peak energy (Epeak) implications
4. Comparison to spectral models
5. Temporal evolution of spectral properties if available

Write in professional scientific prose suitable for an astronomy journal."""

        response = self._call_claude(
            system_prompt="You are an expert in GRB spectral analysis and relativistic astrophysics.",
            user_message=prompt,
            max_tokens=2048,
        )

        logger.info("Spectral section generated")
        return response.strip()

    def generate_temporal_section(self, temporal_data: Dict) -> str:
        """Generate temporal properties narrative.

        Args:
            temporal_data: Dictionary with temporal properties

        Returns:
            Narrative text on temporal behavior
        """
        logger.info("Generating temporal section")

        if not temporal_data:
            logger.warning("No temporal data provided")
            return "No temporal analysis data available."

        data_str = json.dumps(temporal_data, indent=2, default=str)

        prompt = f"""Write a scientific section describing the temporal properties of this GRB:

{data_str}

Include:
1. T90 duration and classification implications
2. Temporal variability features
3. Light curve structure and evolution
4. Comparison to GRB population statistics
5. Physical interpretation of temporal behavior

Write in professional scientific prose."""

        response = self._call_claude(
            system_prompt="You are an expert in GRB temporal analysis and light curve modeling.",
            user_message=prompt,
            max_tokens=2048,
        )

        logger.info("Temporal section generated")
        return response.strip()

    def generate_afterglow_section(self, afterglow_data: Dict) -> str:
        """Generate afterglow evolution narrative.

        Args:
            afterglow_data: Dictionary with afterglow properties

        Returns:
            Narrative text on afterglow behavior
        """
        logger.info("Generating afterglow section")

        if not afterglow_data:
            logger.warning("No afterglow data provided")
            return "No afterglow analysis data available."

        data_str = json.dumps(afterglow_data, indent=2, default=str)

        prompt = f"""Write a comprehensive section on afterglow evolution based on these properties:

{data_str}

Include:
1. Multi-wavelength afterglow light curves and evolution
2. Spectral energy distribution evolution
3. Temporal decay indices and break identification
4. Consistency with standard afterglow models (forward shock, reverse shock)
5. Jet break signatures if present
6. Host galaxy dust extinction and implications

Write as a professional scientific section."""

        response = self._call_claude(
            system_prompt="You are an expert in GRB afterglow physics and multi-wavelength observations.",
            user_message=prompt,
            max_tokens=2048,
        )

        logger.info("Afterglow section generated")
        return response.strip()

    def generate_correlation_section(self, correlation_data: Dict) -> str:
        """Generate section on known GRB correlations.

        Args:
            correlation_data: Dictionary with correlation analysis results

        Returns:
            Narrative text on correlations
        """
        logger.info("Generating correlation section")

        if not correlation_data:
            logger.warning("No correlation data provided")
            return "No correlation analysis data available."

        data_str = json.dumps(correlation_data, indent=2, default=str)

        prompt = f"""Write a section analyzing how this GRB fits known GRB correlations:

{data_str}

Discuss:
1. Amati correlation (Epeak - Eiso relation)
2. Yonetoku correlation (Lpeak - Epeak relation)
3. Ghirlanda correlation (Epeak - Egamma relation)
4. Other relevant correlations in the literature
5. Any departures from expected correlations
6. Implications for GRB physics

Write as a professional scientific analysis."""

        response = self._call_claude(
            system_prompt="You are an expert in GRB correlations and their physical origins.",
            user_message=prompt,
            max_tokens=1536,
        )

        logger.info("Correlation section generated")
        return response.strip()

    def generate_multimessenger_section(self, mm_data: Dict) -> str:
        """Generate multi-messenger context section.

        Args:
            mm_data: Dictionary with multi-messenger data (GW, neutrino, etc.)

        Returns:
            Narrative text on multi-messenger context
        """
        logger.info("Generating multi-messenger section")

        if not mm_data:
            logger.warning("No multi-messenger data provided")
            return "No multi-messenger observations available for this GRB."

        data_str = json.dumps(mm_data, indent=2, default=str)

        prompt = f"""Write a section on the multi-messenger context of this GRB:

{data_str}

Include:
1. Any associated gravitational wave events
2. Neutrino observations or limits
3. Host galaxy properties and redshift
4. Potential progenitor constraints from host properties
5. Implications for multi-messenger astronomy

Write as a professional scientific section."""

        response = self._call_claude(
            system_prompt="You are an expert in multi-messenger astronomy and GRB observations.",
            user_message=prompt,
            max_tokens=1536,
        )

        logger.info("Multi-messenger section generated")
        return response.strip()

    def generate_html_report(
        self, grb_name: str, analysis_result: Dict, plot_paths: Optional[List[str]] = None
    ) -> str:
        """Generate full HTML report with embedded plots.

        Args:
            grb_name: GRB designation
            analysis_result: Dictionary with analysis results
            plot_paths: List of paths to plot files to embed

        Returns:
            Path to saved HTML report
        """
        logger.info(f"Generating HTML report for {grb_name}")

        # Generate markdown report first
        markdown_path = self.generate_full_report(grb_name, analysis_result, "markdown")

        # Read markdown content
        with open(markdown_path, "r") as f:
            markdown_content = f.read()

        # Convert to HTML with embedded plots
        html_content = self._markdown_to_html(markdown_content, plot_paths or [])

        # Save HTML
        html_path = self._save_report(html_content, grb_name, "html")

        logger.info(f"HTML report saved to {html_path}")
        return str(html_path)

    def _assemble_report(self, sections: Dict, grb_name: str, format: str = "markdown") -> str:
        """Combine all sections into final document.

        Args:
            sections: Dictionary with section names and content
            grb_name: GRB designation
            format: Output format ('markdown' or 'html')

        Returns:
            Complete report text
        """
        logger.info(f"Assembling {format} report for {grb_name}")

        timestamp = datetime.now().isoformat()

        if format.lower() == "html":
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GRB {grb_name} Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
        }}
        .header {{
            background-color: #0066cc;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GRB {grb_name} Comprehensive Analysis Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>{sections.get('executive_summary', 'No summary available.')}</p>
    </div>

    <div class="section">
        <h2>Spectral Analysis</h2>
        <p>{sections.get('spectral', 'No spectral analysis available.')}</p>
    </div>

    <div class="section">
        <h2>Temporal Properties</h2>
        <p>{sections.get('temporal', 'No temporal analysis available.')}</p>
    </div>

    <div class="section">
        <h2>Afterglow Evolution</h2>
        <p>{sections.get('afterglow', 'No afterglow analysis available.')}</p>
    </div>

    <div class="section">
        <h2>Correlation Analysis</h2>
        <p>{sections.get('correlation', 'No correlation analysis available.')}</p>
    </div>

    <div class="section">
        <h2>Multi-Messenger Context</h2>
        <p>{sections.get('multimessenger', 'No multi-messenger data available.')}</p>
    </div>
</body>
</html>"""
            return html

        else:  # markdown format
            report = f"""# GRB {grb_name} Comprehensive Analysis Report

**Generated:** {timestamp}

---

## Executive Summary

{sections.get('executive_summary', 'No summary available.')}

---

## Spectral Analysis

{sections.get('spectral', 'No spectral analysis available.')}

---

## Temporal Properties

{sections.get('temporal', 'No temporal analysis available.')}

---

## Afterglow Evolution

{sections.get('afterglow', 'No afterglow analysis available.')}

---

## Correlation Analysis

{sections.get('correlation', 'No correlation analysis available.')}

---

## Multi-Messenger Context

{sections.get('multimessenger', 'No multi-messenger data available.')}

---

*This report was automatically generated by the GRB Analysis Pipeline with AI assistance.*
"""
            return report

    def _save_report(self, content: str, grb_name: str, format: str) -> Path:
        """Save report to file and return path.

        Args:
            content: Report content
            grb_name: GRB designation
            format: File format ('markdown' or 'html')

        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "html" if format.lower() == "html" else "md"
        filename = f"{grb_name}_report_{timestamp}.{extension}"
        filepath = self.output_dir / filename

        logger.info(f"Saving report to {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Report successfully saved to {filepath}")
        return filepath

    def _markdown_to_html(self, markdown_content: str, plot_paths: List[str]) -> str:
        """Convert markdown to HTML and embed plots.

        Args:
            markdown_content: Markdown text
            plot_paths: List of plot file paths

        Returns:
            HTML content
        """
        # Simple markdown to HTML conversion
        html = markdown_content.replace("\n\n", "</p><p>").replace("\n", "<br>")

        # Embed plot images
        for plot_path in plot_paths:
            if os.path.exists(plot_path):
                relative_path = os.path.relpath(plot_path)
                html += f'<img src="{relative_path}" alt="Plot">\n'

        return html

    def _call_claude(
        self, system_prompt: str, user_message: str, max_tokens: int = 4096
    ) -> str:
        """Low-level API call to Claude with error handling and retries.

        Args:
            system_prompt: System prompt to set behavior
            user_message: User message/query
            max_tokens: Maximum tokens in response

        Returns:
            Raw response text from Claude

        Raises:
            APIError: If API call fails
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

            except APIError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API error on attempt {attempt + 1}: {e}")
                    continue
                else:
                    logger.error(f"API error after {self.max_retries} attempts: {e}")
                    raise

        raise APIError("Failed to call Claude API after all retries")
