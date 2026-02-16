"""AI-powered analysis and interpretation of GRB data."""
import json
import logging
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


class AnalysisEngine:
    """AI-powered engine for interpreting GRB analysis results."""

    def __init__(self, config: Dict):
        """Initialize Analysis Engine.

        Args:
            config: Configuration dictionary containing:
                - ai_api_key: Anthropic API key
                - ai_model: Model name (default: claude-3-5-sonnet-20241022)
                - max_retries: Maximum API retry attempts (default: 3)
        """
        self.api_key = config.get("ai_api_key")
        if not self.api_key:
            raise ValueError("ai_api_key required in config")

        self.client = Anthropic(api_key=self.api_key)
        self.model = config.get("ai_model", "claude-3-5-sonnet-20241022")
        self.max_retries = config.get("max_retries", 3)

        logger.info(f"Initialized AnalysisEngine with model: {self.model}")

    def interpret_results(self, analysis_result: Dict) -> str:
        """Send complete analysis results to Claude for scientific interpretation.

        Args:
            analysis_result: Dictionary containing analysis results (spectral fits,
                           temporal properties, afterglow parameters, etc.)

        Returns:
            Narrative text with scientific interpretation
        """
        logger.info("Interpreting GRB analysis results")

        # Format analysis context
        context = self._format_analysis_context(analysis_result)

        response = self._call_claude(
            system_prompt=PromptTemplates.ANALYSIS_INTERPRETATION_PROMPT,
            user_message=f"Please interpret the following GRB analysis results:\n\n{context}",
        )

        logger.info("Successfully interpreted analysis results")
        return response

    def detect_anomalies(self, grb_data: Dict) -> List[Dict]:
        """Identify unusual and anomalous properties in GRB data.

        Args:
            grb_data: Dictionary with GRB parameters including spectral, temporal,
                     and afterglow properties

        Returns:
            List of anomaly dictionaries with:
            - property: name of the anomalous property
            - value: measured value
            - expected_range: typical range for GRB population
            - significance: confidence level in anomaly
            - explanation: physical interpretation
        """
        logger.info("Detecting anomalies in GRB data")

        # Format data for Claude
        data_str = json.dumps(grb_data, indent=2, default=str)

        response = self._call_claude(
            system_prompt=PromptTemplates.ANOMALY_DETECTION_PROMPT,
            user_message=f"Please identify anomalies in this GRB:\n\n{data_str}",
        )

        try:
            anomalies = self._parse_json_response(response)
            if not isinstance(anomalies, list):
                anomalies = [anomalies]
        except ValueError:
            logger.warning("Could not parse anomalies as JSON, returning text response")
            anomalies = [{"interpretation": response, "confidence": 0.5}]

        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    def compare_to_population(self, grb_data: Dict, catalog_stats: Dict) -> str:
        """Compare GRB to population statistics and identify where it falls.

        Args:
            grb_data: Dictionary with GRB parameters
            catalog_stats: Dictionary with population statistics

        Returns:
            Narrative text comparing GRB to population
        """
        logger.info("Comparing GRB to population statistics")

        comparison_context = (
            f"GRB Properties:\n{json.dumps(grb_data, indent=2, default=str)}\n\n"
            f"Population Statistics:\n{json.dumps(catalog_stats, indent=2, default=str)}"
        )

        response = self._call_claude(
            system_prompt=PromptTemplates.COMPARISON_PROMPT,
            user_message=f"Please compare this GRB to the known GRB population:\n\n{comparison_context}",
        )

        logger.info("Successfully compared GRB to population")
        return response

    def suggest_followup(self, grb_data: Dict) -> List[str]:
        """Generate AI-powered follow-up observation recommendations.

        Args:
            grb_data: Dictionary with GRB parameters and current analysis state

        Returns:
            List of follow-up observation recommendations (human-readable strings)
        """
        logger.info("Generating follow-up recommendations")

        data_str = json.dumps(grb_data, indent=2, default=str)

        response = self._call_claude(
            system_prompt=PromptTemplates.FOLLOWUP_RECOMMENDATION_PROMPT,
            user_message=f"What follow-up observations would be most valuable for this GRB?\n\n{data_str}",
        )

        # Parse recommendations - try JSON format first, then split by lines
        try:
            rec_data = self._parse_json_response(response)
            if isinstance(rec_data, dict):
                recommendations = rec_data.get("recommendations", [response])
            elif isinstance(rec_data, list):
                recommendations = rec_data
            else:
                recommendations = [response]
        except ValueError:
            # If not JSON, split response into bullet points or lines
            recommendations = [
                line.strip() for line in response.split("\n") if line.strip()
            ]

        logger.info(f"Generated {len(recommendations)} follow-up recommendations")
        return recommendations

    def answer_question(self, question: str, context: Dict) -> str:
        """Answer a specific question about the GRB with analysis context.

        Args:
            question: Scientific question about the GRB
            context: Dictionary with analysis results and GRB properties

        Returns:
            Answer to the question as narrative text
        """
        logger.info(f"Answering question: {question}")

        context_str = json.dumps(context, indent=2, default=str)

        user_message = f"""Given this GRB analysis context:
{context_str}

Please answer the following question with scientific rigor:
{question}"""

        response = self._call_claude(
            system_prompt="You are an expert gamma-ray burst astronomer. "
            "Answer scientific questions about GRBs using the provided analysis context.",
            user_message=user_message,
        )

        logger.info("Successfully answered question")
        return response

    def think_through_analysis(self, grb_data: Dict, question: str = None) -> str:
        """Extended thinking mode - have Claude reason step-by-step about the physics.

        Args:
            grb_data: Dictionary with GRB parameters
            question: Optional specific question to address

        Returns:
            Detailed reasoning and analysis
        """
        logger.info("Initiating extended thinking analysis")

        data_str = json.dumps(grb_data, indent=2, default=str)

        if question:
            prompt = f"""Analyze this GRB data and answer the following question:
{question}

GRB Data:
{data_str}

Please think through the physics step-by-step, considering:
1. Standard GRB models and their predictions
2. Observational constraints
3. Possible physical mechanisms
4. Uncertainties and systematic effects"""
        else:
            prompt = f"""Please provide a comprehensive physical analysis of this GRB:

GRB Data:
{data_str}

Consider:
1. The progenitor system
2. Explosion and acceleration mechanisms
3. Radiation processes
4. Afterglow evolution
5. Constraints on fundamental physics"""

        response = self._call_claude(
            system_prompt="You are a theoretical GRB physicist. Think carefully through "
            "all physical aspects of gamma-ray bursts. Consider multiple models and "
            "explain your reasoning in detail.",
            user_message=prompt,
            max_tokens=8192,
        )

        logger.info("Completed extended thinking analysis")
        return response

    def classify_with_reasoning(self, grb_data: Dict) -> Dict:
        """AI-powered GRB classification with detailed reasoning chain.

        Args:
            grb_data: Dictionary with GRB parameters

        Returns:
            Dictionary with:
            - classification: GRB type classification
            - confidence: confidence level (0-1)
            - reasoning: detailed explanation
            - evidence: supporting evidence for classification
            - alternatives: alternative classifications considered
        """
        logger.info("Classifying GRB with reasoning")

        data_str = json.dumps(grb_data, indent=2, default=str)

        prompt = f"""Classify this GRB and provide detailed reasoning for the classification.

GRB Data:
{data_str}

For your classification, provide:
1. Primary classification (e.g., long GRB, short GRB, embedded long, hybrid)
2. Confidence level (0-100%)
3. Key diagnostic properties that support this classification
4. Alternative classifications considered and why they are less likely
5. Any ambiguities or unusual aspects of the classification

Format your response as JSON:
{{
  "classification": "string",
  "confidence_percent": int,
  "key_diagnostics": ["string"],
  "reasoning": "string",
  "alternative_classifications": [
    {{"classification": "string", "likelihood": "string"}}
  ],
  "ambiguities": ["string"]
}}"""

        response = self._call_claude(
            system_prompt="You are an expert in GRB classification with detailed knowledge "
            "of GRB types, their characteristics, and diagnostic properties.",
            user_message=prompt,
        )

        try:
            result = self._parse_json_response(response)
        except ValueError:
            logger.warning("Could not parse classification JSON, returning text response")
            result = {
                "classification": "Unknown",
                "confidence_percent": 0,
                "reasoning": response,
            }

        logger.info(f"Classified GRB as: {result.get('classification', 'Unknown')}")
        return result

    def _format_analysis_context(self, data: Dict) -> str:
        """Format analysis results into clean text for Claude prompt.

        Args:
            data: Analysis results dictionary

        Returns:
            Formatted text representation
        """
        formatted = "GRB Analysis Results:\n\n"

        # Format spectral parameters
        if "spectral_parameters" in data:
            formatted += "Spectral Parameters:\n"
            spec = data["spectral_parameters"]
            for key, value in spec.items():
                formatted += f"  {key}: {value}\n"
            formatted += "\n"

        # Format temporal properties
        if "temporal_properties" in data:
            formatted += "Temporal Properties:\n"
            temp = data["temporal_properties"]
            for key, value in temp.items():
                formatted += f"  {key}: {value}\n"
            formatted += "\n"

        # Format afterglow properties
        if "afterglow_parameters" in data:
            formatted += "Afterglow Parameters:\n"
            aglow = data["afterglow_parameters"]
            for key, value in aglow.items():
                formatted += f"  {key}: {value}\n"
            formatted += "\n"

        # Add any additional data
        for key, value in data.items():
            if key not in [
                "spectral_parameters",
                "temporal_properties",
                "afterglow_parameters",
            ]:
                if isinstance(value, (dict, list)):
                    formatted += f"{key}:\n{json.dumps(value, indent=2, default=str)}\n\n"
                else:
                    formatted += f"{key}: {value}\n"

        return formatted

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

    def _parse_json_response(self, response: str) -> Dict:
        """Robustly extract and parse JSON from Claude response.

        Args:
            response: Raw response text from Claude

        Returns:
            Parsed JSON as dictionary or list

        Raises:
            ValueError: If JSON cannot be extracted
        """
        import re

        logger.debug("Parsing JSON from response")

        # Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in response
        for start_char in ["{", "["]:
            idx = response.find(start_char)
            if idx != -1:
                for end_idx in range(len(response), idx, -1):
                    try:
                        return json.loads(response[idx:end_idx])
                    except json.JSONDecodeError:
                        continue

        raise ValueError(f"Could not parse JSON from response: {response[:200]}")
