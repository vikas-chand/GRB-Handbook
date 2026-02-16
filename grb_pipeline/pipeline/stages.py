"""Pipeline stages for GRB analysis workflow."""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    def __init__(self, name: str, description: str, required_inputs: List[str], outputs: List[str]):
        """Initialize a pipeline stage.

        Args:
            name: Stage identifier
            description: Human-readable description
            required_inputs: List of required context keys
            outputs: List of keys that will be added to context
        """
        self.name = name
        self.description = description
        self.required_inputs = required_inputs
        self.outputs = outputs
        self.logger = logging.getLogger(f"grb_pipeline.stages.{name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this stage of the pipeline.

        Args:
            context: Pipeline context dict with input data

        Returns:
            Updated context dict with stage outputs

        Raises:
            ValueError: If validation fails
            RuntimeError: If execution fails
        """
        pass

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate that required inputs are present in context.

        Args:
            context: Pipeline context dict

        Returns:
            True if all required inputs are present

        Raises:
            ValueError: If required inputs are missing
        """
        missing = [inp for inp in self.required_inputs if inp not in context]
        if missing:
            raise ValueError(
                f"Stage '{self.name}' missing required inputs: {missing}. "
                f"Available inputs: {list(context.keys())}"
            )
        return True

    def __repr__(self) -> str:
        """String representation of stage."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"inputs={self.required_inputs}, "
            f"outputs={self.outputs})"
        )


class DataAcquisitionStage(PipelineStage):
    """Fetch raw GRB data from Swift, Fermi, and other archives."""

    def __init__(self):
        """Initialize data acquisition stage."""
        super().__init__(
            name="data_acquisition",
            description="Download GRB data from Swift, Fermi archives and parse GCN circulars",
            required_inputs=["grb_name", "data_manager", "gcn_fetcher"],
            outputs=[
                "raw_data_paths",
                "gcn_data",
                "multi_messenger_matches",
                "acquisition_metadata",
                "acquisition_errors",
            ],
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Download GRB data and GCN circulars.

        Args:
            context: Must contain grb_name, data_manager, gcn_fetcher

        Returns:
            Context with raw_data_paths, gcn_data, multi_messenger_matches
        """
        self.validate_inputs(context)
        grb_name = context["grb_name"]
        data_manager = context["data_manager"]
        gcn_fetcher = context["gcn_fetcher"]

        self.logger.info(f"Starting data acquisition for {grb_name}")
        acquisition_errors = []

        try:
            # Acquire Swift data
            self.logger.info(f"Fetching Swift data for {grb_name}")
            swift_data = {}
            try:
                swift_data = data_manager.fetch_swift_data(grb_name)
                self.logger.debug(f"Swift data acquired: {len(swift_data)} files")
            except Exception as e:
                msg = f"Failed to fetch Swift data: {str(e)}"
                self.logger.warning(msg)
                acquisition_errors.append(("swift", str(e)))

            # Acquire Fermi data
            self.logger.info(f"Fetching Fermi data for {grb_name}")
            fermi_data = {}
            try:
                fermi_data = data_manager.fetch_fermi_data(grb_name)
                self.logger.debug(f"Fermi data acquired: {len(fermi_data)} files")
            except Exception as e:
                msg = f"Failed to fetch Fermi data: {str(e)}"
                self.logger.warning(msg)
                acquisition_errors.append(("fermi", str(e)))

            # Fetch GCN circulars
            self.logger.info(f"Fetching GCN circulars for {grb_name}")
            gcn_data = {}
            try:
                gcn_data = gcn_fetcher.fetch_grb_circulars(grb_name)
                self.logger.debug(f"GCN circulars acquired: {len(gcn_data)} circulars")
            except Exception as e:
                msg = f"Failed to fetch GCN circulars: {str(e)}"
                self.logger.warning(msg)
                acquisition_errors.append(("gcn", str(e)))

            # Check multi-messenger associations
            self.logger.info(f"Checking multi-messenger associations for {grb_name}")
            multi_messenger_matches = {}
            try:
                if gcn_data:
                    multi_messenger_matches = data_manager.check_multi_messenger(
                        grb_name, gcn_data
                    )
                    self.logger.info(
                        f"Found {len(multi_messenger_matches)} multi-messenger associations"
                    )
            except Exception as e:
                msg = f"Failed to check multi-messenger associations: {str(e)}"
                self.logger.warning(msg)
                acquisition_errors.append(("multi_messenger", str(e)))

            # Compile raw data paths
            raw_data_paths = {
                "swift": swift_data,
                "fermi": fermi_data,
                "gcn_circulars": gcn_data,
            }

            # Store results in database
            self.logger.info(f"Storing acquisition results in database for {grb_name}")
            try:
                if hasattr(data_manager, "store_raw_data"):
                    data_manager.store_raw_data(grb_name, raw_data_paths)
                if hasattr(data_manager, "store_gcn_data"):
                    data_manager.store_gcn_data(grb_name, gcn_data)
            except Exception as e:
                msg = f"Failed to store results in database: {str(e)}"
                self.logger.warning(msg)
                acquisition_errors.append(("storage", str(e)))

            # Update context with results
            context["raw_data_paths"] = raw_data_paths
            context["gcn_data"] = gcn_data
            context["multi_messenger_matches"] = multi_messenger_matches
            context["acquisition_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "swift_files": len(swift_data),
                "fermi_files": len(fermi_data),
                "gcn_circulars": len(gcn_data),
                "mm_associations": len(multi_messenger_matches),
            }
            context["acquisition_errors"] = acquisition_errors

            self.logger.info(f"Data acquisition completed for {grb_name}")
            return context

        except Exception as e:
            self.logger.error(f"Data acquisition failed for {grb_name}: {str(e)}")
            context["acquisition_errors"] = acquisition_errors + [("general", str(e))]
            context["raw_data_paths"] = {}
            context["gcn_data"] = {}
            context["multi_messenger_matches"] = {}
            context["acquisition_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": str(e),
            }
            return context


class TemporalAnalysisStage(PipelineStage):
    """Extract and analyze temporal properties of GRB emission."""

    def __init__(self):
        """Initialize temporal analysis stage."""
        super().__init__(
            name="temporal_analysis",
            description="Extract light curves and compute temporal properties (T90, T50, variability, pulses)",
            required_inputs=["grb_name", "temporal_analyzer", "raw_data_paths"],
            outputs=[
                "light_curves",
                "temporal_properties",
                "pulse_detection",
                "hardness_ratios",
                "temporal_errors",
            ],
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract light curves and calculate temporal properties.

        Args:
            context: Must contain grb_name, temporal_analyzer, raw_data_paths

        Returns:
            Context with light_curves, temporal_properties, pulse_detection
        """
        self.validate_inputs(context)
        grb_name = context["grb_name"]
        temporal_analyzer = context["temporal_analyzer"]
        raw_data_paths = context["raw_data_paths"]

        self.logger.info(f"Starting temporal analysis for {grb_name}")
        temporal_errors = []
        light_curves = {}
        temporal_properties = {}
        pulse_detection = {}
        hardness_ratios = {}

        try:
            # Extract light curves from available data
            self.logger.info(f"Extracting light curves for {grb_name}")
            try:
                if raw_data_paths.get("swift"):
                    light_curves["swift"] = temporal_analyzer.extract_light_curve(
                        grb_name, raw_data_paths["swift"], instrument="swift"
                    )
                    self.logger.debug(f"Swift light curve extracted")

                if raw_data_paths.get("fermi"):
                    light_curves["fermi"] = temporal_analyzer.extract_light_curve(
                        grb_name, raw_data_paths["fermi"], instrument="fermi"
                    )
                    self.logger.debug(f"Fermi light curve extracted")
            except Exception as e:
                msg = f"Failed to extract light curves: {str(e)}"
                self.logger.warning(msg)
                temporal_errors.append(("light_curve_extraction", str(e)))

            # Calculate temporal properties (T90, T50, variability)
            self.logger.info(f"Calculating temporal properties for {grb_name}")
            try:
                if light_curves:
                    temporal_properties = temporal_analyzer.calculate_temporal_properties(
                        light_curves
                    )
                    self.logger.debug(
                        f"Temporal properties: T90={temporal_properties.get('t90', 'N/A')}s"
                    )
            except Exception as e:
                msg = f"Failed to calculate temporal properties: {str(e)}"
                self.logger.warning(msg)
                temporal_errors.append(("temporal_properties", str(e)))

            # Detect pulses in light curve
            self.logger.info(f"Detecting pulses for {grb_name}")
            try:
                if light_curves:
                    pulse_detection = temporal_analyzer.detect_pulses(light_curves)
                    self.logger.info(f"Detected {len(pulse_detection)} pulses")
            except Exception as e:
                msg = f"Failed to detect pulses: {str(e)}"
                self.logger.warning(msg)
                temporal_errors.append(("pulse_detection", str(e)))

            # Compute hardness ratios
            self.logger.info(f"Computing hardness ratios for {grb_name}")
            try:
                if light_curves:
                    hardness_ratios = temporal_analyzer.compute_hardness_ratios(
                        light_curves
                    )
                    self.logger.debug(f"Hardness ratios computed: {len(hardness_ratios)} bands")
            except Exception as e:
                msg = f"Failed to compute hardness ratios: {str(e)}"
                self.logger.warning(msg)
                temporal_errors.append(("hardness_ratios", str(e)))

            # Update context with results
            context["light_curves"] = light_curves
            context["temporal_properties"] = temporal_properties
            context["pulse_detection"] = pulse_detection
            context["hardness_ratios"] = hardness_ratios
            context["temporal_errors"] = temporal_errors

            self.logger.info(f"Temporal analysis completed for {grb_name}")
            return context

        except Exception as e:
            self.logger.error(f"Temporal analysis failed for {grb_name}: {str(e)}")
            context["temporal_errors"] = temporal_errors + [("general", str(e))]
            context["light_curves"] = {}
            context["temporal_properties"] = {}
            context["pulse_detection"] = {}
            context["hardness_ratios"] = {}
            return context


class SpectralAnalysisStage(PipelineStage):
    """Fit spectral models and compute spectral properties."""

    def __init__(self):
        """Initialize spectral analysis stage."""
        super().__init__(
            name="spectral_analysis",
            description="Fit spectral models (Band, CPL, PL) and compute flux/luminosity",
            required_inputs=["grb_name", "spectral_analyzer", "raw_data_paths"],
            outputs=[
                "spectral_fits",
                "spectral_parameters",
                "energy_fluxes",
                "derived_quantities",
                "model_comparison",
                "spectral_errors",
            ],
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fit spectral models to available data.

        Args:
            context: Must contain grb_name, spectral_analyzer, raw_data_paths

        Returns:
            Context with spectral_fits, spectral_parameters, derived_quantities
        """
        self.validate_inputs(context)
        grb_name = context["grb_name"]
        spectral_analyzer = context["spectral_analyzer"]
        raw_data_paths = context["raw_data_paths"]
        redshift = context.get("redshift")

        self.logger.info(f"Starting spectral analysis for {grb_name}")
        spectral_errors = []
        spectral_fits = {}
        spectral_parameters = {}
        energy_fluxes = {}
        derived_quantities = {}
        model_comparison = {}

        try:
            # Fit spectral models
            self.logger.info(f"Fitting spectral models for {grb_name}")
            try:
                spectral_fits = spectral_analyzer.fit_spectral_models(
                    grb_name, raw_data_paths, models=["band", "cpl", "pl"]
                )
                self.logger.info(f"Fitted {len(spectral_fits)} spectral models")
            except Exception as e:
                msg = f"Failed to fit spectral models: {str(e)}"
                self.logger.warning(msg)
                spectral_errors.append(("model_fitting", str(e)))

            # Extract spectral parameters
            self.logger.info(f"Extracting spectral parameters for {grb_name}")
            try:
                if spectral_fits:
                    spectral_parameters = spectral_analyzer.extract_parameters(
                        spectral_fits
                    )
                    self.logger.debug(
                        f"Extracted parameters: {len(spectral_parameters)} parameters"
                    )
            except Exception as e:
                msg = f"Failed to extract spectral parameters: {str(e)}"
                self.logger.warning(msg)
                spectral_errors.append(("parameter_extraction", str(e)))

            # Calculate energy fluxes
            self.logger.info(f"Calculating energy fluxes for {grb_name}")
            try:
                if spectral_fits:
                    energy_fluxes = spectral_analyzer.calculate_energy_fluxes(
                        spectral_fits, energy_ranges=[(15, 150), (50, 300), (100, 1000)]
                    )
                    self.logger.debug(f"Energy fluxes calculated: {len(energy_fluxes)} ranges")
            except Exception as e:
                msg = f"Failed to calculate energy fluxes: {str(e)}"
                self.logger.warning(msg)
                spectral_errors.append(("flux_calculation", str(e)))

            # Compute derived quantities (Epeak, Eiso, Lpeak)
            self.logger.info(f"Computing derived quantities for {grb_name}")
            try:
                if spectral_parameters and redshift:
                    derived_quantities = spectral_analyzer.compute_derived_quantities(
                        spectral_parameters, redshift
                    )
                    self.logger.info(
                        f"Derived quantities: "
                        f"Epeak={derived_quantities.get('epeak', 'N/A')}, "
                        f"Eiso={derived_quantities.get('eiso', 'N/A')}"
                    )
                elif spectral_parameters:
                    derived_quantities = spectral_analyzer.compute_derived_quantities(
                        spectral_parameters
                    )
                    self.logger.info("Derived quantities computed (no luminosity without redshift)")
            except Exception as e:
                msg = f"Failed to compute derived quantities: {str(e)}"
                self.logger.warning(msg)
                spectral_errors.append(("derived_quantities", str(e)))

            # Compare models
            self.logger.info(f"Comparing spectral models for {grb_name}")
            try:
                if spectral_fits:
                    model_comparison = spectral_analyzer.compare_models(spectral_fits)
                    best_model = model_comparison.get("best_model")
                    self.logger.info(f"Best-fit model: {best_model}")
            except Exception as e:
                msg = f"Failed to compare models: {str(e)}"
                self.logger.warning(msg)
                spectral_errors.append(("model_comparison", str(e)))

            # Update context with results
            context["spectral_fits"] = spectral_fits
            context["spectral_parameters"] = spectral_parameters
            context["energy_fluxes"] = energy_fluxes
            context["derived_quantities"] = derived_quantities
            context["model_comparison"] = model_comparison
            context["spectral_errors"] = spectral_errors

            self.logger.info(f"Spectral analysis completed for {grb_name}")
            return context

        except Exception as e:
            self.logger.error(f"Spectral analysis failed for {grb_name}: {str(e)}")
            context["spectral_errors"] = spectral_errors + [("general", str(e))]
            context["spectral_fits"] = {}
            context["spectral_parameters"] = {}
            context["energy_fluxes"] = {}
            context["derived_quantities"] = {}
            context["model_comparison"] = {}
            return context


class AfterglowAnalysisStage(PipelineStage):
    """Analyze the afterglow phase of the GRB."""

    def __init__(self):
        """Initialize afterglow analysis stage."""
        super().__init__(
            name="afterglow_analysis",
            description="Fit afterglow decay, detect plateau/jet break, test closure relations",
            required_inputs=["grb_name", "afterglow_analyzer", "raw_data_paths"],
            outputs=[
                "afterglow_light_curves",
                "decay_models",
                "afterglow_parameters",
                "closure_relations",
                "jet_break_detection",
                "afterglow_errors",
            ],
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze GRB afterglow properties.

        Args:
            context: Must contain grb_name, afterglow_analyzer, raw_data_paths

        Returns:
            Context with afterglow_light_curves, decay_models, afterglow_parameters
        """
        self.validate_inputs(context)
        grb_name = context["grb_name"]
        afterglow_analyzer = context["afterglow_analyzer"]
        raw_data_paths = context["raw_data_paths"]

        self.logger.info(f"Starting afterglow analysis for {grb_name}")
        afterglow_errors = []
        afterglow_light_curves = {}
        decay_models = {}
        afterglow_parameters = {}
        closure_relations = {}
        jet_break_detection = {}

        try:
            # Extract afterglow light curves
            self.logger.info(f"Extracting afterglow light curves for {grb_name}")
            try:
                afterglow_light_curves = afterglow_analyzer.extract_afterglow_lcs(
                    grb_name, raw_data_paths
                )
                self.logger.info(
                    f"Extracted afterglow light curves: {len(afterglow_light_curves)} bands"
                )
            except Exception as e:
                msg = f"Failed to extract afterglow light curves: {str(e)}"
                self.logger.warning(msg)
                afterglow_errors.append(("lc_extraction", str(e)))

            # Fit afterglow decay models
            self.logger.info(f"Fitting afterglow decay models for {grb_name}")
            try:
                if afterglow_light_curves:
                    decay_models = afterglow_analyzer.fit_decay_models(
                        afterglow_light_curves
                    )
                    self.logger.info(f"Fitted {len(decay_models)} decay models")
            except Exception as e:
                msg = f"Failed to fit decay models: {str(e)}"
                self.logger.warning(msg)
                afterglow_errors.append(("decay_fitting", str(e)))

            # Extract afterglow parameters
            self.logger.info(f"Extracting afterglow parameters for {grb_name}")
            try:
                if decay_models:
                    afterglow_parameters = afterglow_analyzer.extract_afterglow_params(
                        decay_models
                    )
                    self.logger.debug(f"Extracted parameters: {len(afterglow_parameters)}")
            except Exception as e:
                msg = f"Failed to extract afterglow parameters: {str(e)}"
                self.logger.warning(msg)
                afterglow_errors.append(("param_extraction", str(e)))

            # Test closure relations
            self.logger.info(f"Testing closure relations for {grb_name}")
            try:
                if afterglow_light_curves and afterglow_parameters:
                    closure_relations = afterglow_analyzer.test_closure_relations(
                        afterglow_light_curves, afterglow_parameters
                    )
                    self.logger.info(f"Tested closure relations: {len(closure_relations)} tests")
            except Exception as e:
                msg = f"Failed to test closure relations: {str(e)}"
                self.logger.warning(msg)
                afterglow_errors.append(("closure_relations", str(e)))

            # Detect plateau and jet break
            self.logger.info(f"Detecting plateau and jet break for {grb_name}")
            try:
                if afterglow_light_curves:
                    jet_break_detection = afterglow_analyzer.detect_jet_break_and_plateau(
                        afterglow_light_curves
                    )
                    self.logger.info(f"Jet break detection completed")
            except Exception as e:
                msg = f"Failed to detect jet break: {str(e)}"
                self.logger.warning(msg)
                afterglow_errors.append(("jet_break", str(e)))

            # Multi-wavelength analysis
            self.logger.info(f"Performing multi-wavelength analysis for {grb_name}")
            try:
                if afterglow_light_curves:
                    mw_analysis = afterglow_analyzer.multi_wavelength_analysis(
                        afterglow_light_curves
                    )
                    afterglow_parameters.update({"multi_wavelength": mw_analysis})
                    self.logger.debug(f"Multi-wavelength analysis completed")
            except Exception as e:
                msg = f"Failed multi-wavelength analysis: {str(e)}"
                self.logger.warning(msg)
                afterglow_errors.append(("mw_analysis", str(e)))

            # Update context with results
            context["afterglow_light_curves"] = afterglow_light_curves
            context["decay_models"] = decay_models
            context["afterglow_parameters"] = afterglow_parameters
            context["closure_relations"] = closure_relations
            context["jet_break_detection"] = jet_break_detection
            context["afterglow_errors"] = afterglow_errors

            self.logger.info(f"Afterglow analysis completed for {grb_name}")
            return context

        except Exception as e:
            self.logger.error(f"Afterglow analysis failed for {grb_name}: {str(e)}")
            context["afterglow_errors"] = afterglow_errors + [("general", str(e))]
            context["afterglow_light_curves"] = {}
            context["decay_models"] = {}
            context["afterglow_parameters"] = {}
            context["closure_relations"] = {}
            context["jet_break_detection"] = {}
            return context


class ClassificationStage(PipelineStage):
    """Classify GRB and check correlations."""

    def __init__(self):
        """Initialize classification stage."""
        super().__init__(
            name="classification",
            description="Classify GRB (short/long/ultra-long), check correlations, assess progenitor",
            required_inputs=[
                "grb_name",
                "classifier",
                "temporal_properties",
                "derived_quantities",
            ],
            outputs=[
                "grb_classification",
                "classification_confidence",
                "correlations",
                "progenitor_assessment",
                "classification_errors",
            ],
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify GRB and evaluate correlations.

        Args:
            context: Must contain grb_name, classifier, temporal_properties

        Returns:
            Context with grb_classification, correlations, progenitor_assessment
        """
        self.validate_inputs(context)
        grb_name = context["grb_name"]
        classifier = context["classifier"]
        temporal_properties = context.get("temporal_properties", {})
        derived_quantities = context.get("derived_quantities", {})

        self.logger.info(f"Starting classification for {grb_name}")
        classification_errors = []
        grb_classification = {}
        classification_confidence = {}
        correlations = {}
        progenitor_assessment = {}

        try:
            # Classify GRB as short/long/ultra-long
            self.logger.info(f"Classifying GRB {grb_name}")
            try:
                grb_classification = classifier.classify_grb_type(
                    temporal_properties, derived_quantities
                )
                grb_type = grb_classification.get("type", "unknown")
                self.logger.info(f"GRB classified as: {grb_type}")
            except Exception as e:
                msg = f"Failed to classify GRB: {str(e)}"
                self.logger.warning(msg)
                classification_errors.append(("classification", str(e)))

            # Check correlations (Amati, Yonetoku, Ghirlanda)
            self.logger.info(f"Checking correlations for {grb_name}")
            try:
                if derived_quantities:
                    correlations = classifier.check_correlations(derived_quantities)
                    self.logger.info(f"Evaluated {len(correlations)} correlations")
            except Exception as e:
                msg = f"Failed to check correlations: {str(e)}"
                self.logger.warning(msg)
                classification_errors.append(("correlations", str(e)))

            # Assess progenitor type
            self.logger.info(f"Assessing progenitor type for {grb_name}")
            try:
                progenitor_assessment = classifier.assess_progenitor(
                    grb_classification, temporal_properties, derived_quantities
                )
                progenitor = progenitor_assessment.get("progenitor", "unknown")
                self.logger.info(f"Progenitor assessment: {progenitor}")
            except Exception as e:
                msg = f"Failed to assess progenitor: {str(e)}"
                self.logger.warning(msg)
                classification_errors.append(("progenitor", str(e)))

            # Calculate classification confidence
            self.logger.info(f"Computing classification confidence for {grb_name}")
            try:
                classification_confidence = classifier.compute_confidence_scores(
                    grb_classification, progenitor_assessment
                )
                confidence = classification_confidence.get("overall", 0.0)
                self.logger.info(f"Classification confidence: {confidence:.2%}")
            except Exception as e:
                msg = f"Failed to compute confidence: {str(e)}"
                self.logger.warning(msg)
                classification_errors.append(("confidence", str(e)))

            # Update context with results
            context["grb_classification"] = grb_classification
            context["classification_confidence"] = classification_confidence
            context["correlations"] = correlations
            context["progenitor_assessment"] = progenitor_assessment
            context["classification_errors"] = classification_errors

            self.logger.info(f"Classification completed for {grb_name}")
            return context

        except Exception as e:
            self.logger.error(f"Classification failed for {grb_name}: {str(e)}")
            context["classification_errors"] = classification_errors + [("general", str(e))]
            context["grb_classification"] = {}
            context["classification_confidence"] = {}
            context["correlations"] = {}
            context["progenitor_assessment"] = {}
            return context


class AIAnalysisStage(PipelineStage):
    """Run AI interpretation and generate comprehensive reports."""

    def __init__(self):
        """Initialize AI analysis stage."""
        super().__init__(
            name="ai_analysis",
            description="Run AI interpretation, detect anomalies, generate report and plots",
            required_inputs=["grb_name", "ai_interpreter"],
            outputs=[
                "ai_interpretation",
                "anomaly_detection",
                "report_path",
                "plot_paths",
                "summary",
                "ai_errors",
            ],
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI analysis and generate visualizations.

        Args:
            context: Must contain grb_name, ai_interpreter, and previous stage results

        Returns:
            Context with ai_interpretation, report_path, plot_paths
        """
        self.validate_inputs(context)
        grb_name = context["grb_name"]
        ai_interpreter = context["ai_interpreter"]

        self.logger.info(f"Starting AI analysis for {grb_name}")
        ai_errors = []
        ai_interpretation = {}
        anomaly_detection = {}
        report_path = None
        plot_paths = []
        summary = {}

        try:
            # Prepare analysis data from all previous stages
            analysis_data = {
                "grb_name": grb_name,
                "temporal": context.get("temporal_properties", {}),
                "spectral": context.get("spectral_parameters", {}),
                "afterglow": context.get("afterglow_parameters", {}),
                "classification": context.get("grb_classification", {}),
                "derived_quantities": context.get("derived_quantities", {}),
                "correlations": context.get("correlations", {}),
            }

            # Run AI interpretation
            self.logger.info(f"Running AI interpretation for {grb_name}")
            try:
                ai_interpretation = ai_interpreter.interpret_analysis(analysis_data)
                self.logger.info(f"AI interpretation completed")
            except Exception as e:
                msg = f"Failed to run AI interpretation: {str(e)}"
                self.logger.warning(msg)
                ai_errors.append(("interpretation", str(e)))

            # Detect anomalies
            self.logger.info(f"Running anomaly detection for {grb_name}")
            try:
                anomaly_detection = ai_interpreter.detect_anomalies(analysis_data)
                num_anomalies = len(anomaly_detection.get("anomalies", []))
                self.logger.info(f"Detected {num_anomalies} anomalies")
            except Exception as e:
                msg = f"Failed to detect anomalies: {str(e)}"
                self.logger.warning(msg)
                ai_errors.append(("anomaly_detection", str(e)))

            # Generate comprehensive report
            self.logger.info(f"Generating comprehensive report for {grb_name}")
            try:
                report_path = ai_interpreter.generate_report(
                    grb_name, analysis_data, ai_interpretation, anomaly_detection
                )
                self.logger.info(f"Report generated: {report_path}")
            except Exception as e:
                msg = f"Failed to generate report: {str(e)}"
                self.logger.warning(msg)
                ai_errors.append(("report_generation", str(e)))

            # Create visualization plots
            self.logger.info(f"Creating visualization plots for {grb_name}")
            try:
                plot_paths = ai_interpreter.create_plots(grb_name, analysis_data)
                self.logger.info(f"Created {len(plot_paths)} plots")
            except Exception as e:
                msg = f"Failed to create plots: {str(e)}"
                self.logger.warning(msg)
                ai_errors.append(("plot_creation", str(e)))

            # Generate executive summary
            self.logger.info(f"Generating executive summary for {grb_name}")
            try:
                summary = ai_interpreter.generate_summary(
                    grb_name, analysis_data, ai_interpretation
                )
                self.logger.debug(f"Summary: {summary.get('title', 'N/A')}")
            except Exception as e:
                msg = f"Failed to generate summary: {str(e)}"
                self.logger.warning(msg)
                ai_errors.append(("summary", str(e)))

            # Update context with results
            context["ai_interpretation"] = ai_interpretation
            context["anomaly_detection"] = anomaly_detection
            context["report_path"] = report_path
            context["plot_paths"] = plot_paths
            context["summary"] = summary
            context["ai_errors"] = ai_errors

            self.logger.info(f"AI analysis completed for {grb_name}")
            return context

        except Exception as e:
            self.logger.error(f"AI analysis failed for {grb_name}: {str(e)}")
            context["ai_errors"] = ai_errors + [("general", str(e))]
            context["ai_interpretation"] = {}
            context["anomaly_detection"] = {}
            context["report_path"] = None
            context["plot_paths"] = []
            context["summary"] = {}
            return context
