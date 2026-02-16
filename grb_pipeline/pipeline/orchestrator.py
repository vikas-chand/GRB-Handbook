"""Pipeline orchestration and execution engine."""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .stages import (
    PipelineStage,
    DataAcquisitionStage,
    TemporalAnalysisStage,
    SpectralAnalysisStage,
    AfterglowAnalysisStage,
    ClassificationStage,
    AIAnalysisStage,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full GRB analysis pipeline."""

    # Default stage sequence
    DEFAULT_STAGES = [
        "data_acquisition",
        "temporal_analysis",
        "spectral_analysis",
        "afterglow_analysis",
        "classification",
        "ai_analysis",
    ]

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the pipeline orchestrator.

        Args:
            config_path: Path to configuration file (JSON/YAML)
            config: Configuration dict (overrides config_path if both provided)

        Raises:
            FileNotFoundError: If config_path doesn't exist
            ValueError: If configuration is invalid
        """
        self.logger = logging.getLogger("grb_pipeline.orchestrator")
        self.config = config or {}

        # Load configuration from file if provided
        if config_path and not config:
            self._load_config(config_path)

        # Validate and set configuration defaults
        self._validate_and_set_defaults()

        # Initialize pipeline components
        self._initialize_components()

        # Initialize stage registry
        self.stages_registry = self._build_stage_registry()

        # Track execution state
        self.execution_history = {}
        self.current_context = {}

        self.logger.info(f"Pipeline orchestrator initialized with config: {self.config}")

    def _load_config(self, config_path: str) -> None:
        """Load configuration from file.

        Args:
            config_path: Path to JSON or YAML config file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() == ".json":
                    self.config = json.load(f)
                else:
                    try:
                        import yaml
                        self.config = yaml.safe_load(f)
                    except ImportError:
                        raise ValueError(
                            "YAML support requires 'pyyaml' package. "
                            "Use JSON format or install pyyaml."
                        )
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {str(e)}")

    def _validate_and_set_defaults(self) -> None:
        """Validate configuration and set sensible defaults."""
        # Database configuration
        if "database" not in self.config:
            self.config["database"] = {
                "type": "sqlite",
                "path": "./grb_analysis.db",
            }

        # Output directories
        if "output_dir" not in self.config:
            self.config["output_dir"] = "./results"

        if "data_dir" not in self.config:
            self.config["data_dir"] = "./data"

        if "temp_dir" not in self.config:
            self.config["temp_dir"] = "./temp"

        # Pipeline behavior
        if "continue_on_error" not in self.config:
            self.config["continue_on_error"] = True

        if "max_workers" not in self.config:
            self.config["max_workers"] = 4

        # Create directories if they don't exist
        for key in ["output_dir", "data_dir", "temp_dir"]:
            Path(self.config[key]).mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Configuration validated and defaults set")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components (fetchers, analyzers, etc).

        This method initializes dummy/placeholder components. In production,
        these would be initialized with actual implementations from the
        grb_pipeline modules.
        """
        # Initialize data manager
        try:
            from grb_pipeline.data import DataManager
            self.data_manager = DataManager(self.config.get("database", {}))
            self.logger.info("DataManager initialized")
        except ImportError:
            self.logger.warning("Could not import DataManager, using placeholder")
            self.data_manager = _PlaceholderComponent("DataManager")

        # Initialize GCN fetcher
        try:
            from grb_pipeline.fetchers import GCNFetcher
            self.gcn_fetcher = GCNFetcher(self.config.get("database", {}))
            self.logger.info("GCNFetcher initialized")
        except ImportError:
            self.logger.warning("Could not import GCNFetcher, using placeholder")
            self.gcn_fetcher = _PlaceholderComponent("GCNFetcher")

        # Initialize temporal analyzer
        try:
            from grb_pipeline.analysis import TemporalAnalyzer
            self.temporal_analyzer = TemporalAnalyzer(self.config)
            self.logger.info("TemporalAnalyzer initialized")
        except ImportError:
            self.logger.warning("Could not import TemporalAnalyzer, using placeholder")
            self.temporal_analyzer = _PlaceholderComponent("TemporalAnalyzer")

        # Initialize spectral analyzer
        try:
            from grb_pipeline.analysis import SpectralAnalyzer
            self.spectral_analyzer = SpectralAnalyzer(self.config)
            self.logger.info("SpectralAnalyzer initialized")
        except ImportError:
            self.logger.warning("Could not import SpectralAnalyzer, using placeholder")
            self.spectral_analyzer = _PlaceholderComponent("SpectralAnalyzer")

        # Initialize afterglow analyzer
        try:
            from grb_pipeline.analysis import AfterglowAnalyzer
            self.afterglow_analyzer = AfterglowAnalyzer(self.config)
            self.logger.info("AfterglowAnalyzer initialized")
        except ImportError:
            self.logger.warning("Could not import AfterglowAnalyzer, using placeholder")
            self.afterglow_analyzer = _PlaceholderComponent("AfterglowAnalyzer")

        # Initialize classifier
        try:
            from grb_pipeline.analysis import Classifier
            self.classifier = Classifier(self.config)
            self.logger.info("Classifier initialized")
        except ImportError:
            self.logger.warning("Could not import Classifier, using placeholder")
            self.classifier = _PlaceholderComponent("Classifier")

        # Initialize AI interpreter
        try:
            from grb_pipeline.ai import AIInterpreter
            self.ai_interpreter = AIInterpreter(self.config)
            self.logger.info("AIInterpreter initialized")
        except ImportError:
            self.logger.warning("Could not import AIInterpreter, using placeholder")
            self.ai_interpreter = _PlaceholderComponent("AIInterpreter")

    def _build_stage_registry(self) -> Dict[str, PipelineStage]:
        """Build registry of available pipeline stages.

        Returns:
            Dict mapping stage names to stage instances
        """
        stages = {
            "data_acquisition": DataAcquisitionStage(),
            "temporal_analysis": TemporalAnalysisStage(),
            "spectral_analysis": SpectralAnalysisStage(),
            "afterglow_analysis": AfterglowAnalysisStage(),
            "classification": ClassificationStage(),
            "ai_analysis": AIAnalysisStage(),
        }
        self.logger.debug(f"Stage registry built with {len(stages)} stages")
        return stages

    def run(
        self,
        grb_name: str,
        stages: Optional[List[str]] = None,
        skip_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline for a single GRB.

        Args:
            grb_name: Name/ID of the GRB to analyze
            stages: List of stages to run (default: all). If specified, only runs these stages.
            skip_stages: List of stages to skip (default: none)

        Returns:
            Complete results dictionary with all stage outputs

        Raises:
            ValueError: If invalid stage names provided
        """
        self.logger.info(f"Starting pipeline for {grb_name}")
        self.logger.info(f"Stages: {stages}, Skip: {skip_stages}")

        # Determine which stages to run
        if stages is None:
            stages_to_run = self.DEFAULT_STAGES.copy()
        else:
            stages_to_run = stages

        if skip_stages:
            stages_to_run = [s for s in stages_to_run if s not in skip_stages]

        # Validate stage names
        invalid_stages = set(stages_to_run) - set(self.stages_registry.keys())
        if invalid_stages:
            raise ValueError(
                f"Invalid stage names: {invalid_stages}. "
                f"Valid stages: {list(self.stages_registry.keys())}"
            )

        # Build pipeline context
        context = self._build_context(grb_name)

        # Execute pipeline
        results = self._execute_stages(grb_name, stages_to_run, context)

        # Save results
        self._save_results(grb_name, results)

        # Generate report and plots
        try:
            report_path = self._generate_report(grb_name, results)
            plot_paths = self._create_plots(grb_name, results)
            results["report_path"] = report_path
            results["plot_paths"] = plot_paths
        except Exception as e:
            self.logger.error(f"Failed to generate report/plots: {str(e)}")

        self.logger.info(f"Pipeline completed for {grb_name}")
        return results

    def run_batch(
        self, grb_names: List[str], parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """Process multiple GRBs.

        Args:
            grb_names: List of GRB names to process
            parallel: If True, process in parallel (limited by max_workers config)

        Returns:
            List of results dicts, one per GRB
        """
        self.logger.info(
            f"Starting batch processing for {len(grb_names)} GRBs "
            f"(parallel={parallel})"
        )

        if not parallel:
            # Sequential processing
            results = []
            for i, grb_name in enumerate(grb_names, 1):
                self.logger.info(f"Processing {i}/{len(grb_names)}: {grb_name}")
                try:
                    result = self.run(grb_name)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {grb_name}: {str(e)}")
                    results.append(
                        {
                            "grb_name": grb_name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
            return results

        # Parallel processing
        results = []
        max_workers = self.config.get("max_workers", 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run, grb_name): grb_name
                for grb_name in grb_names
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                grb_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed {completed}/{len(grb_names)}: {grb_name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {grb_name}: {str(e)}")
                    results.append(
                        {
                            "grb_name": grb_name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

        self.logger.info(f"Batch processing completed: {len(results)} GRBs processed")
        return results

    def run_from_stage(self, grb_name: str, stage_name: str) -> Dict[str, Any]:
        """Resume pipeline from a specific stage, loading prior results from DB.

        Args:
            grb_name: Name/ID of the GRB
            stage_name: Stage to resume from

        Returns:
            Complete results from stage onwards

        Raises:
            ValueError: If invalid stage name
        """
        if stage_name not in self.stages_registry:
            raise ValueError(
                f"Invalid stage: {stage_name}. "
                f"Valid stages: {list(self.stages_registry.keys())}"
            )

        self.logger.info(f"Resuming pipeline for {grb_name} from stage: {stage_name}")

        # Build context and load prior results from database
        context = self._build_context(grb_name)
        try:
            prior_results = self.data_manager.load_prior_results(grb_name)
            context.update(prior_results)
            self.logger.info(f"Loaded prior results for {grb_name}")
        except Exception as e:
            self.logger.warning(f"Could not load prior results: {str(e)}")

        # Get stages from resume point onwards
        stage_index = self.DEFAULT_STAGES.index(stage_name)
        stages_to_run = self.DEFAULT_STAGES[stage_index:]

        # Execute pipeline
        results = self._execute_stages(grb_name, stages_to_run, context)

        # Save results
        self._save_results(grb_name, results)

        self.logger.info(f"Pipeline resumed and completed for {grb_name}")
        return results

    def get_status(self, grb_name: str) -> Dict[str, Any]:
        """Check what stages have been completed for a GRB.

        Args:
            grb_name: Name/ID of the GRB

        Returns:
            Dict with completion status for each stage
        """
        self.logger.debug(f"Getting status for {grb_name}")

        status = {
            "grb_name": grb_name,
            "timestamp": datetime.utcnow().isoformat(),
            "stages": {},
        }

        try:
            db_status = self.data_manager.get_completion_status(grb_name)
            status["stages"] = db_status
        except Exception as e:
            self.logger.warning(f"Could not get status from database: {str(e)}")
            status["error"] = str(e)

        return status

    def _build_context(self, grb_name: str) -> Dict[str, Any]:
        """Initialize pipeline context with GRB info and components.

        Args:
            grb_name: Name/ID of the GRB

        Returns:
            Initial context dict
        """
        self.logger.debug(f"Building context for {grb_name}")

        context = {
            "grb_name": grb_name,
            "timestamp_start": datetime.utcnow().isoformat(),
            "config": self.config,
            # Core components
            "data_manager": self.data_manager,
            "gcn_fetcher": self.gcn_fetcher,
            "temporal_analyzer": self.temporal_analyzer,
            "spectral_analyzer": self.spectral_analyzer,
            "afterglow_analyzer": self.afterglow_analyzer,
            "classifier": self.classifier,
            "ai_interpreter": self.ai_interpreter,
        }

        return context

    def _execute_stages(
        self, grb_name: str, stages: List[str], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a sequence of stages.

        Args:
            grb_name: GRB name for logging
            stages: List of stage names to execute
            context: Pipeline context dict

        Returns:
            Updated context with all results
        """
        self.logger.info(f"Executing {len(stages)} stages for {grb_name}")

        for i, stage_name in enumerate(stages, 1):
            self.logger.info(f"[{i}/{len(stages)}] Executing stage: {stage_name}")

            try:
                stage = self.stages_registry[stage_name]
                self.logger.debug(f"Stage description: {stage.description}")

                # Execute stage
                context = stage.execute(context)

                # Track execution
                self.execution_history[f"{grb_name}_{stage_name}"] = {
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat(),
                }

                self.logger.info(f"Stage {stage_name} completed successfully")

            except ValueError as e:
                # Input validation failure
                error_msg = f"Stage {stage_name} validation error: {str(e)}"
                self.logger.error(error_msg)

                if not self.config.get("continue_on_error", True):
                    self.execution_history[f"{grb_name}_{stage_name}"] = {
                        "status": "failed",
                        "error": error_msg,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    raise

                self.execution_history[f"{grb_name}_{stage_name}"] = {
                    "status": "skipped",
                    "reason": "input_validation_error",
                    "error": error_msg,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            except Exception as e:
                # Stage execution error
                error_msg = f"Stage {stage_name} execution error: {str(e)}"
                self.logger.error(error_msg)
                self.logger.debug(traceback.format_exc())

                if not self.config.get("continue_on_error", True):
                    self.execution_history[f"{grb_name}_{stage_name}"] = {
                        "status": "failed",
                        "error": error_msg,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    raise

                self.execution_history[f"{grb_name}_{stage_name}"] = {
                    "status": "failed",
                    "error": error_msg,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        context["timestamp_end"] = datetime.utcnow().isoformat()
        self.logger.info(f"Stage execution completed for {grb_name}")
        return context

    def _save_results(self, grb_name: str, results: Dict[str, Any]) -> None:
        """Persist all results to database.

        Args:
            grb_name: GRB name
            results: Results dict from all stages
        """
        self.logger.info(f"Saving results to database for {grb_name}")

        try:
            # Extract key results for database storage
            db_payload = {
                "grb_name": grb_name,
                "timestamp": datetime.utcnow().isoformat(),
                "temporal_properties": results.get("temporal_properties", {}),
                "spectral_parameters": results.get("spectral_parameters", {}),
                "afterglow_parameters": results.get("afterglow_parameters", {}),
                "grb_classification": results.get("grb_classification", {}),
                "derived_quantities": results.get("derived_quantities", {}),
                "correlations": results.get("correlations", {}),
                "ai_interpretation": results.get("ai_interpretation", {}),
                "execution_history": self.execution_history,
            }

            if hasattr(self.data_manager, "save_analysis_results"):
                self.data_manager.save_analysis_results(grb_name, db_payload)

            self.logger.info(f"Results saved to database for {grb_name}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")

    def _generate_report(self, grb_name: str, results: Dict[str, Any]) -> Optional[str]:
        """Create final analysis report.

        Args:
            grb_name: GRB name
            results: Results dict

        Returns:
            Path to generated report file, or None if generation failed
        """
        self.logger.info(f"Generating analysis report for {grb_name}")

        try:
            report_dir = Path(self.config["output_dir"]) / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)

            report_path = report_dir / f"{grb_name}_analysis_report.txt"

            # Generate text report
            with open(report_path, "w") as f:
                f.write(f"GRB ANALYSIS REPORT\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(f"GRB Name: {grb_name}\n")
                f.write(
                    f"Generated: {results.get('timestamp_end', datetime.utcnow().isoformat())}\n"
                )
                f.write(f"\n")

                # Temporal properties
                if results.get("temporal_properties"):
                    f.write(f"TEMPORAL PROPERTIES\n")
                    f.write(f"{'-' * 40}\n")
                    for key, value in results["temporal_properties"].items():
                        f.write(f"  {key}: {value}\n")
                    f.write(f"\n")

                # Spectral parameters
                if results.get("spectral_parameters"):
                    f.write(f"SPECTRAL PARAMETERS\n")
                    f.write(f"{'-' * 40}\n")
                    for key, value in results["spectral_parameters"].items():
                        f.write(f"  {key}: {value}\n")
                    f.write(f"\n")

                # Classification
                if results.get("grb_classification"):
                    f.write(f"CLASSIFICATION\n")
                    f.write(f"{'-' * 40}\n")
                    for key, value in results["grb_classification"].items():
                        f.write(f"  {key}: {value}\n")
                    f.write(f"\n")

                # AI Summary
                if results.get("summary"):
                    f.write(f"AI SUMMARY\n")
                    f.write(f"{'-' * 40}\n")
                    summary = results["summary"]
                    f.write(f"{summary.get('text', 'No summary available')}\n")

            self.logger.info(f"Report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            return None

    def _create_plots(self, grb_name: str, results: Dict[str, Any]) -> List[str]:
        """Generate all visualization plots.

        Args:
            grb_name: GRB name
            results: Results dict

        Returns:
            List of paths to generated plot files
        """
        self.logger.info(f"Creating visualization plots for {grb_name}")

        plot_paths = []

        try:
            plots_dir = Path(self.config["output_dir"]) / "plots" / grb_name
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Use AI interpreter to create plots if available
            if hasattr(self.ai_interpreter, "create_plots"):
                try:
                    ai_plots = self.ai_interpreter.create_plots(grb_name, results)
                    if ai_plots:
                        plot_paths.extend(ai_plots)
                except Exception as e:
                    self.logger.warning(f"Failed to create AI plots: {str(e)}")

            # Log plot paths
            if plot_paths:
                self.logger.info(f"Created {len(plot_paths)} plots")
            else:
                self.logger.info(f"No plots generated (may be handled by AI interpreter)")

            return plot_paths

        except Exception as e:
            self.logger.error(f"Failed to create plots: {str(e)}")
            return []


class _PlaceholderComponent:
    """Placeholder component for development/testing when real components unavailable."""

    def __init__(self, name: str):
        """Initialize placeholder."""
        self.name = name
        self.logger = logging.getLogger(f"grb_pipeline.placeholder.{name}")
        self.logger.warning(f"Using placeholder for {name}")

    def __getattr__(self, attr: str):
        """Return a placeholder method for any attribute access."""
        def placeholder_method(*args, **kwargs):
            self.logger.debug(
                f"Placeholder method called: {attr}({args}, {kwargs})"
            )
            return {}

        return placeholder_method
