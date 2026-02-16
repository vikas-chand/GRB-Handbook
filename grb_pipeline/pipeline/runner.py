"""Command-line interface for GRB pipeline execution."""
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .orchestrator import PipelineOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging verbosity.

    Args:
        verbose: If True, set logging to DEBUG level
    """
    if verbose:
        logging.getLogger("grb_pipeline").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        logging.getLogger("grb_pipeline").setLevel(logging.INFO)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="grb_pipeline",
        description="GRB Analysis Pipeline - Comprehensive analysis of gamma-ray bursts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for a single GRB
  python -m grb_pipeline run GRB230101A

  # Run specific stages only
  python -m grb_pipeline run GRB230101A --stages temporal,spectral

  # Process multiple GRBs from file
  python -m grb_pipeline batch grbs.txt

  # Query analysis results
  python -m grb_pipeline query GRB230101A

  # Show all analyzed GRBs
  python -m grb_pipeline catalog

  # Generate report from existing results
  python -m grb_pipeline report GRB230101A

  # Parse specific GCN circular
  python -m grb_pipeline gcn 12345

  # Initialize database and directories
  python -m grb_pipeline init
        """,
    )

    # Global arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON/YAML)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute",
        required=True,
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run pipeline for a single GRB",
    )
    run_parser.add_argument(
        "grb_name",
        type=str,
        help="GRB name/ID to analyze (e.g., GRB230101A)",
    )
    run_parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run "
             "(e.g., 'temporal,spectral'). "
             "Valid stages: data_acquisition, temporal_analysis, spectral_analysis, "
             "afterglow_analysis, classification, ai_analysis",
    )
    run_parser.add_argument(
        "--skip-stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to skip",
    )
    run_parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI analysis stage",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running pipeline",
    )
    run_parser.set_defaults(func=cmd_run)

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple GRBs from file",
    )
    batch_parser.add_argument(
        "file",
        type=str,
        help="Path to file with GRB names (one per line)",
    )
    batch_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process GRBs in parallel",
    )
    batch_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running pipeline",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query database for existing analysis results",
    )
    query_parser.add_argument(
        "grb_name",
        type=str,
        help="GRB name/ID to query",
    )
    query_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    query_parser.set_defaults(func=cmd_query)

    # catalog command
    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Show all analyzed GRBs in database",
    )
    catalog_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of entries to show (default: 50)",
    )
    catalog_parser.add_argument(
        "--sort",
        type=str,
        default="timestamp",
        choices=["timestamp", "name", "type"],
        help="Sort order (default: timestamp)",
    )
    catalog_parser.set_defaults(func=cmd_catalog)

    # report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate report from existing results",
    )
    report_parser.add_argument(
        "grb_name",
        type=str,
        help="GRB name/ID to generate report for",
    )
    report_parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json", "html"],
        help="Report format (default: text)",
    )
    report_parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Include plot paths in report",
    )
    report_parser.set_defaults(func=cmd_report)

    # gcn command
    gcn_parser = subparsers.add_parser(
        "gcn",
        help="Parse and display specific GCN circular",
    )
    gcn_parser.add_argument(
        "circular_number",
        type=int,
        help="GCN circular number to parse",
    )
    gcn_parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw GCN circular text",
    )
    gcn_parser.set_defaults(func=cmd_gcn)

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check pipeline status for a GRB",
    )
    status_parser.add_argument(
        "grb_name",
        type=str,
        help="GRB name/ID to check status for",
    )
    status_parser.set_defaults(func=cmd_status)

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize database and directories",
    )
    init_parser.set_defaults(func=cmd_init)

    return parser


def cmd_run(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline run <GRB_NAME>

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    grb_name = args.grb_name
    logger.info(f"Running pipeline for {grb_name}")

    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
        logger.info(f"Running specific stages: {stages}")

    skip_stages = None
    if args.skip_stages:
        skip_stages = [s.strip() for s in args.skip_stages.split(",")]

    if args.no_ai:
        if skip_stages is None:
            skip_stages = []
        skip_stages.append("ai_analysis")
        logger.info("Skipping AI analysis stage")

    # Dry run
    if args.dry_run:
        logger.info("DRY RUN - Pipeline not executed")
        stages_to_run = stages or orchestrator.DEFAULT_STAGES
        if skip_stages:
            stages_to_run = [s for s in stages_to_run if s not in skip_stages]
        logger.info(f"Would execute stages: {stages_to_run}")
        return 0

    # Run pipeline
    try:
        results = orchestrator.run(grb_name, stages=stages, skip_stages=skip_stages)

        # Print summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Pipeline completed successfully for {grb_name}")
        logger.info(f"{'=' * 60}")

        # Print key results
        if results.get("temporal_properties"):
            logger.info(f"Temporal: T90={results['temporal_properties'].get('t90', 'N/A')}")
        if results.get("grb_classification"):
            grb_type = results["grb_classification"].get("type", "unknown")
            logger.info(f"Classification: {grb_type}")
        if results.get("report_path"):
            logger.info(f"Report: {results['report_path']}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_batch(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline batch <FILE>

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    batch_file = Path(args.file)

    if not batch_file.exists():
        logger.error(f"Batch file not found: {batch_file}")
        return 1

    # Read GRB names from file
    try:
        with open(batch_file, "r") as f:
            grb_names = [line.strip() for line in f if line.strip()]

        if not grb_names:
            logger.error("Batch file contains no GRB names")
            return 1

        logger.info(f"Loaded {len(grb_names)} GRB names from {batch_file}")

    except Exception as e:
        logger.error(f"Failed to read batch file: {str(e)}")
        return 1

    # Dry run
    if args.dry_run:
        logger.info("DRY RUN - Pipeline not executed")
        logger.info(f"Would process: {grb_names}")
        logger.info(f"Parallel: {args.parallel}")
        return 0

    # Run batch
    try:
        logger.info(f"Starting batch processing ({len(grb_names)} GRBs, parallel={args.parallel})")

        results = orchestrator.run_batch(grb_names, parallel=args.parallel)

        # Print summary
        successful = sum(1 for r in results if r.get("status") != "failed")
        failed = len(results) - successful

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Batch processing completed")
        logger.info(f"  Successful: {successful}/{len(results)}")
        logger.info(f"  Failed: {failed}/{len(results)}")
        logger.info(f"{'=' * 60}")

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_query(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline query <GRB_NAME>

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    grb_name = args.grb_name
    logger.info(f"Querying results for {grb_name}")

    try:
        # Get status
        status = orchestrator.get_status(grb_name)

        if args.json:
            # Output as JSON
            print(json.dumps(status, indent=2))
        else:
            # Output as text
            print(f"\nGRB Analysis Status: {grb_name}")
            print(f"Timestamp: {status.get('timestamp', 'N/A')}")

            if status.get("error"):
                print(f"Error: {status['error']}")
            else:
                print(f"\nStage Completion Status:")
                for stage, info in status.get("stages", {}).items():
                    completed = "✓" if info.get("completed") else "✗"
                    print(f"  {completed} {stage}")

        return 0

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return 1


def cmd_catalog(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline catalog

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    logger.info("Retrieving GRB catalog from database")

    try:
        # Try to get catalog from data manager
        if hasattr(orchestrator.data_manager, "get_catalog"):
            catalog = orchestrator.data_manager.get_catalog(limit=args.limit, sort=args.sort)

            print(f"\nGRB Catalog ({len(catalog)} entries)")
            print(f"{'=' * 60}")
            print(f"{'GRB Name':<20} {'Type':<12} {'Timestamp':<25}")
            print(f"{'-' * 60}")

            for entry in catalog:
                grb_name = entry.get("grb_name", "N/A")
                grb_type = entry.get("type", "N/A")
                timestamp = entry.get("timestamp", "N/A")
                print(f"{grb_name:<20} {grb_type:<12} {timestamp:<25}")

            return 0
        else:
            logger.warning("Data manager does not support catalog retrieval")
            print("Catalog not available (feature not implemented)")
            return 0

    except Exception as e:
        logger.error(f"Failed to retrieve catalog: {str(e)}")
        return 1


def cmd_report(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline report <GRB_NAME>

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    grb_name = args.grb_name
    logger.info(f"Generating report for {grb_name}")

    try:
        # Get status
        status = orchestrator.get_status(grb_name)

        if status.get("error"):
            logger.error(f"No data available for {grb_name}")
            return 1

        # Generate report (text format)
        print(f"\nGRB Analysis Report: {grb_name}")
        print(f"{'=' * 60}")
        print(f"Generated: {datetime.utcnow().isoformat()}")
        print(f"\nStage Completion Status:")
        for stage, info in status.get("stages", {}).items():
            completed = "✓" if info.get("completed") else "✗"
            print(f"  {completed} {stage}")

        return 0

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return 1


def cmd_gcn(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline gcn <CIRCULAR_NUMBER>

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    circular_number = args.circular_number
    logger.info(f"Fetching GCN circular {circular_number}")

    try:
        # Fetch GCN circular
        if hasattr(orchestrator.gcn_fetcher, "fetch_circular"):
            circular = orchestrator.gcn_fetcher.fetch_circular(circular_number)

            if args.raw:
                # Show raw text
                print(circular.get("text", "No text available"))
            else:
                # Show parsed circular
                print(f"\nGCN Circular {circular_number}")
                print(f"{'=' * 60}")
                print(f"Title: {circular.get('title', 'N/A')}")
                print(f"Author: {circular.get('author', 'N/A')}")
                print(f"Date: {circular.get('date', 'N/A')}")
                print(f"\n{circular.get('summary', 'No summary available')}")

            return 0
        else:
            logger.warning("GCN fetcher does not support circular retrieval")
            print("GCN circular not available (feature not implemented)")
            return 0

    except Exception as e:
        logger.error(f"Failed to fetch GCN circular: {str(e)}")
        return 1


def cmd_status(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline status <GRB_NAME>

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    grb_name = args.grb_name
    logger.info(f"Checking status for {grb_name}")

    try:
        status = orchestrator.get_status(grb_name)

        print(f"\nPipeline Status: {grb_name}")
        print(f"{'=' * 60}")
        print(f"Timestamp: {status.get('timestamp', 'N/A')}")

        if status.get("error"):
            print(f"Status: No data available")
        else:
            print(f"Status: Analysis completed")
            print(f"\nStage Details:")
            for stage, info in status.get("stages", {}).items():
                completed = "✓ COMPLETED" if info.get("completed") else "✗ PENDING"
                print(f"  {stage}: {completed}")

        return 0

    except Exception as e:
        logger.error(f"Failed to check status: {str(e)}")
        return 1


def cmd_init(args, orchestrator: PipelineOrchestrator) -> int:
    """Execute: python -m grb_pipeline init

    Args:
        args: Parsed command-line arguments
        orchestrator: Pipeline orchestrator instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    logger.info("Initializing pipeline")

    try:
        # Directories are already created by orchestrator
        logger.info(f"Output directory: {orchestrator.config['output_dir']}")
        logger.info(f"Data directory: {orchestrator.config['data_dir']}")
        logger.info(f"Database: {orchestrator.config['database']}")

        # Initialize database
        if hasattr(orchestrator.data_manager, "initialize"):
            orchestrator.data_manager.initialize()
            logger.info("Database initialized")

        print(f"\nPipeline initialized successfully")
        print(f"Configuration: {orchestrator.config}")

        return 0

    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        return 1


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0=success, 1=failure)
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    logger.info(f"GRB Pipeline started with command: {args.command}")

    try:
        # Load configuration
        config = None
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            orchestrator = PipelineOrchestrator(config_path=args.config)
        else:
            orchestrator = PipelineOrchestrator()

        # Override directories if specified
        if args.output_dir:
            orchestrator.config["output_dir"] = args.output_dir
        if args.data_dir:
            orchestrator.config["data_dir"] = args.data_dir

        # Execute command
        if hasattr(args, "func"):
            exit_code = args.func(args, orchestrator)
            return exit_code
        else:
            parser.print_help()
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
