# GRB Analysis Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-Alpha-orange)](https://github.com/yourusername/grb-pipeline)

A comprehensive end-to-end analysis framework for gamma-ray bursts (GRBs) with multi-wavelength, multi-messenger, and AI-powered capabilities.

## Overview

The GRB Analysis Pipeline is a professional-grade framework designed for astronomers, astrophysicists, and data scientists analyzing gamma-ray burst observations. It provides:

- **Full end-to-end analysis** from raw Swift and Fermi observations to multi-wavelength correlations
- **Multi-messenger integration** supporting gravitational wave, neutrino, and electromagnetic data
- **AI-powered analysis** using Claude for GCN circular parsing and intelligent interpretation
- **Automated workflows** with configurable pipeline stages and data reduction
- **Advanced statistics** including Bayesian fitting, MCMC sampling, and correlation analysis
- **Rich visualizations** with interactive plots and publication-quality figures

The pipeline handles complex multi-instrument datasets automatically, applying state-of-the-art analysis techniques while providing full transparency and customizability.

## Features

- **Data Acquisition**
  - Automatic Swift/XRT light curve and spectral extraction
  - Fermi/GBM burst analysis with automatic trigger identification
  - Multi-wavelength data aggregation (optical, X-ray, radio, infrared)
  - Gravitational wave and neutrino multi-messenger data integration
  - Real-time GCN circular monitoring and parsing

- **Temporal Analysis**
  - Automatic T90/T50 calculation with statistical uncertainties
  - Rise time and decay parameter extraction
  - Variability quantification (fractional RMS, hardness ratios)
  - Flare detection and characterization
  - Stochastic process model fitting

- **Spectral Fitting**
  - Multi-component spectral decomposition
  - Band function and power-law fitting with iminuit/emcee
  - Photon spectral hardness calculation
  - Spectral evolution tracking over time
  - Systematic uncertainty quantification

- **Afterglow Modeling**
  - Forward shock model implementation
  - Reverse shock afterglow analysis
  - Multi-band afterglow light curve fitting
  - Energy injection scenarios
  - Jet break identification

- **Multi-wavelength Correlations**
  - Amati relation analysis (Epeak vs Eiso)
  - Yonetoku relation (Epeak vs Liso)
  - Hardness-duration classification
  - GRB variability correlations
  - Statistical significance testing

- **AI-Powered Analysis**
  - Claude-based GCN circular automatic parsing
  - Intelligent afterglow interpretation and anomaly detection
  - Natural language report generation
  - Contextual analysis combining multi-wavelength data

- **Automated Reporting**
  - PDF/HTML report generation with figures
  - Catalog-ready metadata extraction
  - Publication-formatted tables and figures
  - Multi-source data synthesis

- **Visualization**
  - Interactive light curve plots with Plotly
  - Spectral SED displays with model overlays
  - Hardness-duration diagrams
  - Multi-wavelength correlation plots
  - Bayesian corner plots for parameter posteriors
  - Publication-quality matplotlib figures

## Installation

### Basic Installation

```bash
pip install grb-pipeline
```

### Development Installation

```bash
git clone https://github.com/yourusername/grb-pipeline.git
cd grb-pipeline
pip install -e ".[dev]"
```

### Requirements

- **Python**: 3.9 or higher
- **Core Dependencies**: numpy, scipy, pandas, astropy, astroquery
- **Fitting**: iminuit, emcee, corner
- **Visualization**: matplotlib, plotly, seaborn
- **API**: anthropic (for Claude AI features)

### Optional: HEASoft Setup

For advanced Swift/Fermi analysis, install HEASoft:

```bash
# Download from: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/
# Follow installation instructions, then verify:
xselect -v
xspec -v
```

### Environment Configuration

Set your Anthropic API key for AI features:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.env` file in your project root:

```
ANTHROPIC_API_KEY=your-api-key-here
```

## Quick Start

### Python API

```python
from grb_pipeline.pipeline import PipelineOrchestrator
from grb_pipeline.core import PipelineConfig

# Initialize with default configuration
pipeline = PipelineOrchestrator()

# Run full pipeline on a GRB
results = pipeline.run("GRB230307A")

# Access results
print(results.grb_event.name)
print(results.temporal_analysis.t90)
print(results.spectral_analysis.epeak)
print(results.afterglow_parameters.alpha)
print(results.ai_interpretation)

# Generate automated report
report_path = pipeline.generate_report("GRB230307A", output_format="pdf")
```

### Command Line Interface

```bash
# Run full pipeline
python -m grb_pipeline run GRB230307A

# List available commands
python -m grb_pipeline --help

# Analyze with custom config
python -m grb_pipeline run GRB230307A --config custom_config.yaml

# Generate report
python -m grb_pipeline report GRB230307A --format pdf --output reports/

# Query database
python -m grb_pipeline query --t90-min 0.1 --t90-max 2.0

# Batch processing
python -m grb_pipeline batch --list grb_list.txt --parallel 4
```

## Pipeline Stages

The analysis pipeline consists of 6 integrated stages:

### Stage 1: Data Acquisition
Automatically fetches and aggregates data from multiple sources (Swift, Fermi, multi-wavelength archives). Handles GCN circular parsing, trigger identification, and data quality assessment.

### Stage 2: Temporal Analysis
Calculates temporal properties including T90, T50, rise time, and decay slopes. Detects variability features and performs stochastic process modeling.

### Stage 3: Spectral Analysis
Performs energy-dependent spectral fitting using the Band function and power-law models. Extracts Epeak, photon indices, and spectral hardness evolution.

### Stage 4: Afterglow Analysis
Models multi-wavelength afterglow data using forward/reverse shock theory. Identifies jet breaks, energy injection, and constrains progenitor properties.

### Stage 5: Correlation Analysis
Calculates GRB correlations (Amati, Yonetoku, hardness-duration). Tests statistical significance and identifies outliers or special events.

### Stage 6: AI Interpretation & Reporting
Uses Claude to parse GCN circulars, interpret analysis results, and generate comprehensive automated reports with visualizations.

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `run` | Execute full pipeline for one GRB | `python -m grb_pipeline run GRB230307A` |
| `report` | Generate formatted report | `python -m grb_pipeline report GRB230307A --format pdf` |
| `query` | Search GRB database | `python -m grb_pipeline query --t90-min 0.1 --t90-max 2` |
| `batch` | Process multiple GRBs | `python -m grb_pipeline batch --list grbs.txt` |
| `catalog` | Update GRB catalog from online sources | `python -m grb_pipeline catalog --update` |
| `validate` | Check data integrity | `python -m grb_pipeline validate GRB230307A` |
| `export` | Export analysis results | `python -m grb_pipeline export GRB230307A --format json` |
| `ai-analyze` | Run AI interpretation | `python -m grb_pipeline ai-analyze GRB230307A` |
| `status` | Check processing status | `python -m grb_pipeline status` |
| `--version` | Display version | `python -m grb_pipeline --version` |

## Project Structure

```
grb-pipeline/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── pyproject.toml            # Project metadata
├── setup.py                  # Setup configuration
├── requirements.txt          # Python dependencies
│
├── grb_pipeline/             # Main package
│   ├── __init__.py
│   ├── __main__.py           # CLI entry point
│   │
│   ├── core/                 # Core data models and database
│   │   ├── models.py         # GRBEvent, TimingAnalysis, SpectralAnalysis
│   │   ├── database.py       # SQLite database interface
│   │   ├── config.py         # Configuration management
│   │   └── constants.py      # Physical and cosmological constants
│   │
│   ├── data/                 # Data acquisition and processing
│   │   ├── swift.py          # Swift/XRT data fetching
│   │   ├── fermi.py          # Fermi/GBM data handling
│   │   ├── gcn_parser.py     # GCN circular parsing
│   │   ├── multiwavelength.py# Multi-wavelength integration
│   │   └── cache.py          # Data caching system
│   │
│   ├── analysis/             # Analysis modules
│   │   ├── temporal.py       # T90, T50, variability analysis
│   │   ├── spectral.py       # Band function, spectral fitting
│   │   ├── lightcurve.py     # Light curve fitting models
│   │   ├── afterglow.py      # Afterglow modeling
│   │   ├── correlations.py   # Amati, Yonetoku relations
│   │   └── classification.py # GRB classification (long/short/extended)
│   │
│   ├── pipeline/             # Pipeline orchestration
│   │   ├── orchestrator.py   # PipelineOrchestrator main class
│   │   ├── stages.py         # Pipeline stage implementations
│   │   └── runner.py         # CLI interface
│   │
│   ├── ai/                   # Claude AI integration
│   │   ├── interpreter.py    # AI-based interpretation
│   │   ├── gcn_analyzer.py   # GCN circular analysis
│   │   ├── report_generator.py # Automated report generation
│   │   └── prompts.py        # AI prompt templates
│   │
│   ├── visualization/        # Plotting and visualization
│   │   ├── lightcurve.py     # Light curve plots
│   │   ├── spectral.py       # Spectral plots
│   │   ├── correlations.py   # Correlation diagrams
│   │   ├── interactive.py    # Plotly interactive plots
│   │   └── publication.py    # Publication-quality figures
│   │
│   └── utils/                # Utility functions
│       ├── fits.py           # FITS file handling
│       ├── heasoft.py        # HEASoft integration
│       ├── cosmology.py      # Cosmological calculations
│       ├── statistics.py     # Statistical utilities
│       └── logging.py        # Logging configuration
│
├── config/                   # Configuration files
│   ├── default.yaml          # Default pipeline configuration
│   ├── swift.yaml            # Swift-specific settings
│   ├── fermi.yaml            # Fermi-specific settings
│   └── analysis.yaml         # Analysis parameter defaults
│
├── data/                     # Data directory
│   ├── catalogs/             # GRB catalogs and references
│   ├── cache/                # Cached data from online sources
│   └── examples/             # Example GRB data
│
├── notebooks/                # Jupyter analysis notebooks
│   ├── 01_single_grb_analysis.ipynb
│   ├── 02_population_study.ipynb
│   └── 03_advanced_fitting.ipynb
│
├── tests/                    # Unit and integration tests
│   ├── conftest.py           # Pytest fixtures
│   ├── test_core.py          # Core module tests
│   ├── test_analysis.py      # Analysis module tests
│   ├── test_pipeline.py      # Pipeline integration tests
│   └── fixtures/             # Test data fixtures
│
└── docs/                     # Documentation
    ├── api.md                # API documentation
    ├── tutorial.md           # Tutorials
    └── faq.md                # Frequently asked questions
```

## Configuration

### Default Configuration

The pipeline uses YAML configuration files. View default settings:

```bash
cat config/default.yaml
```

### Customizing Configuration

Create a custom configuration file and pass it to the pipeline:

```python
from grb_pipeline.core import PipelineConfig

config = PipelineConfig.from_yaml("my_config.yaml")
pipeline = PipelineOrchestrator(config=config)
results = pipeline.run("GRB230307A")
```

### Key Configuration Options

```yaml
pipeline:
  stages: [1, 2, 3, 4, 5, 6]  # Enable/disable stages
  parallel: false              # Parallel execution
  timeout: 3600               # Timeout in seconds

data_acquisition:
  missions: [swift, fermi, xrt, gbm]
  cache_data: true
  query_timeout: 300

analysis:
  spectral_model: "band"      # band, powerlaw, cutoff
  fit_method: "iminuit"       # iminuit, emcee, nested
  mcmc_samples: 2000

visualization:
  format: "pdf"               # pdf, png, html
  dpi: 300
  style: "publication"

ai:
  enabled: true
  model: "claude-opus-4-6"
  parse_gcn: true
  generate_report: true
```

## AI Integration

The pipeline integrates Claude for intelligent analysis:

### Setup

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Features

- **GCN Circular Parsing**: Automatically extracts key information from GCN circulars
- **Intelligent Interpretation**: Claude analyzes results and identifies noteworthy features
- **Automated Reporting**: Generates natural language analysis summaries
- **Anomaly Detection**: Flags unusual GRB properties or potential follow-up targets
- **Multi-wavelength Context**: Synthesizes multi-wavelength data into coherent interpretation

### Example Usage

```python
from grb_pipeline.ai import AIInterpreter

interpreter = AIInterpreter()
report = interpreter.analyze_grb(results)
print(report)

# Generate full report
full_report = interpreter.generate_full_report(results, include_gcn=True)
```

## Supported Missions

### Gamma-ray Missions

- **Swift**: XRT, UVOT, BAT (2004-present)
- **Fermi**: GBM, LAT (2008-present)
- **Suzaku**: XIS, PIN (legacy)
- **INTEGRAL**: IBIS/ISGRI (legacy)

### Multi-wavelength Follow-up

- **Optical/IR**: SDSS, Pan-STARRS, 2MASS, Gaia
- **X-ray**: Chandra, XMM-Newton, Swift XRT
- **Radio**: VLA, ALMA, ATCA
- **Ultra-high Energy**: IceCube, HAWC

### Multi-messenger Data

- **Gravitational Waves**: LIGO, Virgo (GW events)
- **Neutrinos**: IceCube alerts
- **Transient Alerts**: ZTF, ASAS-SN

## Analysis Modules

### Temporal Analysis (`temporal.py`)
- T90/T50 calculation with error estimation
- Rise time and decay slope extraction
- Flare detection using Bayesian blocks
- Variability quantification (fractional RMS, hardness)
- Stochastic variability modeling

### Spectral Analysis (`spectral.py`)
- Band function fitting with iminuit/emcee
- Photon and energy spectral parameters
- Spectral hardness tracking
- Systematic uncertainty propagation
- Multi-component spectral decomposition

### Light Curve Modeling (`lightcurve.py`)
- Gaussian pulse model fitting
- Broken power-law decay modeling
- Multi-peak decomposition
- Automatic model selection via AIC
- Parameter uncertainty quantification

### Afterglow Analysis (`afterglow.py`)
- Forward shock modeling
- Reverse shock analysis
- Jet break identification
- Energy injection scenarios
- Constraints on initial Lorentz factor

### Correlation Analysis (`correlations.py`)
- Amati relation (Epeak vs Eiso) with Bayesian fit
- Yonetoku relation (Epeak vs Liso)
- Hardness-duration classification
- Statistical significance testing
- Outlier identification and characterization

### Classification (`classification.py`)
- Long/short/extended GRB classification
- Hardness-duration diagram analysis
- Multi-dimensional clustering
- Progenitor type inference

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Format with `black`, check with `flake8`
2. **Type Hints**: Use full type annotations
3. **Testing**: Add tests for new features (pytest)
4. **Documentation**: Update docstrings and README
5. **Commits**: Use descriptive commit messages

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check code style
black --check grb_pipeline/
flake8 grb_pipeline/

# Format code
black grb_pipeline/
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{grb_pipeline_2024,
  title={GRB Analysis Pipeline: End-to-End Gamma-Ray Burst Analysis with Claude AI},
  author={Vikas},
  year={2024},
  url={https://github.com/yourusername/grb-pipeline}
}
```

## References

Key papers and resources:

- Band et al. 1993 - Band function spectral model
- Amati et al. 2002 - Epeak-Eiso correlation
- Yonetoku et al. 2004 - Epeak-Liso correlation
- Kouveliotou et al. 1993 - GRB classification
- Frail et al. 2001 - Jet breaks and collimation

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact & Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/grb-pipeline/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/grb-pipeline/discussions)
- **Email**: vikas@example.com
- **Documentation**: [Wiki & Tutorials](https://github.com/yourusername/grb-pipeline/wiki)

---

**Status**: This project is under active development. Features and APIs may change.

**Last Updated**: February 2024

**Version**: 0.1.0
