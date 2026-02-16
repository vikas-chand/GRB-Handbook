"""Setup configuration for GRB Analysis Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="grb-pipeline",
    version="0.1.0",
    author="Vikas",
    description="Comprehensive Gamma-Ray Burst Analysis Framework with Claude AI Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/grb-pipeline",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "grb-pipeline=grb_pipeline.pipeline.runner:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="gamma-ray burst GRB astronomy X-ray spectroscopy",
    project_urls={
        "Documentation": "https://github.com/yourusername/grb-pipeline/wiki",
        "Source": "https://github.com/yourusername/grb-pipeline",
        "Tracker": "https://github.com/yourusername/grb-pipeline/issues",
    },
)
