"""Configuration management for GRB analysis pipeline."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class PipelineConfig:
    """
    Configuration manager for GRB analysis pipeline.

    Handles loading, storing, and accessing configuration from YAML files
    with support for environment variable overrides and reasonable defaults.

    Example:
        >>> config = PipelineConfig.from_yaml("config/default.yaml")
        >>> data_dir = config.get("paths.data_dir")
        >>> config.set("analysis.log_level", "DEBUG")
        >>> config.save("config/custom.yaml")
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Optional dictionary of configuration values,
                         or a string path to a YAML config file.
        """
        if isinstance(config_dict, (str, Path)):
            # Allow passing a file path directly
            config_path = Path(config_dict)
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_dict}, using defaults")
                self.config = {}
        else:
            self.config = config_dict or {}
        self._apply_defaults()
        logger.debug(f"Initialized config with {len(self._flatten_dict(self.config))} items")

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PipelineConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {path}")
            return cls(config_dict)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """
        Create configuration from dictionary.

        Args:
            d: Configuration dictionary

        Returns:
            PipelineConfig instance
        """
        return cls(d)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key with optional dot notation (e.g., "paths.data_dir")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Creates intermediate dictionaries as needed.

        Args:
            key: Configuration key with optional dot notation
            value: Value to set
        """
        keys = key.split(".")
        current = self.config

        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        logger.debug(f"Set config {key} = {value}")

    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save configuration
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self.config.copy()

    def _apply_defaults(self) -> None:
        """Apply default configuration values."""
        defaults = {
            "pipeline": {
                "name": "GRB Analysis Pipeline",
                "version": "0.1.0",
                "log_level": "INFO",
            },
            "paths": {
                "data_dir": "./data",
                "cache_dir": "./data/cache",
                "output_dir": "./output",
                "database": "./data/grb_database.db",
            },
            "archives": {
                "swift_base_url": "https://www.swift.ac.uk/xrt_curves",
                "swift_bat_url": "https://swift.gsfc.nasa.gov/results/batgrbcat",
                "fermi_gbm_url": "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm",
                "fermi_lat_url": "https://fermi.gsfc.nasa.gov/ssc/data/access",
                "gcn_base_url": "https://gcn.gsfc.nasa.gov/gcn3",
                "heasarc_url": "https://heasarc.gsfc.nasa.gov",
            },
            "cosmology": {
                "H0": 67.4,
                "Omega_m": 0.315,
                "Omega_Lambda": 0.685,
            },
            "analysis": {
                "default_energy_range": [0.3, 10000],
                "spectral_models": ["Band", "CPL", "PL"],
                "t90_confidence": 0.9,
                "afterglow_min_points": 5,
                "classification_t90_threshold": 2.0,
            },
            "ai": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "temperature": 0.3,
            },
            "visualization": {
                "figure_dpi": 150,
                "default_figsize": [10, 7],
                "color_palette": "viridis",
                "save_format": "png",
            },
        }

        # Merge defaults with existing config (existing values take precedence)
        self._merge_defaults(defaults, self.config)

    def _merge_defaults(self, defaults: Dict, config: Dict) -> None:
        """
        Recursively merge default values into config.

        Args:
            defaults: Default configuration dictionary
            config: Current configuration dictionary (modified in place)
        """
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config.get(key), dict):
                self._merge_defaults(value, config[key])

    def _flatten_dict(self, d: Dict, parent_key: str = "") -> Dict:
        """
        Flatten nested dictionary using dot notation keys.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix

        Returns:
            Flattened dictionary
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    # ========================================================================
    # Convenience Properties
    # ========================================================================

    @property
    def data_dir(self) -> Path:
        """Data directory path."""
        return Path(self.get("paths.data_dir"))

    @property
    def cache_dir(self) -> Path:
        """Cache directory path."""
        return Path(self.get("paths.cache_dir"))

    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        return Path(self.get("paths.output_dir"))

    @property
    def db_path(self) -> Path:
        """Database file path."""
        return Path(self.get("paths.database"))

    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get("pipeline.log_level", "INFO")

    @property
    def api_model(self) -> str:
        """Anthropic API model name."""
        return self.get("ai.model", "claude-sonnet-4-20250514")

    @property
    def max_tokens(self) -> int:
        """Maximum tokens for API requests."""
        return self.get("ai.max_tokens", 4096)

    @property
    def temperature(self) -> float:
        """Temperature for API requests."""
        return self.get("ai.temperature", 0.3)

    @property
    def h0(self) -> float:
        """Hubble constant (km/s/Mpc)."""
        return self.get("cosmology.H0", 67.4)

    @property
    def omega_matter(self) -> float:
        """Matter density parameter."""
        return self.get("cosmology.Omega_m", 0.315)

    @property
    def omega_lambda(self) -> float:
        """Dark energy density parameter."""
        return self.get("cosmology.Omega_Lambda", 0.685)

    @property
    def swift_bat_url(self) -> str:
        """Swift BAT catalog URL."""
        return self.get("archives.swift_bat_url", "https://swift.gsfc.nasa.gov/results/batgrbcat")

    @property
    def fermi_gbm_url(self) -> str:
        """Fermi GBM archive URL."""
        return self.get("archives.fermi_gbm_url", "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm")

    @property
    def gcn_base_url(self) -> str:
        """GCN circular base URL."""
        return self.get("archives.gcn_base_url", "https://gcn.gsfc.nasa.gov/gcn3")

    def ensure_paths_exist(self) -> None:
        """Create all configured directories if they don't exist."""
        for path_key in ["data_dir", "cache_dir", "output_dir"]:
            path = self.get(f"paths.{path_key}")
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path}")
