"""HEASoft command-line tool wrappers for GRB data processing."""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any
import shutil

logger = logging.getLogger(__name__)


class HEASoftError(Exception):
    """Exception raised for HEASoft command execution errors."""

    pass


class HEASoftNotFoundError(HEASoftError):
    """Exception raised when HEASoft tools are not found in PATH."""

    pass


def _find_tool(tool_name: str) -> str:
    """
    Find HEASoft tool in system PATH.

    Args:
        tool_name: Name of HEASoft tool (e.g., 'xselect', 'xspec')

    Returns:
        Full path to tool

    Raises:
        HEASoftNotFoundError: If tool not found
    """
    tool_path = shutil.which(tool_name)
    if tool_path is None:
        logger.error(f"HEASoft tool '{tool_name}' not found in PATH")
        raise HEASoftNotFoundError(f"HEASoft tool '{tool_name}' not found. Install HEASoft or add to PATH.")
    return tool_path


def run_command(
    command: List[str],
    input_text: Optional[str] = None,
    timeout: int = 300,
    working_dir: Optional[str] = None,
) -> str:
    """
    Execute shell command with error handling.

    Args:
        command: List of command arguments
        input_text: Standard input text (for interactive commands)
        timeout: Command timeout in seconds
        working_dir: Working directory for command

    Returns:
        Standard output from command

    Raises:
        HEASoftError: If command fails
    """
    try:
        logger.debug(f"Running command: {' '.join(command)}")

        result = subprocess.run(
            command,
            input=input_text.encode() if input_text else None,
            capture_output=True,
            timeout=timeout,
            cwd=working_dir,
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Command failed with return code {result.returncode}: {error_msg}")
            raise HEASoftError(f"Command failed: {error_msg}")

        output = result.stdout.decode()
        logger.debug(f"Command completed successfully")
        return output

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        raise HEASoftError(f"Command timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        raise HEASoftError(f"Command execution error: {e}")


def run_xselect(
    input_script: str,
    timeout: int = 300,
    working_dir: Optional[str] = None,
) -> str:
    """
    Run XSELECT (interactive FITS file filtering).

    XSELECT is used for advanced light curve and spectral extraction,
    filtering by energy, time, and other criteria.

    Args:
        input_script: XSELECT command script (newline-separated commands)
        timeout: Command timeout in seconds
        working_dir: Working directory

    Returns:
        XSELECT output

    Raises:
        HEASoftNotFoundError: If xselect not found
        HEASoftError: If xselect fails
    """
    xselect_path = _find_tool("xselect")

    try:
        result = subprocess.run(
            [xselect_path],
            input=input_script.encode(),
            capture_output=True,
            timeout=timeout,
            cwd=working_dir,
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"xselect failed: {error_msg}")
            raise HEASoftError(f"xselect failed: {error_msg}")

        output = result.stdout.decode()
        logger.info("xselect completed successfully")
        return output

    except subprocess.TimeoutExpired:
        logger.error(f"xselect timed out after {timeout} seconds")
        raise HEASoftError(f"xselect timed out")
    except Exception as e:
        logger.error(f"xselect error: {e}")
        raise HEASoftError(f"xselect error: {e}")


def check_3ml_installation() -> Dict[str, bool]:
    """
    Check 3ML (threeML) and related package availability.

    3ML is the primary spectral fitting framework for this pipeline.
    It uses pyXSPEC as a backend when available.

    Returns:
        Dictionary mapping package names to availability
    """
    packages = {
        'threeML': False,
        'astromodels': False,
        'pyXSPEC': False,
    }

    for pkg in packages:
        try:
            __import__(pkg.lower() if pkg != 'pyXSPEC' else 'xspec')
            packages[pkg] = True
            logger.info(f"  {pkg} available")
        except ImportError:
            # threeML uses lowercase
            try:
                __import__(pkg)
                packages[pkg] = True
                logger.info(f"  {pkg} available")
            except ImportError:
                logger.warning(f"  {pkg} not found")

    return packages


def run_batbinevt(
    event_file: str,
    output_file: str,
    energy_min: float = 15.0,
    energy_max: float = 150.0,
    time_bin: float = 0.064,
    timeout: int = 300,
) -> str:
    """
    Run BATBINEVT (Swift BAT event binning).

    Bins BAT event file into light curve with specified time resolution
    and energy range.

    Args:
        event_file: Input BAT event file
        output_file: Output light curve file
        energy_min: Minimum energy (keV)
        energy_max: Maximum energy (keV)
        time_bin: Time bin size (seconds)
        timeout: Command timeout

    Returns:
        Command output

    Raises:
        HEASoftNotFoundError: If batbinevt not found
        HEASoftError: If command fails
    """
    batbinevt_path = _find_tool("batbinevt")

    try:
        cmd = [
            batbinevt_path,
            f"infile={event_file}",
            f"outfile={output_file}",
            f"timebin={time_bin}",
            f"energymin={energy_min}",
            f"energymax={energy_max}",
            "clobber=yes",
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=timeout)

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"batbinevt failed: {error_msg}")
            raise HEASoftError(f"batbinevt failed: {error_msg}")

        output = result.stdout.decode()
        logger.info(f"batbinevt completed: {output_file}")
        return output

    except subprocess.TimeoutExpired:
        logger.error(f"batbinevt timed out")
        raise HEASoftError(f"batbinevt timed out")
    except Exception as e:
        logger.error(f"batbinevt error: {e}")
        raise HEASoftError(f"batbinevt error: {e}")


def run_fselect(
    input_file: str,
    output_file: str,
    filter_expr: str,
    timeout: int = 300,
) -> str:
    """
    Run FSELECT (FITS file filtering).

    Filters FITS file by column expression (e.g., energy range, time range).

    Args:
        input_file: Input FITS file
        output_file: Output FITS file
        filter_expr: Filter expression (e.g., "energy >= 0.3 && energy <= 10")
        timeout: Command timeout

    Returns:
        Command output

    Raises:
        HEASoftNotFoundError: If fselect not found
        HEASoftError: If command fails
    """
    fselect_path = _find_tool("fselect")

    try:
        cmd = [fselect_path, f"{input_file}[{filter_expr}]", output_file, "yes"]

        result = subprocess.run(cmd, capture_output=True, timeout=timeout)

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"fselect failed: {error_msg}")
            raise HEASoftError(f"fselect failed: {error_msg}")

        output = result.stdout.decode()
        logger.info(f"fselect completed: {output_file}")
        return output

    except subprocess.TimeoutExpired:
        logger.error("fselect timed out")
        raise HEASoftError("fselect timed out")
    except Exception as e:
        logger.error(f"fselect error: {e}")
        raise HEASoftError(f"fselect error: {e}")


def run_fkeyprint(
    fits_file: str,
    keyword: str,
    hdu: int = 0,
) -> str:
    """
    Run FKEYPRINT (extract FITS header keyword).

    Args:
        fits_file: FITS file path
        keyword: Keyword name to extract
        hdu: HDU index

    Returns:
        Keyword value as string

    Raises:
        HEASoftNotFoundError: If fkeyprint not found
        HEASoftError: If command fails
    """
    fkeyprint_path = _find_tool("fkeyprint")

    try:
        cmd = [fkeyprint_path, f"{fits_file}[{hdu}]", keyword]

        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"fkeyprint failed: {error_msg}")
            raise HEASoftError(f"fkeyprint failed: {error_msg}")

        output = result.stdout.decode().strip()
        logger.debug(f"fkeyprint: {keyword} = {output}")
        return output

    except Exception as e:
        logger.error(f"fkeyprint error: {e}")
        raise HEASoftError(f"fkeyprint error: {e}")


def check_heasoft_installation() -> Dict[str, bool]:
    """
    Check which HEASoft tools and spectral fitting packages are available.

    Returns:
        Dictionary mapping tool/package names to availability (True/False)
    """
    # HEASoft command-line tools
    tools = ["xselect", "batbinevt", "fselect", "fkeyprint", "ftools"]
    available = {}

    for tool in tools:
        try:
            _find_tool(tool)
            available[tool] = True
            logger.info(f"  {tool} available")
        except HEASoftNotFoundError:
            available[tool] = False
            logger.warning(f"  {tool} not found")

    # 3ML and spectral fitting packages
    available.update(check_3ml_installation())

    return available


def create_3ml_spectral_plugins(
    tte_files: Dict[str, str],
    rsp_files: Dict[str, str],
    src_interval: tuple,
    bg_intervals: tuple = ((-50, -5), (100, 200)),
    nai_energy: str = "8-900",
    bgo_energy: str = "250-30000",
) -> str:
    """
    Generate 3ML (threeML) code snippet for loading GBM data as spectral plugins.

    3ML's TimeSeriesBuilder handles GBM TTE/CSPEC data natively, with
    polynomial background fitting and pyXSPEC as the spectral engine.

    Args:
        tte_files: Dict of {detector: tte_file_path}
        rsp_files: Dict of {detector: rsp_file_path}
        src_interval: Source time interval (tstart, tstop) in seconds
        bg_intervals: Background intervals ((pre_start, pre_stop), (post_start, post_stop))
        nai_energy: NaI energy selection string for 3ML
        bgo_energy: BGO energy selection string for 3ML

    Returns:
        Python code string for 3ML plugin setup
    """
    code_lines = [
        "from threeML import TimeSeriesBuilder",
        "",
        "plugins = []",
    ]

    for det, tte in tte_files.items():
        rsp = rsp_files.get(det, '')
        code_lines.extend([
            f"",
            f"# Detector {det}",
            f"ts_{det} = TimeSeriesBuilder.from_gbm_tte(",
            f"    '{det}',",
            f"    tte_file='{tte}',",
            f"    rsp_file='{rsp}',",
            f")",
            f"ts_{det}.set_background_interval(",
            f"    '{bg_intervals[0][0]:.1f}-{bg_intervals[0][1]:.1f}',",
            f"    '{bg_intervals[1][0]:.1f}-{bg_intervals[1][1]:.1f}',",
            f")",
            f"ts_{det}.set_active_time_interval('{src_interval[0]:.3f}-{src_interval[1]:.3f}')",
            f"plugin_{det} = ts_{det}.to_spectrumlike()",
        ])

        if det.startswith('n'):
            code_lines.append(f"plugin_{det}.set_active_measurements('{nai_energy}')")
        else:
            code_lines.append(f"plugin_{det}.set_active_measurements('{bgo_energy}')")

        code_lines.append(f"plugins.append(plugin_{det})")

    return '\n'.join(code_lines)
