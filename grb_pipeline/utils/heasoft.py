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


def run_xspec(
    input_script: str,
    timeout: int = 600,
    working_dir: Optional[str] = None,
) -> str:
    """
    Run XSPEC (spectral fitting).

    XSPEC is the primary tool for GRB spectral analysis, supporting
    various models (Band, Cutoff Power Law, Blackbody, etc.).

    Args:
        input_script: XSPEC command script
        timeout: Command timeout in seconds
        working_dir: Working directory

    Returns:
        XSPEC output and fit results

    Raises:
        HEASoftNotFoundError: If xspec not found
        HEASoftError: If xspec fails
    """
    xspec_path = _find_tool("xspec")

    try:
        result = subprocess.run(
            [xspec_path, "-"],
            input=input_script.encode(),
            capture_output=True,
            timeout=timeout,
            cwd=working_dir,
        )

        output = result.stdout.decode()

        # xspec returns non-zero even on success sometimes
        logger.info("xspec execution completed")
        return output

    except subprocess.TimeoutExpired:
        logger.error(f"xspec timed out after {timeout} seconds")
        raise HEASoftError(f"xspec timed out")
    except Exception as e:
        logger.error(f"xspec error: {e}")
        raise HEASoftError(f"xspec error: {e}")


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
    Check which HEASoft tools are available.

    Returns:
        Dictionary mapping tool names to availability (True/False)
    """
    tools = ["xselect", "xspec", "batbinevt", "fselect", "fkeyprint", "ftools"]
    available = {}

    for tool in tools:
        try:
            _find_tool(tool)
            available[tool] = True
            logger.info(f"✓ {tool} available")
        except HEASoftNotFoundError:
            available[tool] = False
            logger.warning(f"✗ {tool} not found")

    return available


def create_xspec_script(
    spectrum_file: str,
    response_file: str,
    background_file: Optional[str] = None,
    models: List[str] = None,
    energy_range: tuple = (0.3, 10.0),
) -> str:
    """
    Generate XSPEC command script for spectral fitting.

    Args:
        spectrum_file: Input spectrum FITS file
        response_file: Instrument response matrix
        background_file: Background spectrum file
        models: List of models to fit (default: ["band", "cutoffpl", "po"])
        energy_range: Energy range to fit (keV)

    Returns:
        XSPEC script string
    """
    if models is None:
        models = ["band", "cutoffpl", "po"]

    script = f"""
data {spectrum_file}
ign 0.0-{energy_range[0]}
ign {energy_range[1]}-**
"""

    if response_file:
        script += f"response {response_file}\n"
    if background_file:
        script += f"back {background_file}\n"

    script += """
method leven 10 0.01
abund wilm
"""

    for model in models:
        if model.lower() == "band":
            script += """
model grbcomp
1.0, -1.0, -2.0, 100.0, 1.0
fit
"""
        elif model.lower() == "cutoffpl":
            script += """
model cutoffpl
1.0, -1.0, 100.0
fit
"""
        elif model.lower() == "po":
            script += """
model pow
1.0, -1.0
fit
"""

    script += """
log grb_fit.log
show all
save all grb_fit.xcm
quit
"""

    return script
