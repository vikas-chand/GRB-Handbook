"""
Fermi GBM detector geometry and angle calculation.

Calculates the angle between a GRB source position and each Fermi GBM detector,
then selects the best detectors based on proximity. Based on the spacecraft
coordinate system from Meegan et al. and angular distance code from
Vianello's gtburst.

Key functions:
    select_gbm_detectors(ra, dec, trigdat_file) -> list: Select best NaI + BGO detectors
    get_angular_distance(ra1, dec1, ra2, dec2) -> float: Vincenty formula on sphere

Usage:
    from grb_pipeline.utils.gbm_geometry import select_gbm_detectors
    detectors = select_gbm_detectors(ra=123.45, dec=-45.67, trigdat_file='trigdat.fits')
    # Returns sorted list of (detector_name, angle_degrees)
"""

import math
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# GBM Detector directions in spacecraft coordinates (Meegan et al.)
# Format: {detector_name: (azimuth_deg, zenith_deg)}
# ------------------------------------------------------------------
DETECTOR_DIRECTIONS = {
    'n0':  (45.89,  20.58),
    'n1':  (45.11,  45.31),
    'n2':  (58.44,  90.21),
    'n3':  (314.87, 45.24),
    'n4':  (303.15, 90.27),
    'n5':  (3.35,   89.79),
    'n6':  (224.93, 20.43),
    'n7':  (224.62, 46.18),
    'n8':  (236.61, 89.97),
    'n9':  (135.19, 45.55),
    'na':  (123.73, 90.42),
    'nb':  (183.74, 90.32),
    'b0':  (0.0,    90.0),   # BGO 0 — +X face
    'b1':  (180.0,  90.0),   # BGO 1 — -X face
}


class Vector:
    """
    3D vector for spacecraft coordinate transformations.

    Based on Vianello's gtburst angularDistance.py code. Supports rotation,
    cross product, and angle calculations needed for GBM detector geometry.
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.vector = [x, y, z]

    @classmethod
    def from_array(cls, arr) -> 'Vector':
        """Create from array-like [x, y, z]."""
        return cls(arr[0], arr[1], arr[2])

    @classmethod
    def from_spherical(cls, azimuth_deg: float, zenith_deg: float) -> 'Vector':
        """
        Create vector from spacecraft spherical coordinates.

        Parameters
        ----------
        azimuth_deg : float
            Azimuth angle in degrees (0-360)
        zenith_deg : float
            Zenith angle in degrees (0 = along +Z axis, 90 = in XY plane)
        """
        az = math.radians(azimuth_deg)
        zen = math.radians(zenith_deg)
        return cls(
            x=math.sin(zen) * math.cos(az),
            y=math.sin(zen) * math.sin(az),
            z=math.cos(zen),
        )

    @classmethod
    def from_ra_dec(cls, ra_deg: float, dec_deg: float) -> 'Vector':
        """Create unit vector from RA/Dec in degrees."""
        ra = math.radians(ra_deg)
        dec = math.radians(dec_deg)
        return cls(
            x=math.cos(dec) * math.cos(ra),
            y=math.cos(dec) * math.sin(ra),
            z=math.sin(dec),
        )

    def dot(self, other: 'Vector') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector') -> 'Vector':
        """Cross product."""
        import numpy as np
        c = np.cross(self.vector, other.vector)
        return Vector(c[0], c[1], c[2])

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def angle_to(self, other: 'Vector') -> float:
        """Angle between two vectors in degrees."""
        cos_angle = self.dot(other) / (self.norm() * other.norm())
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def rotate(self, angle_deg: float, axis: 'Vector') -> 'Vector':
        """
        Rotate this vector around an axis by a given angle (Rodrigues' formula).

        From Vianello's gtburst Vector class.

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees
        axis : Vector
            Axis of rotation

        Returns
        -------
        Vector
            Rotated vector
        """
        import numpy as np
        ang = math.radians(angle_deg)
        ax = np.array(axis.vector)
        ax = ax / np.sqrt(np.dot(ax, ax))
        a = math.cos(ang / 2)
        b, c, d = -ax * math.sin(ang / 2)
        matrix = np.array([
            [a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c)],
            [2*(b*c-a*d), a*a+c*c-b*b-d*d, 2*(c*d+a*b)],
            [2*(b*d+a*c), 2*(c*d-a*b), a*a+d*d-b*b-c*c],
        ])
        result = np.dot(matrix, self.vector)
        return Vector(result[0], result[1], result[2])

    def __getitem__(self, idx):
        return self.vector[idx]


def get_ra_dec(
    ra_scx: float,
    dec_scx: float,
    ra_scz: float,
    dec_scz: float,
    theta: float,
    phi: float,
) -> Tuple[float, float]:
    """
    Convert spacecraft (theta, phi) to (RA, Dec) in J2000.

    Given the spacecraft X-axis (SCX) and Z-axis (SCZ) pointing directions,
    convert detector coordinates (theta, phi) to sky coordinates (RA, Dec).

    From Vianello's gtburst getRaDec function.

    Parameters
    ----------
    ra_scx, dec_scx : float
        Spacecraft X-axis RA/Dec in degrees
    ra_scz, dec_scz : float
        Spacecraft Z-axis RA/Dec in degrees
    theta : float
        Zenith angle in spacecraft coordinates (degrees)
    phi : float
        Azimuth angle in spacecraft coordinates (degrees)

    Returns
    -------
    tuple of (ra, dec)
        Sky coordinates in degrees
    """
    vx = Vector.from_ra_dec(ra_scx, dec_scx)
    vz = Vector.from_ra_dec(ra_scz, dec_scz)

    # Rotate X around Z by phi
    vxx = vx.rotate(phi, vz)
    # Y = Z cross X'
    vy = vz.cross(vxx)
    # Rotate Z around Y by theta
    vzz = vz.rotate(theta, vy)

    ra = math.degrees(math.atan2(vzz[1], vzz[0]))
    dec = math.degrees(math.asin(vzz[2]))

    if ra < 0:
        ra += 360.0
    return ra, dec


def get_theta_phi(
    ra_scx: float,
    dec_scx: float,
    ra_scz: float,
    dec_scz: float,
    ra: float,
    dec: float,
) -> Tuple[float, float]:
    """
    Convert sky (RA, Dec) to spacecraft (theta, phi) coordinates.

    From Vianello's gtburst getThetaPhi function.

    Parameters
    ----------
    ra_scx, dec_scx : float
        Spacecraft X-axis RA/Dec in degrees
    ra_scz, dec_scz : float
        Spacecraft Z-axis RA/Dec in degrees
    ra, dec : float
        Source RA/Dec in degrees

    Returns
    -------
    tuple of (theta, phi)
        Spacecraft coordinates in degrees
    """
    v0 = Vector.from_ra_dec(ra, dec)
    vx = Vector.from_ra_dec(ra_scx, dec_scx)
    vz = Vector.from_ra_dec(ra_scz, dec_scz)
    vy = vz.cross(vx)

    theta = v0.angle_to(vz)
    phi = math.degrees(math.atan2(vy.dot(v0), vx.dot(v0)))
    if phi < 0:
        phi += 360.0
    return theta, phi


def get_detector_angle(
    ra_scx: float,
    dec_scx: float,
    ra_scz: float,
    dec_scz: float,
    source_ra: float,
    source_dec: float,
    detector: str,
) -> float:
    """
    Calculate angle between a source and a specific GBM detector.

    Uses the spacecraft X and Z axis pointing directions to transform
    detector coordinates to sky coordinates, then calculates the angular
    distance to the source.

    From Vianello's gtburst getDetectorAngle function.

    Parameters
    ----------
    ra_scx, dec_scx : float
        Spacecraft X-axis RA/Dec in degrees
    ra_scz, dec_scz : float
        Spacecraft Z-axis RA/Dec in degrees
    source_ra, source_dec : float
        Source RA/Dec in degrees
    detector : str
        Detector name ('n0'-'nb', 'b0', 'b1')

    Returns
    -------
    float
        Angle in degrees
    """
    if detector not in DETECTOR_DIRECTIONS:
        raise ValueError(f"Unknown detector: {detector}")

    az, zen = DETECTOR_DIRECTIONS[detector]
    det_ra, det_dec = get_ra_dec(ra_scx, dec_scx, ra_scz, dec_scz, zen, az)
    return get_angular_distance(source_ra, source_dec, det_ra, det_dec)


def get_all_detector_angles(
    ra_scx: float,
    dec_scx: float,
    ra_scz: float,
    dec_scz: float,
    source_ra: float,
    source_dec: float,
) -> List[Tuple[str, float]]:
    """
    Calculate angle from source to all 14 GBM detectors using spacecraft pointing.

    This is the method from the GBMangles notebook — uses SCX and SCZ pointing
    directions with Vianello's coordinate transformation code.

    Parameters
    ----------
    ra_scx, dec_scx : float
        Spacecraft X-axis RA/Dec (from trigdat header RA_SCX, DEC_SCX)
    ra_scz, dec_scz : float
        Spacecraft Z-axis RA/Dec (from trigdat header RA_SCZ, DEC_SCZ)
    source_ra, source_dec : float
        Source RA/Dec in degrees

    Returns
    -------
    list of (detector_name, angle_deg)
        All 14 detectors with angles, sorted by angle (closest first)
    """
    angles = []
    for det_name in DETECTOR_DIRECTIONS:
        angle = get_detector_angle(
            ra_scx, dec_scx, ra_scz, dec_scz, source_ra, source_dec, det_name
        )
        angles.append((det_name, angle))

    angles.sort(key=lambda x: x[1])
    return angles


def get_angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate angular distance between two sky positions using the Vincenty formula.

    More numerically stable than the simple arccos formula for small and large angles.

    Parameters
    ----------
    ra1, dec1 : float
        First position in degrees
    ra2, dec2 : float
        Second position in degrees

    Returns
    -------
    float
        Angular distance in degrees
    """
    ra1_r = math.radians(ra1)
    dec1_r = math.radians(dec1)
    ra2_r = math.radians(ra2)
    dec2_r = math.radians(dec2)

    delta_ra = ra2_r - ra1_r

    # Vincenty formula
    numerator = math.sqrt(
        (math.cos(dec2_r) * math.sin(delta_ra)) ** 2 +
        (math.cos(dec1_r) * math.sin(dec2_r) -
         math.sin(dec1_r) * math.cos(dec2_r) * math.cos(delta_ra)) ** 2
    )
    denominator = (
        math.sin(dec1_r) * math.sin(dec2_r) +
        math.cos(dec1_r) * math.cos(dec2_r) * math.cos(delta_ra)
    )

    return math.degrees(math.atan2(numerator, denominator))


def get_detector_angles_from_trigdat(
    ra: float,
    dec: float,
    trigdat_file: str,
) -> List[Tuple[str, float]]:
    """
    Calculate angle from source to each GBM detector using trigdat spacecraft attitude.

    The trigdat file contains the spacecraft quaternion at trigger time,
    which defines the transformation from J2000 to spacecraft coordinates.

    Parameters
    ----------
    ra : float
        Source RA in degrees (J2000)
    dec : float
        Source Dec in degrees (J2000)
    trigdat_file : str
        Path to trigdat FITS file

    Returns
    -------
    list of (detector_name, angle_deg)
        All 14 detectors with their angles to the source, sorted by angle.
    """
    try:
        from astropy.io import fits
        from astropy.coordinates import SkyCoord
        import numpy as np
    except ImportError:
        logger.error("astropy is required for trigdat processing")
        raise

    # Read spacecraft quaternion from trigdat
    with fits.open(trigdat_file) as hdul:
        # The quaternion is in the TRIGRATE extension
        trigrate = hdul['TRIGRATE'].data
        quat = trigrate['SCATTITD'][0]  # Quaternion [q1, q2, q3, q4]

    # Convert source position to spacecraft frame using quaternion rotation
    source_j2000 = Vector.from_ra_dec(ra, dec)

    # Quaternion rotation: v' = q * v * q^(-1)
    # Following the convention in the Fermi GBM data
    q1, q2, q3, q4 = quat

    # Rotation matrix from quaternion
    r11 = q1**2 - q2**2 - q3**2 + q4**2
    r12 = 2 * (q1 * q2 + q3 * q4)
    r13 = 2 * (q1 * q3 - q2 * q4)
    r21 = 2 * (q1 * q2 - q3 * q4)
    r22 = -q1**2 + q2**2 - q3**2 + q4**2
    r23 = 2 * (q2 * q3 + q1 * q4)
    r31 = 2 * (q1 * q3 + q2 * q4)
    r32 = 2 * (q2 * q3 - q1 * q4)
    r33 = -q1**2 - q2**2 + q3**2 + q4**2

    # Transform source to spacecraft frame
    sc_x = r11 * source_j2000.x + r12 * source_j2000.y + r13 * source_j2000.z
    sc_y = r21 * source_j2000.x + r22 * source_j2000.y + r23 * source_j2000.z
    sc_z = r31 * source_j2000.x + r32 * source_j2000.y + r33 * source_j2000.z

    source_sc = Vector(sc_x, sc_y, sc_z)

    # Calculate angle to each detector
    angles = []
    for det_name, (az, zen) in DETECTOR_DIRECTIONS.items():
        det_vector = Vector.from_spherical(az, zen)
        angle = source_sc.angle_to(det_vector)
        angles.append((det_name, angle))

    # Sort by angle (closest first)
    angles.sort(key=lambda x: x[1])

    return angles


def get_detector_angles_approximate(
    ra: float,
    dec: float,
    sc_ra: float,
    sc_dec: float,
) -> List[Tuple[str, float]]:
    """
    Approximate detector angles using spacecraft pointing direction.

    This is a simplified calculation when the full quaternion is not available.
    It uses only the spacecraft Z-axis pointing (RA, Dec) and assumes the
    spacecraft X-axis points roughly to the sun.

    Parameters
    ----------
    ra : float
        Source RA in degrees
    dec : float
        Source Dec in degrees
    sc_ra : float
        Spacecraft Z-axis RA in degrees
    sc_dec : float
        Spacecraft Z-axis Dec in degrees

    Returns
    -------
    list of (detector_name, angle_deg)
        Sorted by angle (approximate).
    """
    # The angle from spacecraft Z-axis to the source
    z_angle = get_angular_distance(ra, dec, sc_ra, sc_dec)

    # For approximate calculation, we can estimate that:
    # - Detectors near the Z-axis (small zenith) see sources near sc pointing
    # - Detectors in the XY plane (zenith~90) see sources ~90 deg from sc pointing
    # This is very rough but useful for quick checks

    angles = []
    for det_name, (az, zen) in DETECTOR_DIRECTIONS.items():
        # Very approximate: angle ~ |z_angle - zen|
        # This ignores azimuthal dependence but gives a rough ordering
        approx_angle = abs(z_angle - zen)
        angles.append((det_name, approx_angle))

    angles.sort(key=lambda x: x[1])
    return angles


def select_gbm_detectors(
    ra: float,
    dec: float,
    trigdat_file: Optional[str] = None,
    ra_scx: Optional[float] = None,
    dec_scx: Optional[float] = None,
    ra_scz: Optional[float] = None,
    dec_scz: Optional[float] = None,
    sc_ra: Optional[float] = None,
    sc_dec: Optional[float] = None,
    max_angle: float = 60.0,
    n_nai: int = 3,
    n_bgo: int = 1,
) -> Dict[str, Any]:
    """
    Select the best GBM detectors for analysis.

    Tries methods in order of accuracy:
        1. trigdat quaternion (most precise)
        2. SCX/SCZ pointing directions (Vianello method, from GBMangles notebook)
        3. SC Z-axis only (rough approximation)

    Parameters
    ----------
    ra : float
        Source RA in degrees
    dec : float
        Source Dec in degrees
    trigdat_file : str, optional
        Path to trigdat FITS file (for quaternion-based calculation)
    ra_scx, dec_scx : float, optional
        Spacecraft X-axis RA/Dec (from trigdat header)
    ra_scz, dec_scz : float, optional
        Spacecraft Z-axis RA/Dec (from trigdat header)
    sc_ra : float, optional
        Spacecraft Z-axis RA (alias for ra_scz, for backward compat)
    sc_dec : float, optional
        Spacecraft Z-axis Dec (alias for dec_scz, for backward compat)
    max_angle : float
        Maximum acceptable angle in degrees (default: 60)
    n_nai : int
        Number of NaI detectors to select (default: 3)
    n_bgo : int
        Number of BGO detectors to select (default: 1)

    Returns
    -------
    dict
        {
            'nai_detectors': [(name, angle), ...],
            'bgo_detectors': [(name, angle), ...],
            'all_angles': [(name, angle), ...],
            'method': 'trigdat' | 'scx_scz' | 'approximate',
        }
    """
    # Method 1: trigdat quaternion
    if trigdat_file:
        try:
            all_angles = get_detector_angles_from_trigdat(ra, dec, trigdat_file)
            method = 'trigdat'
        except Exception as e:
            logger.warning(f"trigdat quaternion failed: {e}")
            # Try to read SCX/SCZ from trigdat as fallback
            try:
                from astropy.io import fits
                with fits.open(trigdat_file) as hdul:
                    ra_scx = ra_scx or hdul[0].header.get('RA_SCX')
                    dec_scx = dec_scx or hdul[0].header.get('DEC_SCX')
                    ra_scz = ra_scz or hdul[0].header.get('RA_SCZ')
                    dec_scz = dec_scz or hdul[0].header.get('DEC_SCZ')
            except Exception:
                pass
            all_angles = None
            method = None
    else:
        all_angles = None
        method = None

    # Method 2: SCX/SCZ pointing (Vianello's method — from GBMangles notebook)
    if all_angles is None and ra_scx is not None and dec_scx is not None and ra_scz is not None and dec_scz is not None:
        all_angles = get_all_detector_angles(
            ra_scx, dec_scx, ra_scz, dec_scz, ra, dec
        )
        method = 'scx_scz'

    # Method 3: Approximate using Z-axis only
    if all_angles is None:
        _sc_ra = sc_ra or ra_scz
        _sc_dec = sc_dec or dec_scz
        if _sc_ra is not None and _sc_dec is not None:
            all_angles = get_detector_angles_approximate(ra, dec, _sc_ra, _sc_dec)
            method = 'approximate'
        else:
            logger.error("Provide trigdat_file, SCX/SCZ pointing, or sc_ra/sc_dec")
            return {'nai_detectors': [], 'bgo_detectors': [], 'all_angles': [], 'method': 'failed'}

    # Split NaI and BGO
    nai_angles = [(name, angle) for name, angle in all_angles if name.startswith('n')]
    bgo_angles = [(name, angle) for name, angle in all_angles if name.startswith('b')]

    # Select best within max_angle
    selected_nai = [(n, a) for n, a in nai_angles if a <= max_angle][:n_nai]
    selected_bgo = [(n, a) for n, a in bgo_angles if a <= max_angle][:n_bgo]

    # If not enough within max_angle, take the closest anyway
    if len(selected_nai) < n_nai:
        selected_nai = nai_angles[:n_nai]
    if len(selected_bgo) < n_bgo:
        selected_bgo = bgo_angles[:n_bgo]

    logger.info(f"Selected NaI: {[n for n, a in selected_nai]}, "
                f"BGO: {[n for n, a in selected_bgo]} (method: {method})")

    return {
        'nai_detectors': selected_nai,
        'bgo_detectors': selected_bgo,
        'all_angles': all_angles,
        'method': method,
    }
