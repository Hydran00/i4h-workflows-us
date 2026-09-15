"""Shared geometry for the fixed-orientation ultrasound scan (public quaternions: wxyz)."""
import numpy as np

from i4h_common.types import quat_mul

APPROACH = (0.0030, 0.0, 0.2000)
CONTACT = (0.0030, 0.0, 0.0663)
SWEEP = ((-0.0310, -0.06075, 0.0600), (-0.0650, -0.0715, 0.0537))
# Exactly the previous final sweep orientation: down * local-Z rotation of 30 degrees.
PROBE_SCAN_LOCAL_WXYZ = (0.0, float(np.cos(np.pi / 12)), -float(np.sin(np.pi / 12)), 0.0)
CONTACT_FORCE_THRESHOLD_N = 0.1
CONTACT_SETTLE_S = 0.1


def scan_orientation(phantom_quat_wxyz):
    """Compose the old final waypoint orientation with the fixed phantom pose."""
    phantom = np.asarray(phantom_quat_wxyz, dtype=np.float32)
    return quat_mul(phantom, np.broadcast_to(PROBE_SCAN_LOCAL_WXYZ, phantom.shape))
