"""
3-D visualisation of two 8-mic arrays and the rays defined by manual
azimuth / elevation angles.  Requires only NumPy & Matplotlib.

Author : ChatGPT (OpenAI-o3) – 2025-06-05
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

# ── USER INPUT ───────────────────────────────────────────────────────────
# Angles are ENU: 0° azimuth = North, +90° = East.  Elevation 0–90° up.
AZ_LEFT,  EL_LEFT  = 45.0,  90.0    # deg – ray leaving the LEFT array
AZ_RIGHT, EL_RIGHT = -45.0, 90.0    # deg – ray leaving the RIGHT array
RAY_LENGTH = 20000.0                  # m – visual length of each arrow

AZ_LEFT,  EL_LEFT  = -90.0,  68.0    # deg – ray leaving the LEFT array
AZ_RIGHT, EL_RIGHT = -98.0, 68.0    # deg – ray leaving the RIGHT array
RAY_LENGTH = 20000.0                  # m – visual length of each arrow
# ─────────────────────────────────────────────────────────────────────────

# ── Array centres (UTM 56 S, metres) ─────────────────────────────────────
P_N = np.array([322955.1, 6256643.2, 0.0])
P_S  = np.array([322951.6, 6256580.0, 0.0])
P_E  = np.array([322980.8, 6256638.4, 0.0])
P_W  = np.array([322918.0, 6256605.4, 0.0])

# Your local 8-mic geometry helper
from Utilities.functions import microphone_positions_8_helicop
MIC_LOCAL  = microphone_positions_8_helicop()          # shape (8, 3)
MICS_LEFT  = P_LEFT  + MIC_LOCAL
MICS_RIGHT = P_RIGHT + MIC_LOCAL

# ── Helper functions ─────────────────────────────────────────────────────
def az_el_to_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation (ENU, deg) to a 3-D *unit* vector."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    dx = np.cos(el) * np.sin(az)     # East  component
    dy = np.cos(el) * np.cos(az)     # North component
    dz = np.sin(el)                  # Up    component
    vec = np.array([dx, dy, dz])
    return vec / np.linalg.norm(vec)  # ensure unit length


def triangulate_two_rays(P1, d1, P2, d2):
    """
    Smallest-distance mid-point between the *infinite* rays

        L1 : P1 + λ d1 ,  λ∈ℝ
        L2 : P2 + μ d2 ,  μ∈ℝ

    Returns
    -------
    X      : np.ndarray | None
        Mid-point of the shortest segment, or None if rays parallel/diverging.
    err    : float
        Half of the shortest distance ∥Q1 − Q2∥/2  (∞ if invalid).
    dist_L : float
        Distance from P1 to X.
    dist_R : float
        Distance from P2 to X.
    """
    c    = float(np.dot(d1, d2))     # cosine of angle between the two rays
    dP   = P2 - P1
    denom = 1.0 - c**2
    if abs(denom) < 1e-9:
        return None, np.inf, np.inf, np.inf  # rays (almost) parallel

    # λ, μ that minimise ∥(P1+λd1) − (P2+μd2)∥²
    lam = ( np.dot(dP, d1) - c * np.dot(dP, d2) ) / denom
    mu  = (-np.dot(dP, d2) + c * np.dot(dP, d1) ) / denom  # sign-corrected

    # Guard: only accept *forward* intersections
    if lam < 0.0 or mu < 0.0:
        return None, np.inf, np.inf, np.inf  # rays diverge in front

    Q1 = P1 + lam * d1
    Q2 = P2 + mu  * d2
    X  = 0.5 * (Q1 + Q2)
    err = 0.5 * np.linalg.norm(Q1 - Q2)
    return X, err, np.linalg.norm(X - P1), np.linalg.norm(X - P2)


# ── Compute unit vectors & intersection ──────────────────────────────────
d_left  = az_el_to_unit_vector(AZ_LEFT,  EL_LEFT)
d_right = az_el_to_unit_vector(AZ_RIGHT, EL_RIGHT)
X_est, uncert, dist_L, dist_R = triangulate_two_rays(P_LEFT, d_left,
                                                     P_RIGHT, d_right)

# ── 3-D Plot ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.set_proj_type('persp')  # nicer perspective view

# Microphone positions
ax.scatter(*MICS_LEFT.T,  s=15, c='royalblue', label='Left microphones')
ax.scatter(*MICS_RIGHT.T, s=15, c='firebrick',  label='Right microphones')

# Array centres
ax.scatter(*P_LEFT,  s=60, c='navy',    marker='P', label='Left centre')
ax.scatter(*P_RIGHT, s=60, c='darkred', marker='P', label='Right centre')

# Visual rays (finite length)
ax.quiver(*P_LEFT,  *(RAY_LENGTH * d_left),  color='blue', linewidth=2)
ax.quiver(*P_RIGHT, *(RAY_LENGTH * d_right), color='red',  linewidth=2)

# Draw estimated source (if intersection is valid)
if X_est is not None and np.isfinite(uncert):
    ax.scatter(*X_est, s=80, c='limegreen', marker='o', label='Estimated source')

    # Segment depicting uncertainty
    ax.plot([X_est[0] - uncert * d_left[0],  X_est[0] + uncert * d_left[0]],
            [X_est[1] - uncert * d_left[1],  X_est[1] + uncert * d_left[1]],
            [X_est[2] - uncert * d_left[2],  X_est[2] + uncert * d_left[2]],
            color='grey', linewidth=1.2, alpha=0.6)
    ax.plot([X_est[0] - uncert * d_right[0], X_est[0] + uncert * d_right[0]],
            [X_est[1] - uncert * d_right[1], X_est[1] + uncert * d_right[1]],
            [X_est[2] - uncert * d_right[2], X_est[2] + uncert * d_right[2]],
            color='grey', linewidth=1.2, alpha=0.6)

    print(f"Estimated source @ {X_est} m")
    print(f"Uncertainty (½ min distance between rays): {uncert:.2f} m")
    print(f"Distance from left array : {dist_L:.2f} m")
    print(f"Distance from right array: {dist_R:.2f} m")

else:
    print("⚠️  Rays diverge – no forward intersection.")

# Cosmetics
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_zlabel('Up [m]')
ax.set_title('3-D geometry of arrays and user-defined rays')
ax.legend(loc='upper left')
ax.grid(True, linewidth=0.3, alpha=0.5)

# Force equal aspect ratio
lims   = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).flatten()
centre = lims.reshape(3, 2).mean(axis=1)
radius = (lims[1::2] - lims[::2]).max() / 2.0
ax.set_xlim3d([centre[0] - radius, centre[0] + radius])
ax.set_ylim3d([centre[1] - radius, centre[1] + radius])
ax.set_zlim3d([centre[2] - radius, centre[2] + radius])

plt.tight_layout()
plt.show()
