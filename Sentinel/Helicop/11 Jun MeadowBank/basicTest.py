#!/usr/bin/env python3
"""
3-D visualisation of FOUR 6-mic arrays and the rays defined by manual
azimuth/elevation angles.  Requires only NumPy & Matplotlib.


"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3-D projection)

# ── USER INPUT ────────────────────────────────────────────────────────────
# Angles are ENU: 0° azimuth = North, +90° = East.  Elevation 0–90° up.
AZEL_DEG = {                           # azimuth [deg], elevation [deg]
    'N': ( 0.0, 35.3),
    'S': ( 1, 37),
    'E': ( -2.0, 36.3),
    'W': ( 3.0, 35.7),
}


AZEL_DEG = {                           # azimuth [deg], elevation [deg]
    'N': ( -102-40, 15),           # -142     -144
    'S': ( -138-5, 16),            # -141     -143
    'E': ( -146, 16),              # -144     -146
    'W': ( -118-30, 12),           # -139     -141
}

AZEL_DEG = {                           # azimuth [deg], elevation [deg]
    'N': ( -144, 35.3),
    'S': ( -143, 37),
    'E': ( -146, 36.3),
    'W': ( -141, 35.7),
}





RAY_LENGTH = 20.0   # m – visual length of each arrow
# ──────────────────────────────────────────────────────────────────────────

# ── Array centres (UTM zone 56 S, metres) ────────────────────────────────
P_N = np.array([322955.1, 6256643.2, 0.0])
P_S = np.array([322951.6, 6256580.0, 0.0])
P_E = np.array([322980.8, 6256638.4, 0.0])
P_W = np.array([322918.0, 6256605.4, 0.0])

# Local six-mic geometries (Utilities.functions must already contain them)
from Utilities.mic_geo import (
    mic_6_N_black_thin,
    mic_6_S_orange,
    mic_6_E_orange,
    mic_6_W_black,
)

ARRAYS = {
    'N': {'centre': P_N, 'mics': P_N + mic_6_N_black_thin()},
    'S': {'centre': P_S, 'mics': P_S + mic_6_S_orange()},
    'E': {'centre': P_E, 'mics': P_E + mic_6_E_orange()},
    'W': {'centre': P_W, 'mics': P_W + mic_6_W_black()},
}

# ── Helper functions ──────────────────────────────────────────────────────
def az_el_to_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation (ENU, deg) to a 3-D *unit* vector."""
    az, el = np.deg2rad([az_deg, el_deg])
    dx = np.cos(el) * np.sin(az)   # East
    dy = np.cos(el) * np.cos(az)   # North
    dz = np.sin(el)                # Up
    vec = np.array([dx, dy, dz])
    return vec / np.linalg.norm(vec)


def triangulate_multi_rays(P_list, d_list):
    """
    Least-squares intersection for ≥2 rays.

    Parameters
    ----------
    P_list : (M, 3) array of ray origins.
    d_list : (M, 3) array of *unit* direction vectors.

    Returns
    -------
    X̂        : np.ndarray(3,)
        Point that minimises Σ‖(I-dᵢdᵢᵀ)(X̂ − Pᵢ)‖².
    rms_err   : float
        Root-mean-square distance from X̂ to the rays.
    dist_list : list[float]
        Distances from each array centre to X̂.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for P, d in zip(P_list, d_list):
        A_i = np.eye(3) - np.outer(d, d)
        A += A_i
        b += A_i @ P
    X_hat = np.linalg.solve(A, b)

    # Diagnostics
    residuals = []
    for P, d in zip(P_list, d_list):
        proj = P + d * np.dot(X_hat - P, d)  # closest point on ray to X̂
        residuals.append(np.linalg.norm(X_hat - proj))
    rms_err = float(np.sqrt(np.mean(np.square(residuals))))
    dist_list = [float(np.linalg.norm(X_hat - P)) for P in P_list]
    return X_hat, rms_err, dist_list


# ── Compute unit vectors & intersection ───────────────────────────────────
origins   = []
directions = []
for key, cfg in ARRAYS.items():
    az, el = AZEL_DEG[key]
    d = az_el_to_unit_vector(az, el)
    directions.append(d)
    origins.append(cfg['centre'])
    ARRAYS[key]['d'] = d           # store for plotting convenience

X_est, rms_error, dists = triangulate_multi_rays(np.vstack(origins),
                                                 np.vstack(directions))

# ── 3-D Plot ─────────────────────────────────────────────────────────────
cmap = {'N': 'darkgreen', 'S': 'crimson',
        'E': 'orange',     'W': 'navy'}

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.set_proj_type('persp')

for idx, (key, cfg) in enumerate(ARRAYS.items()):
    # Microphones
    ax.scatter(*cfg['mics'].T, s=15, c=cmap[key], alpha=0.7,
               label=f'{key} microphones' if idx == 0 else None)
    # Array centres
    ax.scatter(*cfg['centre'], s=60, c=cmap[key], marker='P',
               label=f'{key} centre')
    # Visual rays (finite length)
    ax.quiver(*cfg['centre'],
              *(RAY_LENGTH * cfg['d']), color=cmap[key], linewidth=2)

# Estimated source
ax.scatter(*X_est, s=100, c='lime', marker='*', edgecolors='k',
           label='Estimated source $\\hat{X}$')

# Cosmetics
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_zlabel('Up [m]')
ax.set_title('3-D geometry of four arrays and user-defined rays')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, linewidth=0.3, alpha=0.5)

# Equal aspect ratio
lims   = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).flatten()
centre = lims.reshape(3, 2).mean(axis=1)
radius = (lims[1::2] - lims[::2]).max() / 2.0
ax.set_xlim3d([centre[0] - radius, centre[0] + radius])
ax.set_ylim3d([centre[1] - radius, centre[1] + radius])
ax.set_zlim3d([centre[2] - radius, centre[2] + radius])

plt.tight_layout()


# ── Console diagnostics ──────────────────────────────────────────────────
print(f'Estimated source @ {X_est} m')
print(f'RMS distance to rays      : {rms_error:.2f} m')
for key, dist in zip(ARRAYS.keys(), dists):
    print(f'Distance from {key} array : {dist:.2f} m')

plt.show()