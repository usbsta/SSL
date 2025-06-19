#!/usr/bin/env python3
"""
Online localisation of a sound source seen by four microphone arrays.

For every 100 ms record contained in `triangulation_debug_4arrays.csv`
the script prints:

    t [s] |  x_est, y_est, z_est [m] | mean_range [m] | rms_error [m]

The algorithm is a closed-form least-squares fit of a point P to four
3-D rays C_i + t_i d_i (array centres C_i, unit directions d_i).

Author : OpenAI-o3 (ChatGPT)
Date   : 2025-06-20
"""

from __future__ import annotations
import pathlib
import time
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────
CSV_PATH   = pathlib.Path("triangulation_debug_4arrays.csv")
DT_SECONDS = 0.1                     # 100 ms → 10 Hz print-out rate
REAL_TIME  = False                   # True → pace with time.sleep(DT_SECONDS)

ARRAY_CENTRES = {                    # ENU metres (UTM56S)
    "N": np.array([322955.1, 6256643.2, 0.0]),
    "S": np.array([322951.6, 6256580.0, 0.0]),
    "E": np.array([322980.8, 6256638.4, 0.0]),
    "W": np.array([322918.0, 6256605.4, 0.0]),
}

# ── Helpers ──────────────────────────────────────────────────────────────
def az_el_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    """
    Convert ENU azimuth/elevation (deg) to a 3-D unit vector.
    az=0° → North, +90° → East; el=0° horizontal, +90° up.
    """
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array([
        np.cos(el) * np.sin(az),     # East
        np.cos(el) * np.cos(az),     # North
        np.sin(el)                  # Up
    ])

def least_squares_intersection(origins: list[np.ndarray],
                               dirs:    list[np.ndarray]) -> tuple[np.ndarray, float]:
    """
    Closed-form least-squares point P that minimises the sum of squared
    orthogonal distances to K rays.

    Returns
    -------
    P : np.ndarray shape (3,)
        Estimated source position in ENU metres.
    rms_residual : float
        Root-mean-square orthogonal distance to the rays (error proxy).
    """
    I = np.eye(3)
    M = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in zip(origins, dirs):
        A = I - np.outer(d, d)       # projection matrix onto plane ⟂ d
        M += A
        b += A @ C
    P = np.linalg.solve(M, b)

    # Residuals
    resid2 = [np.linalg.norm((P - C) - ((P - C) @ d) * d)**2
              for C, d in zip(origins, dirs)]
    rms_err = np.sqrt(np.mean(resid2))
    return P, rms_err

# ── Main ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

start_t_wall = time.perf_counter()

for i, row in df.iterrows():
    # Directions (use SMOOTH columns)
    dirs = [az_el_to_unit(row[f"smooth_az_{k}"], row[f"smooth_el_{k}"])
            for k in ARRAY_CENTRES]

    # Estimate source position + residual
    P_est, rms_err = least_squares_intersection(
        list(ARRAY_CENTRES.values()), dirs)

    # Mean range to arrays
    ranges = [np.linalg.norm(P_est - C) for C in ARRAY_CENTRES.values()]
    mean_range = float(np.mean(ranges))

    # Print
    t_sec = i * DT_SECONDS
    print(f"{t_sec:6.1f} s | "
          f"x={P_est[0]:9.2f} m, y={P_est[1]:9.2f} m, z={P_est[2]:6.2f} m | "
          f"rangē={mean_range:7.2f} m | ε_rms={rms_err:5.2f} m")

    # Optional real-time pacing
    if REAL_TIME:
        elapsed = time.perf_counter() - start_t_wall
        sleep_t = (i + 1) * DT_SECONDS - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
