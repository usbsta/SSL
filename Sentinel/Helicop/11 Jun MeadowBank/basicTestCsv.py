#!/usr/bin/env python3
"""
Interactive 3-D debugger with multi-view plots for four-array helicopter triangulation.

Reads the CSV created by 'triangulation_debug_4arrays*.py' and plots for any
selected block the microphone arrays, their acoustic rays, and the least-squares
intersection X̂ from three different perspectives: top, front and side.

"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D proj.

# Geometry: centres in UTM 56 S (E, N, Up) [m] GPS Drone
#P_N = np.array([322962.476, 6256637.765, 0.0])
#P_S = np.array([322953.976, 6256584.241, 0.0])
#P_E = np.array([322969.849, 6256616.766, 0.0])
#P_W = np.array([322921.481, 6256604.606, 0.0])



# Geometry: centres in UTM 56 S (E, N, Up) [m] GPS APP Iphone
P_N = np.array([322955.1, 6256643.2, 0.0])
P_S = np.array([322951.6, 6256580.0, 0.0])
P_E = np.array([322980.8, 6256638.4, 0.0])
P_W = np.array([322918.0, 6256605.4, 0.0])
#ARRAY_CENTRES = dict(N=P_N, S=P_S, E=P_E, W=P_W)

ARRAY_CENTRES = dict(N=P_N, S=P_S, E=P_E, W=P_W)

# Colour map
C_MAP = dict(N="darkgreen", S="crimson", E="orange", W="navy")

AZ_OFFSET = {
    'N': -40.0,
    'E':  0.0,
    'W': -30.0,
    'S':  -5.0,
}

EL_OFFSET = {
    'N': 0.0,
    'E': 0.0,
    'W': 0.0,
    'S': 0.0,
}

def az_el_to_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad([az_deg, el_deg])
    d = np.array([np.cos(el) * np.sin(az),  # East
                  np.cos(el) * np.cos(az),  # North
                  np.sin(el)])              # Up
    return d / np.linalg.norm(d)

def triangulate_multi_rays(origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for P, d in zip(origins, dirs):
        A_i = np.eye(3) - np.outer(d, d)
        A += A_i
        b += A_i @ P
    X_hat = np.linalg.solve(A, b)
    residuals = [np.linalg.norm(X_hat - (P + d * np.dot(X_hat - P, d))) for P, d in zip(origins, dirs)]
    rms_err = float(np.sqrt(np.mean(np.square(residuals))))
    return X_hat, rms_err

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df

def make_multi_axes() -> tuple[plt.Figure, list[plt.Axes]]:
    fig = plt.figure(figsize=(18, 6))
    ax_top    = fig.add_subplot(131, projection="3d")
    ax_front  = fig.add_subplot(132, projection="3d")
    ax_side   = fig.add_subplot(133, projection="3d")
    axes = [ax_top, ax_front, ax_side]
    titles = ["Top view (Z down)", "Front view (Y front)", "Side view (X side)"]
    elev_azims = [(90, -90), (0, -180), (0, -90)]
    for ax, title, (elev, azim) in zip(axes, titles, elev_azims):
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_zlabel("Up [m]")
        ax.grid(True, linewidth=0.2, alpha=0.4)
    return fig, axes

def plot_block(row: pd.Series, axes: list[plt.Axes], ray_len: float = 20.0, clamp_negative_z: bool = True) -> None:
    origins, dirs = [], []
    for key in ("N", "S", "E", "W"):
        az = row[f"raw_az_{key}"] + AZ_OFFSET.get(key, 0.0)  # apply per-array offset
        el = row[f"raw_el_{key}"] + EL_OFFSET.get(key, 0.0)
        d = az_el_to_unit_vector(az, el)
        if d[2] < 0:
            d = -d
        origins.append(ARRAY_CENTRES[key])
        dirs.append(d)
    origins = np.vstack(origins)
    dirs    = np.vstack(dirs)
    X_hat, rms = triangulate_multi_rays(origins, dirs)
    if clamp_negative_z and X_hat[2] < 0:
        X_hat[2] = 0.0
    for ax in axes:
        ax.cla()
        ax.grid(True, linewidth=0.2, alpha=0.4)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_zlabel("Up [m]")
        for key, P, d in zip(("N", "S", "E", "W"), origins, dirs):
            ax.scatter(*P, s=60, marker="P", color=C_MAP[key])
            ax.quiver(*P, *(ray_len * d), linewidth=2, color=C_MAP[key])
        ax.scatter(*X_hat, s=120, marker="*", color="lime", edgecolors="k", label=f"X̂ (RMS={rms:.1f} m)")
        ax.legend(fontsize=8, loc="upper left")
        lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centre = lims.mean(axis=1)
        radius = (lims[:, 1] - lims[:, 0]).max() / 2
        ax.set_xlim3d(centre[0] - radius, centre[0] + radius)
        ax.set_ylim3d(centre[1] - radius, centre[1] + radius)
        ax.set_zlim3d(centre[2] - radius, centre[2] + radius)

def main() -> None:
    parser = argparse.ArgumentParser(description="CSV visual debugger for 4-array beamforming.")
    parser.add_argument("csv", type=Path, help="Path to triangulation_debug_4arrays*.csv")
    parser.add_argument("--block", type=int, default=None, help="Visualise a single block index")
    parser.add_argument("--animate", action="store_true", help="Animate all blocks (slow)")
    parser.add_argument("--save", metavar="OUT.mp4", help="Save animation (mp4/gif) instead of showing")
    parser.add_argument("--ray-len", type=float, default=20.0, help="Visual ray length in metres")
    parser.add_argument("--no-clamp", action="store_true", help="Allow negative-Z intersections")
    args = parser.parse_args()
    df = load_csv(args.csv)
    if args.block is not None and args.block not in df["block"].values:
        sys.exit(f"Block {args.block} not found in CSV.")
    fig, axes = make_multi_axes()
    if not args.animate:
        block_idx = args.block if args.block is not None else df["block"].iloc[0]
        row = df.loc[df["block"] == block_idx].iloc[0]
        plot_block(row, axes, ray_len=args.ray_len, clamp_negative_z=not args.no_clamp)
        plt.tight_layout()
        plt.show()
        return
    print("Animating…  Close the window to stop.")
    def init():
        for ax in axes:
            ax.cla()
    def update(frame):
        init()
        row = df.iloc[frame]
        plot_block(row, axes, ray_len=args.ray_len, clamp_negative_z=not args.no_clamp)
        fig.suptitle(f"Block {int(row['block'])}   t = {row['time_s']:.2f} s")
        return axes
    ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=False, interval=150, repeat=False)
    if args.save:
        print(f"Saving animation to '{args.save}' …")
        ani.save(args.save, fps=6, dpi=150)
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
