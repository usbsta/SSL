#!/usr/bin/env python3
"""
4-array helicopter trajectory via GCC-PHAT TDOA & EKF smoothing.

Assumes:
* 4 synchronised WAVs (6 channels each, 48 kHz, 16-bit).
* Array centres & mic geometries defined in Utilities.mic_geo.
* Post-processing synchronisation better than ±1 sample.

Author: OpenAI-o3
Created: 2025-06-23
"""

# ——— Std-lib
from pathlib import Path
from collections import deque
import csv, sys, wave
from typing import Dict, List, Tuple

# ——— Third-party
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from numpy.linalg import lstsq, inv, norm
from matplotlib.animation import FFMpegWriter

# ——— Project helpers
from Utilities.mic_geo import (
    mic_6_N_black_thin, mic_6_S_orange, mic_6_E_orange, mic_6_W_black
)
#from Utilities.functions import bandpass_biquad                   # a light BPF helper
from Utilities.functions import apply_bandpass_filter as bandpass_biquad


# ───────────────────────────── CONFIG ───────────────────────────────────
ROOT = Path("/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25")                     # <-- adjust
ARRAYS: Dict[str, dict] = {
    "N": {"wav": ROOT / "N.wav", "centre": np.array([322_955.1, 6_256_643.2, 0.0]),
          "mics": mic_6_N_black_thin()},
    "S": {"wav": ROOT / "S.wav", "centre": np.array([322_951.6, 6_256_580.0, 0.0]),
          "mics": mic_6_S_orange()},
    "E": {"wav": ROOT / "E.wav", "centre": np.array([322_980.8, 6_256_638.4, 0.0]),
          "mics": mic_6_E_orange()},
    "W": {"wav": ROOT / "W.wav", "centre": np.array([322_918.0, 6_256_605.4, 0.0]),
          "mics": mic_6_W_black()},
}

RATE          = 48_000           # Hz
BLOCK_LEN_S   = 0.100            # 100 ms
CHUNK         = int(RATE*BLOCK_LEN_S)
LOWCUT, HIGHCUT = 80.0, 350.0    # rotor band
C_SOUND       = 343.0            # m/s

START_S, END_S = 57.0, 70.0      # inside the WAV

CSV_OUT   = Path("tdoa_traj.csv")
VIDEO_OUT = Path("tdoa_traj.mp4")

VERBOSE, MAX_PRINT = True, 6     # console output for first N blocks
DRAW_RAYS = True                 # show 2-D dashed rays for debugging
# ────────────────────────────────────────────────────────────────────────

ARRAY_KEYS = tuple(ARRAYS)
IDX_PAIRS  = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]   # order of TDOAs τ₁₂,τ₁₃,…

# ── Tiny helpers ────────────────────────────────────────────────────────
def gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray,
             max_tau: float, fs: int, interp: int = 4) -> float:
    """Return TDOA in seconds via GCC-PHAT, parabolic interp at peak."""
    n = sig_a.size + sig_b.size
    SIG = np.fft.rfft(sig_a, n=n) * np.conj(np.fft.rfft(sig_b, n=n))
    X = np.fft.irfft(SIG / (np.abs(SIG)+1e-12), n=n*interp)
    max_shift = int(interp*fs*max_tau)
    mid = X.size//2
    corr = X[mid-max_shift:mid+max_shift]
    k = np.argmax(corr)
    if 0 < k < corr.size-1:
        # parabolic fit
        α, β, γ = corr[k-1:k+2]
        k_offset = 0.5*(α-γ)/(α - 2*β + γ)
    else:
        k_offset = 0.0
    tau = (k + k_offset - max_shift) / (interp*fs)
    return tau

def multilaterate(centres: np.ndarray, tdoa: np.ndarray,
                  c: float = 343.0) -> np.ndarray:
    """Hyperbolic LS (Chan & Ho 1994, closed-form stage-1)."""
    p1 = centres[0]
    P  = centres[1:] - p1
    τ  = tdoa / c                           # convert to metres/c
    b  = 0.5*(np.sum(P**2, axis=1) - (c*τ)**2)[:,None]
    A  = -P
    x  = lstsq(A, b, rcond=None)[0].ravel()
    return x + p1

# ── EKF class (constant velocity, cartesian) ────────────────────────────
class EKF:
    def __init__(self, dt: float):
        self.dt = dt
        self.x  = None                    # 6-state
        self.P  = None
        # process & meas covariances
        self.Q = np.diag([0,0,0, 1,1,1])*0.5     # m² / (m/s)²
        self.R = np.eye(3)*25.0                  # m²

    def F(self) -> np.ndarray:
        dt = self.dt
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        return F

    def H(self) -> np.ndarray:
        H = np.zeros((3,6)); H[:3,:3] = np.eye(3)
        return H

    def predict(self):
        F = self.F()
        self.x = F@self.x
        self.P = F@self.P@F.T + self.Q

    def update(self, z: np.ndarray):
        H = self.H()
        y = z - H@self.x
        S = H@self.P@H.T + self.R
        K = self.P@H.T@inv(S)
        self.x += K@y
        self.P = (np.eye(6) - K@H)@self.P

    def step(self, z: np.ndarray):
        if self.x is None:
            self.x = np.hstack([z, 0,0,0])
            self.P = np.eye(6)*100.0
        else:
            self.predict(); self.update(z)
        return self.x[:3]

# ── Basemap figure (same style as before) ───────────────────────────────
_trU2W = Transformer.from_crs(32756,4326,always_xy=True)
_trW2M = Transformer.from_crs(4326,3857,always_xy=True)
utm2merc = lambda e,n: np.array(_trW2M.transform(*_trU2W.transform(e,n)))

P_merc = {k: utm2merc(*ARRAYS[k]['centre'][:2]) for k in ARRAY_KEYS}
P0     = P_merc['N']
margin = 750.0
centre_xy = np.vstack(list(P_merc.values())).mean(axis=0)
xminM,yminM = centre_xy-margin; xmaxM,ymaxM = centre_xy+margin
img,ext = ctx.bounds2img(xminM,yminM,xmaxM,ymaxM,zoom=19,source=ctx.providers.Esri.WorldImagery)
xminL,xmaxL = ext[0]-P0[0], ext[1]-P0[0]
yminL,ymaxL = ext[2]-P0[1], ext[3]-P0[1]

fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(img, extent=(xminL,xmaxL,yminL,ymaxL), origin='upper', zorder=0)
for k in ARRAY_KEYS:
    xy = P_merc[k]-P0; ax.scatter(*xy, marker='^', s=80, edgecolor='k'); ax.text(*(xy+[3,3]),f' {k}',color='white')
traj_line, = ax.plot([],[],'o-',color='red',lw=2,label='EKF')
raw_line,  = ax.plot([],[],'o-',color='yellow',lw=1,label='raw',alpha=.5)
ax.set_xlim(-margin, margin); ax.set_ylim(-margin, margin); ax.legend(loc='upper left'); ax.set_aspect('equal')
err_txt = ax.text(0.97,0.02,'',transform=ax.transAxes,ha='right',va='bottom',color='white',backgroundcolor='black')

# ── Runtime containers ─────────────────────────────────────────────────
traj_raw: List[np.ndarray] = []
traj_ekf: List[np.ndarray] = []

ekf = EKF(BLOCK_LEN_S)

# ── WAV IO ———————————————————————————————————————————————————————
wf = {k: wave.open(ARRAYS[k]['wav'].open('rb'),'rb') for k in ARRAY_KEYS}
for k in ARRAY_KEYS:
    if wf[k].getnchannels()!=6 or wf[k].getsampwidth()!=2 or wf[k].getframerate()!=RATE:
        sys.exit(f"WAV params mismatch on {k}")

start_f,end_f = int(START_S*RATE), int(END_S*RATE)
for k in ARRAY_KEYS: wf[k].setpos(start_f)
n_blocks = (end_f-start_f)//CHUNK

# Precompute broadside weights (simple average)
w_broad = np.ones(6)/6

# ── CSV + MP4 writer ————————————————————————————————————————————
with open(CSV_OUT,'w',newline='') as fcsv, \
     FFMpegWriter(fps=int(1/BLOCK_LEN_S), bitrate=2400).saving(fig, VIDEO_OUT, dpi=150):

    wr = csv.writer(fcsv)
    wr.writerow(['blk','time_s', *[f'tau_{a}{b}_s' for a,b in IDX_PAIRS], 'x_m','y_m','z_m',
                 'x_ekf','y_ekf','z_ekf'])

    plt.show(block=False)

    for blk in range(n_blocks):
        # ----- read + collapse to 1-ch per array -----------------------
        ref = {}
        for k in ARRAY_KEYS:
            data = wf[k].readframes(CHUNK)
            if len(data) < CHUNK*6*2: break
            sig = np.frombuffer(data,np.int16).reshape(-1,6).astype(np.float32)
            sig = bandpass_biquad(sig, LOWCUT, HIGHCUT, RATE, order=4)
            ref[k] = sig @ w_broad        # (samples,)

        # ----- GCC-PHAT pairwise --------------------------------------
        tdoa = []
        max_tau = 0.01                   # ±10 ms lobe search (≈ 3.4 km)
        keys = list(ARRAY_KEYS)
        for i,j in IDX_PAIRS:
            tau = gcc_phat(ref[keys[i]], ref[keys[j]], max_tau, RATE, interp=4)
            tdoa.append(tau)
        tdoa = np.array(tdoa)

        # ----- multilateration ----------------------------------------
        centres = np.vstack([ARRAYS[k]['centre'] for k in ARRAY_KEYS])
        X_raw = multilaterate(centres, tdoa[:3], c=C_SOUND)
        traj_raw.append(X_raw)

        X_ekf = ekf.step(X_raw)
        traj_ekf.append(X_ekf)

        # ----- plot update --------------------------------------------
        local_raw = utm2merc(*X_raw[:2]) - P0
        local_ekf = utm2merc(*X_ekf[:2]) - P0
        raw_line.set_data(*zip(*[utm2merc(*r[:2])-P0 for r in traj_raw]))
        traj_line.set_data(*zip(*[utm2merc(*e[:2])-P0 for e in traj_ekf]))
        err_txt.set_text(f'RMS EKF pos err ≈ {norm(X_raw-X_ekf):.1f} m')
        fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)
        FFMpegWriter.grab_frame(fig)     # add to MP4

        # ----- CSV -----------------------------------------------------
        wr.writerow([blk, f'{START_S+blk*BLOCK_LEN_S:.3f}', *[f'{t:.6f}' for t in tdoa],
                     *X_raw, *X_ekf])

        if VERBOSE and (MAX_PRINT is None or blk < MAX_PRINT):
            print(f'[blk {blk:04}] t={START_S+blk*BLOCK_LEN_S:.1f}s  '
                  + ' '.join(f'τ{a}{b}={t*1e3:+6.1f} ms' for (a,b),t in zip(IDX_PAIRS,tdoa))
                  + f'  pos_raw=({X_raw[0]:.1f},{X_raw[1]:.1f},{X_raw[2]:.1f})')

# Close WAVs
for k in ARRAY_KEYS: wf[k].close()
print('✅ CSV →', CSV_OUT.resolve())
print('✅ MP4 →', VIDEO_OUT.resolve())
