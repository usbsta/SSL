#!/usr/bin/env python3
"""
Four-array localisation (6 mics), live plotting + on-the-fly MP4 recording.
"""

# ── Imports ─────────────────────────────────────────────────────────────
from collections import deque
import csv, sys, wave, time
from pathlib import Path
from typing import Dict, List
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from pyproj import Transformer

from Utilities.functions import (
    calculate_delays_for_direction, apply_beamforming, apply_bandpass_filter
)
from Utilities.mic_geo import (
    mic_6_N_black_thin, mic_6_S_orange, mic_6_E_orange, mic_6_W_black
)

# ── Configuration (idéntica salvo cambios de animación) ─────────────────
ROOT = Path("/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25")

P_N = np.array([322_955.1, 6_256_643.2, 0.0])
P_S = np.array([322_951.6, 6_256_580.0, 0.0])
P_E = np.array([322_980.8, 6_256_638.4, 0.0])
P_W = np.array([322_918.0, 6_256_605.4, 0.0])

ARRAYS: Dict[str, dict] = {
    'N': {'wav': ROOT / "N.wav", 'centre': P_N, 'mics': mic_6_N_black_thin()},
    'S': {'wav': ROOT / "S.wav", 'centre': P_S, 'mics': mic_6_S_orange()},
    'E': {'wav': ROOT / "E.wav", 'centre': P_E, 'mics': mic_6_E_orange()},
    'W': {'wav': ROOT / "W.wav", 'centre': P_W, 'mics': mic_6_W_black()},
}

CSV_OUTPUT   = Path("triangulation_full_NoSmooth.csv")
VIDEO_OUTPUT = Path("triangulation_full_NoSmooth.mp4")

AZ_OFFSET = {'N': -44.0, 'S': -8.0, 'E': 0.0, 'W': -28.0}
EL_OFFSET = {'N': 0.0,  'S': 1.0,  'E': -1.0, 'W': 4.0}

RATE, CHUNK = 48_000, int(0.1 * 48_000)
LOWCUT, HIGHCUT, FILTER_ORDER = 180.0, 700.0, 5
C_SOUND = 343.0
AZIM_RANGE = np.arange(-180, 181, 1)
EL_RANGE   = np.arange(10,  51,  1)
SMOOTH_LEN = 1
START_TIME_S, END_TIME_S = 57, 70           # s dentro del WAV

VERBOSE, MAX_PRINT = True, 100
DRAW_RAYS, RAY_LEN = True, 600.0            # m

# ── Transformadores y mapa base ─────────────────────────────────────────
_trU2W = Transformer.from_crs(32756, 4326, always_xy=True)
_trW2M = Transformer.from_crs(4326, 3857, always_xy=True)
utm2merc = lambda e,n: np.array(_trW2M.transform(*_trU2W.transform(e,n)), dtype=np.float64)

ARRAY_KEYS = tuple(ARRAYS)
P_merc = {k: utm2merc(*ARRAYS[k]['centre'][:2]) for k in ARRAY_KEYS}
P0     = P_merc['N']

margin = 750.0
centre_xy = np.vstack(list(P_merc.values())).mean(axis=0)
xminM,yminM = centre_xy-margin
xmaxM,ymaxM = centre_xy+margin
img, ext = ctx.bounds2img(xminM,yminM,xmaxM,ymaxM, zoom=19, source=ctx.providers.Esri.WorldImagery)
xminL,xmaxL = ext[0]-P0[0], ext[1]-P0[0]
yminL,ymaxL = ext[2]-P0[1], ext[3]-P0[1]

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img, extent=(xminL,xmaxL,yminL,ymaxL), origin='upper', zorder=0)
for k in ARRAY_KEYS:
    xy = P_merc[k]-P0; ax.scatter(*xy, marker='^', edgecolor='k', s=80); ax.text(*(xy+[3,3]),f' {k}',color='white')
traj_line, = ax.plot([],[],'o-',color='lime',lw=2,label='X̂')
err_text = ax.text(0.97,0.02,'',transform=ax.transAxes,ha='right',va='bottom',color='white',backgroundcolor='black')
ax.set_xlim(-margin, margin); ax.set_ylim(-margin, margin); ax.set_aspect('equal')

# ── Delay lookup por arreglo ────────────────────────────────────────────
DELAY_LUT = {}
for k in ARRAY_KEYS:
    m = ARRAYS[k]['mics']; lut = np.empty((len(AZIM_RANGE), len(EL_RANGE), m.shape[0]), np.int32)
    for ia,az in enumerate(AZIM_RANGE):
        for ie,el in enumerate(EL_RANGE):
            lut[ia,ie] = calculate_delays_for_direction(m, az, el, RATE, C_SOUND)
    DELAY_LUT[k] = lut

smooth = {k:{'az':deque(maxlen=SMOOTH_LEN),'el':deque(maxlen=SMOOTH_LEN)} for k in ARRAY_KEYS}
traj_xy: List[np.ndarray] = []

def azel2unit(az,el):
    a,e=np.deg2rad([az,el]); return np.array([np.cos(e)*np.sin(a), np.cos(e)*np.cos(a), np.sin(e)])

def ls_triang(orig,dirs):
    A=b=np.zeros(3); M=np.zeros((3,3))
    for P,d in zip(orig,dirs):
        Ai=np.eye(3)-np.outer(d,d); M+=Ai; b+=Ai@P
    X=np.linalg.solve(M,b)
    rms=np.sqrt(np.mean([np.linalg.norm(np.cross(d,X-P))**2 for P,d in zip(orig,dirs)]))
    return X,rms

def process_block(i,wf,writer):
    filt={}
    for k in ARRAY_KEYS:
        nm = ARRAYS[k]['mics'].shape[0]
        data=wf[k].readframes(CHUNK)
        if len(data)<CHUNK*nm*2: return False
        s=np.frombuffer(data,np.int16).reshape(-1,nm)
        s=apply_bandpass_filter(s,LOWCUT,HIGHCUT,RATE,order=FILTER_ORDER)
        p=np.abs(s).max(); s/=p or 1; filt[k]=s

    raw_az,raw_el={},{}
    for k in ARRAY_KEYS:
        E=np.zeros((len(AZIM_RANGE),len(EL_RANGE)))
        for ia,az in enumerate(AZIM_RANGE):
            for ie,el in enumerate(EL_RANGE):
                y=apply_beamforming(filt[k],DELAY_LUT[k][ia,ie]); E[ia,ie]=np.sum(y**2)/CHUNK
        ia,ie=np.unravel_index(np.argmax(E),E.shape)
        raw_az[k],raw_el[k]=AZIM_RANGE[ia],EL_RANGE[ie]
        smooth[k]['az'].append(raw_az[k]); smooth[k]['el'].append(raw_el[k])

    dirs,orig=[],[]
    for k in ARRAY_KEYS:
        saz=np.mean(smooth[k]['az'])+AZ_OFFSET[k]
        sel=np.mean(smooth[k]['el'])+EL_OFFSET[k]
        dirs.append(azel2unit(saz,sel)); orig.append(ARRAYS[k]['centre'])
    dirs,orig=np.vstack(dirs),np.vstack(orig)

    X,rms=ls_triang(orig,dirs); X[2]=max(X[2],0)
    local=utm2merc(*X[:2])-P0; traj_xy.append(local)
    traj_line.set_data(*zip(*traj_xy)); err_text.set_text(f'RMS={rms:.1f} m')

    if DRAW_RAYS:
        if hasattr(process_block,'hs'):
            for h in process_block.hs: h.remove()
        hs=[]
        for d,P in zip(dirs,orig):
            end=utm2merc(*(P[:2]+RAY_LEN*d[:2]))-P0
            h,=ax.plot([local[0],end[0]],[local[1],end[1]],color='orange',lw=1,alpha=.6)
            hs.append(h)
        process_block.hs=hs

    writer.writerow([i,f'{START_TIME_S+i*CHUNK/RATE:.3f}',
                     *[f'{raw_az[k]:.1f}' for k in ARRAY_KEYS],
                     *[f'{raw_el[k]:.1f}' for k in ARRAY_KEYS],
                     *[f'{np.mean(smooth[k]["az"])+AZ_OFFSET[k]:.1f}' for k in ARRAY_KEYS],
                     *[f'{np.mean(smooth[k]["el"])+EL_OFFSET[k]:.1f}' for k in ARRAY_KEYS],
                     f'{local[0]:.3f}',f'{local[1]:.3f}',f'{rms:.3f}',
                     *[f'{np.linalg.norm(X-ARRAYS[k]["centre"]):.3f}' for k in ARRAY_KEYS]])

    if VERBOSE and (MAX_PRINT is None or i<MAX_PRINT):
        print(f'[blk {i:04d}] t={START_TIME_S+i*CHUNK/RATE:7.2f}s | '
              + ' | '.join(f'{k}:az={raw_az[k]:6.1f},el={raw_el[k]:5.1f}' for k in ARRAY_KEYS)
              + f' | RMS={rms:5.2f}')
    return True

def main():
    wf={k:wave.open(ARRAYS[k]['wav'].open('rb'),'rb') for k in ARRAY_KEYS}
    for k in ARRAY_KEYS:
        if wf[k].getnchannels()!=ARRAYS[k]['mics'].shape[0] or wf[k].getsampwidth()!=2 or wf[k].getframerate()!=RATE:
            sys.exit(f'❌ WAV params mismatch {k}')
    s_f,e_f=int(START_TIME_S*RATE),int(END_TIME_S*RATE)
    total=min(wf[k].getnframes() for k in ARRAY_KEYS)
    if not (0<=s_f<e_f<=total): sys.exit('❌ START/END out of range')
    for k in ARRAY_KEYS: wf[k].setpos(s_f)
    n_blocks=(e_f-s_f)//CHUNK

    # CSV + MP4 writer
    with open(CSV_OUTPUT,'w',newline='') as fcsv, \
         FFMpegWriter(fps=10,bitrate=2400).saving(fig, VIDEO_OUTPUT, dpi=150) as vid:
        wr=csv.writer(fcsv)
        wr.writerow(['block','time_s',
                     *[f'raw_az_{k}' for k in ARRAY_KEYS],
                     *[f'raw_el_{k}' for k in ARRAY_KEYS],
                     *[f'smooth_az_{k}' for k in ARRAY_KEYS],
                     *[f'smooth_el_{k}' for k in ARRAY_KEYS],
                     'local_x_m','local_y_m','rms_m',
                     *[f'dist_{k}_m' for k in ARRAY_KEYS]])
        plt.show(block=False)
        for i in range(n_blocks):
            ok=process_block(i,wf,wr)
            fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)
            vid.grab_frame()           # añade frame al MP4
            if not ok: break

    for k in ARRAY_KEYS: wf[k].close()
    print('✅ CSV →',CSV_OUTPUT.resolve())
    print('✅ MP4 →',VIDEO_OUTPUT.resolve())

if __name__=='__main__':
    main()
