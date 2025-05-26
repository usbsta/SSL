import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from collections import deque
import wave, os, sys, time

from Utilities.functions import (
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# ───────────────────────── constants ────────────────────────────────
RATE, CHUNK      = 192_000, int(0.1 * 192_000)
LOWCUT, HIGHCUT  = 20_000.0, 70_000.0
FILTER_ORDER     = 5
SPEED            = 343.0

DETECT_F, BW     = 25_000.0, 1_000.0
THRESH_DB        = -25.0

AZ               = np.arange(-180, 181, 2)   # −180 … +180 in 2° steps
EL               = np.arange(0, 90, 2)       # 0 …  90 in 2° steps

MIC_POS = [
    (0.000, 0.000, 0.02), (0.000, 0.010, 0.02),
    (0.000,-0.015, 0.00), (0.000, 0.025, 0.00),
    (0.005, 0.005, 0.02), (-0.005, 0.005, 0.02),
    (0.020, 0.005, 0.00), (-0.020, 0.005, 0.00),
]
CHANNELS         = len(MIC_POS)

RECT_BINS_AZ     = 6          # half-width in bins  (6×2° = 12° each side)
ROI_H_DEG        = 20.0       # total ROI height
ROI_COLOR        = "magenta"

WIN_FRAMES       = 3          # temporal smoothing
SIGMA            = 1.0        # spatial smoothing (gaussian σ)

# NEW: playback control ---------------------------------------------------------
START_SEC        = 384.0       # seconds to skip at beginning
VIEW_WIN_SEC     = 60.0       # width of sliding window in leak-tone plot
# ------------------------------------------------------------------------------

# ────────────── pre-compute delay tables for all look-angles ───────────────────
print("Pre-computing delays …")
delays = np.empty((len(AZ), len(EL), CHANNELS), np.int32)
for i, az in enumerate(AZ):
    for j, el in enumerate(EL):
        delays[i, j] = calculate_delays_for_direction(
            MIC_POS, az, el, RATE, SPEED
        )

# ───────────────────────── helper functions ────────────────────────────────────
try:
    from scipy.ndimage import gaussian_filter
    smooth_spatial = lambda m: gaussian_filter(m, sigma=SIGMA, mode="nearest")
except ImportError:  # tiny 3×3 blur fallback
    def smooth_spatial(m):
        k = np.array([[1,2,1],[2,4,2],[1,2,1]], np.float32) / 16
        p = np.pad(m, 1, "edge"); out = np.zeros_like(m)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i,j] = np.sum(p[i:i+3, j:j+3] * k)
        return out

def leak_rms(chunk):
    """Energy (RMS) inside the leak-tone band for a multichannel chunk."""
    win = np.hanning(chunk.shape[0])
    X   = np.fft.rfft(chunk.mean(1) * win)
    f   = np.fft.rfftfreq(chunk.shape[0], 1 / RATE)
    msk = (f >= DETECT_F - BW/2) & (f <= DETECT_F + BW/2)
    return np.sqrt(np.mean(np.abs(X[msk]) ** 2))

# ──────────────────────────── main routine ─────────────────────────────────────
def process(wav_path: str, video_out: str, live: bool = True) -> None:

    # open & sanity-check WAV ----------------------------------------------------
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    wf = wave.open(wav_path, "rb")
    if (wf.getnchannels() != CHANNELS or wf.getsampwidth() != 2
            or wf.getframerate() != RATE):
        raise ValueError("WAV format mismatch.")

    # jump to START_SEC ---------------------------------------------------------
    start_frame = int(START_SEC * RATE)
    if start_frame >= wf.getnframes():
        raise ValueError("START_SEC exceeds file length.")
    wf.setpos(start_frame)

    steps = (wf.getnframes() - start_frame) // CHUNK
    dt    = CHUNK / RATE

    plt.ion() if live else plt.ioff()
    fig, (ax_roi, ax_map, ax_leak) = plt.subplots(
        3, 1, figsize=(12, 8),
        gridspec_kw=dict(height_ratios=[3, 3, 1])
    )

    # ROI panel ---------------------------------------------------------------
    ax_roi.set(xlim=(-180, 180), ylim=(-90, 90),
               xlabel="Azimuth [deg]", ylabel="Elevation [deg]")
    ax_roi.grid(False)
    ax_roi.invert_xaxis()  # azimuth increases to the right
    rect_roi = Rectangle((0, 0), 0, 0, lw=2, ec=ROI_COLOR,
                         fc="none", visible=False)
    ax_roi.add_patch(rect_roi)

    leak_txt = ax_roi.text(0.02, 0.92, "LEAK DETECTED",
                           transform=ax_roi.transAxes,
                           color="red", fontsize=24, weight="bold",
                           ha="left", va="top", visible=False)

    # energy map panel --------------------------------------------------------
    hm = ax_map.imshow(np.zeros((len(AZ), len(EL))).T,
                       extent=[AZ[-1], AZ[0], EL[0], EL[-1]],
                       origin="lower", cmap="inferno", aspect="auto",
                       interpolation="bilinear", visible=False)
    pk, = ax_map.plot([], [], "ro", visible=False)
    ax_map.set(xlabel="Azimuth [deg]", ylabel="Elevation [deg]")
    time_txt = ax_map.text(0.02, 0.95, "", transform=ax_map.transAxes,
                           color="w", fontsize=9,
                           bbox=dict(fc="k", alpha=.4, pad=2, ec="none"))

    # leak-tone panel (sliding window) ---------------------------------------
    leak_line, = ax_leak.plot([], [], lw=1.5,
                              label=f"|X| @ {DETECT_F/1e3:.1f} kHz")
    ax_leak.axhline(THRESH_DB, color="r", ls="--", lw=.8,
                    label="Threshold")
    ax_leak.set_xlim(0, VIEW_WIN_SEC)
    ax_leak.set_ylim(THRESH_DB - 10, THRESH_DB + 20)
    ax_leak.set_xlabel("Time [s]")
    ax_leak.set_ylabel("Magnitude [dB]")
    ax_leak.legend()
    ax_leak.grid(True)

    # video writer ------------------------------------------------------------
    writer = animation.FFMpegWriter(fps=RATE / CHUNK, bitrate=3200)

    hist      = deque(maxlen=WIN_FRAMES)  # spatial histories
    t_buf     = []                        # time axis for leak plot
    dB_buf    = []                        # magnitude history

    with writer.saving(fig, video_out, dpi=150):
        for n in range(steps):

            raw = wf.readframes(CHUNK)
            if len(raw) < CHUNK * CHANNELS * 2:
                break
            x = (np.frombuffer(raw, np.int16)
                   .reshape(-1, CHANNELS)
                   .astype(np.float32) / 32768.0)

            # leak-band magnitude -------------------------------------------
            dB = 20 * np.log10(leak_rms(x) + 1e-12)
            t  = n * dt
            t_buf.append(t)
            dB_buf.append(dB)
            leak_line.set_data(t_buf, dB_buf)

            # sliding x-axis in leak plot -----------------------------------
            if t > VIEW_WIN_SEC:
                ax_leak.set_xlim(t - VIEW_WIN_SEC, t)

            # dynamic y-axis
            y_lo, y_hi = ax_leak.get_ylim()
            if dB > y_hi - 2:
                ax_leak.set_ylim(y_lo, dB + 10)
            if dB < y_lo + 2:
                ax_leak.set_ylim(dB - 10, y_hi)

            # beamforming branch --------------------------------------------
            if dB >= THRESH_DB:
                y_f = apply_bandpass_filter(
                    x, LOWCUT, HIGHCUT, RATE, FILTER_ORDER
                )

                E = np.zeros((len(AZ), len(EL)), np.float32)
                for i in range(len(AZ)):
                    for j in range(len(EL)):
                        E[i, j] = np.sum(
                            apply_beamforming(y_f, delays[i, j]) ** 2
                        )

                hist.append(E)
                S = smooth_spatial(np.mean(hist, 0))

                hm.set_data(np.flip(S, 0).T)
                hm.set_clim(np.percentile(S, 50), S.max())
                hm.set_visible(True)

                idx = np.unravel_index(np.argmax(S), S.shape)
                az_pk, el_pk = AZ[idx[0]], EL[idx[1]]
                pk.set_data([az_pk], [el_pk])
                pk.set_visible(True)

                half_w = RECT_BINS_AZ * 2
                half_h = ROI_H_DEG / 2
                x_left = az_pk - half_w
                y0, y1 = np.clip([el_pk - half_h, el_pk + half_h], -90, 90)
                rect_roi.set_xy((x_left, y0))
                rect_roi.set_width(2 * half_w)
                rect_roi.set_height(y1 - y0)
                rect_roi.set_visible(True)
                leak_txt.set_visible(True)
            else:
                hm.set_visible(False); pk.set_visible(False)
                rect_roi.set_visible(False); leak_txt.set_visible(False)

            time_txt.set_text(f"t = {t + START_SEC:.2f} s")  # absolute time
            writer.grab_frame()

            if live:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(.001)

            sys.stdout.write(f"\rProgress: {(n + 1) / steps * 100:6.2f}%")
            sys.stdout.flush()

    wf.close()
    plt.close(fig)
    print("\nDone.")

# ──────────────────────────────── run ──────────────────────────────────────────
if __name__ == "__main__":
    WAV = (r"C:\Users\30068385\OneDrive - Western Sydney University\recordings"
           r"\GasLeake\robotF2.wav")
    OUT = "robotF2_Final.mp4"
    process(WAV, OUT, live=True)
