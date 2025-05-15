#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.animation as anim
import scipy.signal as signal
import wave, sys

# ──────────────────────────────────────────────────────────────
# External helper functions  (replace with your own imports)
# ──────────────────────────────────────────────────────────────
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,      # (signal, low, high, fs, order)
)

# ──────────── constants & array layout ────────────
RATE   = 48_000
CHUNK  = int(0.1 * RATE)          # 100 ms
ORDER  = 5
c      = 343                      # speed of sound (m s-1)

az_range = np.arange(-180, 181, 4)   # −180 … 180 deg
el_range = np.arange(0,  91,  4)     #    0 …  90 deg

mic_pos  = microphone_positions_8_helicop()
CH       = mic_pos.shape[0]

# Pre-compute integer delays for every (az, el) pair
delays = np.empty((len(az_range), len(el_range), CH), dtype=np.int32)
for i, az in enumerate(az_range):
    for j, el in enumerate(el_range):
        delays[i, j] = calculate_delays_for_direction(mic_pos, az, el, RATE, c)

# ──────────── open WAV file ────────────
WAV = "heli_12052025.wav"
wf = wave.open(WAV, "rb")
if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) != (CH, 2, RATE):
    sys.exit("✖ WAV format does not match expected microphone layout.")

# ──────────── figure 1 – energy map ────────────
fig_map, ax_map = plt.subplots(figsize=(12, 3))
img = ax_map.imshow(np.zeros((len(az_range), len(el_range))).T,
                    extent=[az_range[0], az_range[-1],
                            el_range[0],  el_range[-1]],
                    origin="lower", aspect="auto", cmap="jet")
fig_map.colorbar(img, ax=ax_map, label="Energy")
marker, = ax_map.plot([], [], "ro", ms=5)
ax_map.set(xlabel="Azimuth (deg)", ylabel="Elevation (deg)",
           title="Beamforming Energy Map")
ax_map.grid(True)

# ──────────── figure 2 – Bode plot + sliders ────────────
fig_bode = plt.figure("Filter", figsize=(7, 4))
ax_mag   = fig_bode.add_axes([0.1, 0.35, 0.85, 0.6])
ax_mag.set(ylabel="Magnitude (dB)", ylim=(-80, 5),
           title="Band-pass Bode (Butterworth, order 5)")

ax_low  = fig_bode.add_axes([0.1, 0.22, 0.8, 0.05])
ax_high = fig_bode.add_axes([0.1, 0.13, 0.8, 0.05])

slider_low  = widgets.Slider(ax_low,  "LOWCUT Hz",  20, 2000,
                             valinit=200,  valstep=10, color="tab:blue")
slider_high = widgets.Slider(ax_high, "HIGHCUT Hz", 500, 8000,
                             valinit=3000, valstep=10, color="tab:red")

line_bode, = ax_mag.plot([], [], lw=2)

def update_bode(low, high):
    """Recalculate and draw the filter magnitude response."""
    w, h = signal.freqz(
        *signal.butter(ORDER, [low, high], fs=RATE, btype="band"),
        fs=RATE, worN=1024
    )
    line_bode.set_data(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
    ax_mag.set_xlim(w[0], w[-1])
    fig_bode.canvas.draw_idle()

update_bode(slider_low.val, slider_high.val)  # initial plot

# ──────────── animation callback ────────────
def update(frame):
    """Read 100 ms of audio, apply current filter, update heat-map."""
    frames = wf.readframes(CHUNK)
    if len(frames) < CHUNK * CH * 2:   # end of file → stop animation
        ani.event_source.stop()
        wf.close()
        print("✔ processing finished")
        return img, marker

    audio = np.frombuffer(frames, np.int16).reshape(-1, CH)

    low, high = slider_low.val, slider_high.val
    if high <= low + 50:           # enforce 50 Hz separation
        high = low + 50
        slider_high.set_val(high)

    audio = apply_bandpass_filter(audio, low, high, RATE, order=ORDER)

    energy = np.zeros((len(az_range), len(el_range)))
    for i in range(len(az_range)):
        for j in range(len(el_range)):
            sig = apply_beamforming(audio, delays[i, j])
            energy[i, j] = np.sum(sig * sig)

    idx = np.unravel_index(np.argmax(energy), energy.shape)
    img.set_data(energy.T)
    img.set_clim(energy.min(), energy.max())
    marker.set_data([az_range[idx[0]]], [el_range[idx[1]]])

    return img, marker

# ──────────── slider callbacks ────────────
def on_slider(_):
    update_bode(slider_low.val, slider_high.val)

slider_low.on_changed(on_slider)
slider_high.on_changed(on_slider)

# ──────────── launch animation ────────────
ani = anim.FuncAnimation(fig_map, update, interval=100, blit=False)

print("▶ Move the sliders – the filter and Bode plot update instantly.")
plt.show()
