
# -----------------------------------------------------------------------------#
#  Imports                                                                     #
# -----------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave, os, sys, time

from Utilities.functions import (
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# -----------------------------------------------------------------------------#
#  Constants                                                                   #
# -----------------------------------------------------------------------------#
RATE          = 192_000          # Hz
CHUNK         = int(0.1 * RATE)  # 0.1-s chunks
LOWCUT        = 20_000.0          # Hz (band-pass lower edge)
HIGHCUT       = 70_000.0         # Hz (band-pass upper edge)
FILTER_ORDER  = 5
SPEED_OF_SOUND = 343             # m/s

# Leak-tone detection
DETECTION_F       = 25_000.0     # Hz
F_BANDWIDTH       = 1_000.0      # Hz (±500 Hz)
LEAK_THRESHOLD_DB = -12.0        # dB (beamform when ≥ −40 dB)

# Beamforming scan grid
AZIMUTH_RANGE   = np.arange(-180, 181, 2)   # deg
ELEVATION_RANGE = np.arange( -2.5,  2.5, 0.2)   # deg

# Eight-mic geometry (x, y, z) [m]
MIC_POSITIONS = [
    ( 0.000,  0.000, 0.02), ( 0.000,  0.010, 0.02),
    ( 0.000, -0.015, 0.00), ( 0.000,  0.025, 0.00),
    ( 0.005,  0.005, 0.02), (-0.005,  0.005, 0.02),
    ( 0.020,  0.005, 0.00), (-0.020,  0.005, 0.00),
]
CHANNELS = len(MIC_POSITIONS)

# -----------------------------------------------------------------------------#
#  Pre-compute integer sample delays                                           #
# -----------------------------------------------------------------------------#
precomputed_delays = np.empty(
    (len(AZIMUTH_RANGE), len(ELEVATION_RANGE), CHANNELS), dtype=np.int32
)
for i, az in enumerate(AZIMUTH_RANGE):
    for j, el in enumerate(ELEVATION_RANGE):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            MIC_POSITIONS, az, el, RATE, SPEED_OF_SOUND
        )

# -----------------------------------------------------------------------------#
#  Helper: RMS magnitude in a narrow band                                      #
# -----------------------------------------------------------------------------#
def leak_band_magnitude(chunk, fs, f0, bandwidth):
    """
    Return linear RMS magnitude of |X(f)| inside f0 ± bandwidth/2.
    `chunk` is float32, range ±1, shape (N, C).
    """
    win   = np.hanning(chunk.shape[0])
    X     = np.fft.rfft(chunk.mean(axis=1) * win)
    freqs = np.fft.rfftfreq(chunk.shape[0], 1 / fs)
    idx   = (freqs >= f0 - bandwidth / 2) & (freqs <= f0 + bandwidth / 2)
    return np.sqrt(np.mean(np.abs(X[idx]) ** 2))

# -----------------------------------------------------------------------------#
#  Core routine                                                                #
# -----------------------------------------------------------------------------#
def process_audio_file(
    wav_path: str,
    video_out: str,
    dpi: int = 150,
    start_seconds: float = 0.0,
    show_live: bool = True,
):
    """Beamform `wav_path` and save MP4 while plotting live."""

    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    wf = wave.open(wav_path, "rb")

    # --- sanity checks ---
    if wf.getnchannels() != CHANNELS:
        wf.close(); raise ValueError("Channel mismatch.")
    if wf.getsampwidth() != 2:
        wf.close(); raise ValueError("Only 16-bit PCM WAV supported.")
    if wf.getframerate() != RATE:
        wf.close(); raise ValueError(f"WAV rate {wf.getframerate()} ≠ {RATE} Hz.")

    # --- optional seek ---
    if start_seconds > 0:
        wf.setpos(int(start_seconds * RATE))
    start_frame = wf.tell()

    total_frames = wf.getnframes() - start_frame
    total_steps  = total_frames // CHUNK
    chunk_dur    = CHUNK / RATE             # 0.1 s

    # ---------- plotting ----------
    if show_live:
        plt.ion()
    else:
        plt.ioff()

    fig, (ax_map, ax_leak) = plt.subplots(
        nrows=2, figsize=(12, 6),
        gridspec_kw=dict(height_ratios=[3, 1])
    )

    # (a) beamforming heat-map (initially blank)
    heatmap = ax_map.imshow(
        np.zeros((len(AZIMUTH_RANGE), len(ELEVATION_RANGE))).T,
        extent=[AZIMUTH_RANGE[0], AZIMUTH_RANGE[-1],
                ELEVATION_RANGE[0], ELEVATION_RANGE[-1]],
        origin="lower", cmap="inferno",
        aspect="auto", vmin=0, vmax=1, visible=False
    )
    fig.colorbar(heatmap, ax=ax_map, label="Energy")
    max_marker, = ax_map.plot([], [], "ro", label="Max Energy", visible=False)
    ax_map.set_xlabel("Azimuth [deg]")
    ax_map.set_ylabel("Elevation [deg]")
    ax_map.set_title("Delay-and-Sum Energy Map (gated)")
    ax_map.grid(True); ax_map.legend()

    time_txt = ax_map.text(
        0.02, 0.95, "", transform=ax_map.transAxes,
        color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none")
    )

    # (b) leak-tone magnitude trace
    leak_line, = ax_leak.plot([], [], lw=1.5,
                              label=f"|X|  @  {DETECTION_F/1000:.1f} kHz")
    ax_leak.axhline(LEAK_THRESHOLD_DB, color="r", ls="--",
                    lw=0.8, label="Threshold")
    ax_leak.set_xlim(0, total_steps * chunk_dur)
    ax_leak.set_ylim(LEAK_THRESHOLD_DB - 10, LEAK_THRESHOLD_DB + 20)
    ax_leak.set_xlabel("Time [s]")
    ax_leak.set_ylabel("Magnitude [dB]")
    ax_leak.legend(); ax_leak.grid(True)

    # ---------- video writer ----------
    fps    = int(round(1.0 / chunk_dur))            # 10 fps
    writer = animation.FFMpegWriter(fps=fps, bitrate=3200)

    print(f"Exporting → {video_out}  |  {total_steps} frames @ {fps} fps")
    t0_wall = time.time()

    leak_t, leak_db = [], []

    with writer.saving(fig, video_out, dpi=dpi):
        for step in range(total_steps):
            # --- read chunk ---
            raw = wf.readframes(CHUNK)
            if len(raw) < CHUNK * CHANNELS * 2:
                break
            chunk = (
                np.frombuffer(raw, np.int16)
                .reshape(-1, CHANNELS)
                .astype(np.float32) / 32768.0           # normalised ±1
            )

            # --- leak magnitude ---
            mag_lin = leak_band_magnitude(
                chunk, RATE, DETECTION_F, F_BANDWIDTH
            )
            mag_db  = 20.0 * np.log10(mag_lin + 1e-12)

            # --- update leak plot ---
            t_cur = step * chunk_dur + start_seconds
            leak_t.append(t_cur); leak_db.append(mag_db)
            leak_line.set_data(leak_t, leak_db)
            ax_leak.set_xlim(0, t_cur + chunk_dur)
            y_low, y_high = ax_leak.get_ylim()
            if mag_db > y_high - 2:
                ax_leak.set_ylim(y_low, mag_db + 10)
            if mag_db < y_low + 2:
                ax_leak.set_ylim(mag_db - 10, y_high)

            # ---------- gated beamforming ----------
            if mag_db >= LEAK_THRESHOLD_DB:
                filt = apply_bandpass_filter(
                    chunk, LOWCUT, HIGHCUT, RATE, FILTER_ORDER
                )
                energy = np.zeros(
                    (len(AZIMUTH_RANGE), len(ELEVATION_RANGE)), dtype=np.float32
                )
                for i in range(len(AZIMUTH_RANGE)):
                    for j in range(len(ELEVATION_RANGE)):
                        bf = apply_beamforming(
                            filt, precomputed_delays[i, j, :]
                        )
                        energy[i, j] = np.sum(bf ** 2)

                # autoscale colour map
                vmax = energy.max()
                vmin = np.percentile(energy, 5)
                if vmax <= 0: vmin, vmax = 0, 1

                heatmap.set_data(energy.T)
                heatmap.set_clim(vmin, vmax)
                heatmap.set_visible(True)

                max_idx = np.unravel_index(
                    np.argmax(energy), energy.shape
                )
                est_az = AZIMUTH_RANGE[max_idx[0]]
                est_el = ELEVATION_RANGE[max_idx[1]]
                max_marker.set_data([est_az], [est_el])
                max_marker.set_visible(True)
            else:
                # hide heat-map and marker
                heatmap.set_visible(False)
                max_marker.set_visible(False)

            # overlay time
            time_txt.set_text(f"t = {t_cur:.2f} s")

            # grab frame + live refresh
            writer.grab_frame()
            if show_live:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

            # progress bar
            prog = (step + 1) / total_steps * 100
            sys.stdout.write(f"\rProgress: {prog:6.2f}%")
            sys.stdout.flush()

    wf.close()
    plt.close(fig)

    wall = time.time() - t0_wall
    print(f"\nDone.  Encoding time: {wall:.1f} s  "
          f"({wall/total_steps:.3f} s/frame → RT ×"
          f"{chunk_dur / (wall / total_steps):.2f})")

# -----------------------------------------------------------------------------#
#  Stand-alone call                                                            #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    WAV_PATH  = (
        r"C:\Users\30068385\OneDrive - Western Sydney University\recordings\GasLeake\robotF.wav"
    )
    VIDEO_OUT = "robotF_leakBF.mp4"

    process_audio_file(
        wav_path=WAV_PATH,
        video_out=VIDEO_OUT,
        start_seconds=0.0,  # skip first 330 s
        show_live=True,
    )
