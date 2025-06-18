import numpy as np
import matplotlib.pyplot as plt
import wave
import time
from matplotlib.patches import Rectangle
from collections import deque
from matplotlib.animation import FFMpegWriter

# -------------------------------------------------------------------------
# External DSP helpers (import these from your own Utilities module)
# -------------------------------------------------------------------------
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# -------------------------------------------------------------------------
# Global parameters
# -------------------------------------------------------------------------
RATE = 48_000            # Sampling rate (Hz)
CHUNK = int(0.1 * RATE)  # 100 ms per block
LOWCUT = 180.0           # Band-pass filter low cutoff (Hz)
HIGHCUT = 2000.0         # Band-pass filter high cutoff (Hz)
FILTER_ORDER = 5         # Butterworth filter order
c = 343                  # Speed of sound in air (m/s)

# Azimuth from -180° … +180°, Elevation from 0° … 90°, both at 1° resolution
azimuth_range    = np.arange(-180, 181,   1)  # [–180, –179, …, 179, 180]
elevation_range  = np.arange(  0,   91,   1)  # [0, 1, 2, …, 90]

# Load microphone geometry and infer channel count
mic_positions = microphone_positions_8_helicop()
CHANNELS = mic_positions.shape[0]

# Precompute integer delay samples for every (az, el) pair
precomputed_delays = np.empty(
    (len(azimuth_range), len(elevation_range), CHANNELS),
    dtype=np.int32
)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, c
        )

# Smoothing window length (number of 100 ms blocks)
SMOOTH_LEN = 10  # e.g., 10 blocks = 1 second of smoothing

# Rectangle size as 10% of the total axis span
az_span = azimuth_range[-1] - azimuth_range[0]
el_span = elevation_range[-1] - elevation_range[0]
RECT_WIDTH  = 0.1 * az_span   # 10% of azimuth range
RECT_HEIGHT = 0.1 * el_span   # 10% of elevation range


def process_audio_and_export_two_plots(
    wav_filename: str,
    heatmap_mov: str            = "beamforming_heatmapR.mov",
    rect_mov:     str            = "beamforming_rectangleR.mov",
    fps:          int            = 10,
    dpi:          int            = 200,
    start_time:   float          = 0.0,
    end_time:     float          = None,
):
    """
    Read a multichannel WAV, run delay-and-sum beamforming over 100 ms blocks,
    smooth the azimuth/elevation estimates with a moving average (SMOOTH_LEN), and export:
      1) A .mov file containing only the data-dependent heatmap with alpha channel.
      2) A .mov file containing only the red rectangle outline on a fully transparent background.

    Both outputs cover the same time segment (start_time → end_time) at sampling FPS.
    """

    # --- Open WAV and sanity checks ---
    wf = wave.open(wav_filename, "rb")
    if wf.getnchannels() != CHANNELS:
        raise ValueError(f"Expected {CHANNELS} channels; got {wf.getnchannels()}")
    if wf.getsampwidth() != 2:
        raise ValueError("WAV sample width must be 16‐bit.")
    if wf.getframerate() != RATE:
        raise ValueError(f"WAV sampling rate must be {RATE} Hz.")

    total_frames = wf.getnframes()
    start_frame = int(start_time * RATE)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError("start_time out of range.")
    wf.setpos(start_frame)

    if end_time is not None:
        end_frame = int(end_time * RATE)
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError("end_time out of range or <= start_time.")
        max_blocks = (end_frame - start_frame) // CHUNK
    else:
        max_blocks = (total_frames - start_frame) // CHUNK

    # =========================
    # Figure #1: Heatmap Only
    # =========================
    fig_hm, ax_hm = plt.subplots(figsize=(12, 3), facecolor="none")
    fig_hm.patch.set_alpha(0.0)
    ax_hm.patch.set_alpha(0.0)

    # Initialize an all‐transparent RGBA image
    initial_rgba = np.zeros((len(elevation_range), len(azimuth_range), 4), dtype=float)
    heatmap = ax_hm.imshow(
        initial_rgba,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )
    cbar = fig_hm.colorbar(heatmap, ax=ax_hm, label="Normalized Energy")
    cbar.ax.patch.set_alpha(0.0)
    cbar.outline.set_alpha(0.7)

    ax_hm.set_xlim(azimuth_range[0], azimuth_range[-1])
    ax_hm.set_ylim(elevation_range[0], elevation_range[-1])
    ax_hm.set_xlabel("Azimuth (deg)", color="white")
    ax_hm.set_ylabel("Elevation (deg)", color="white")
    ax_hm.set_title("Beamforming Heatmap Only (Data-Dependent Transparency)", color="white")
    ax_hm.tick_params(colors="white")

    # Time indicator text (top-left, semi-transparent box)
    time_text_hm = ax_hm.text(
        0.02, 0.95, "Time: 0.00 s",
        transform=ax_hm.transAxes,
        color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    # ===============================
    # Figure #2: Rectangle Only (No Heatmap)
    # ===============================
    fig_rect, ax_rect = plt.subplots(figsize=(12, 3), facecolor="none")
    fig_rect.patch.set_alpha(0.0)
    ax_rect.patch.set_alpha(0.0)

    # Create a red rectangle outline (centered off-grid initially)
    rect = Rectangle(
        (azimuth_range[0], elevation_range[0]),  # initial off-grid position
        RECT_WIDTH,
        RECT_HEIGHT,
        facecolor="none",   # transparent interior
        edgecolor="red",    # red border
        linewidth=2.0,      # border thickness
        alpha=0.8           # border opacity
    )
    ax_rect.add_patch(rect)
    ax_rect.set_xlim(azimuth_range[0], azimuth_range[-1])
    ax_rect.set_ylim(elevation_range[0], elevation_range[-1])
    ax_rect.set_xticks([])  # hide x-axis ticks
    ax_rect.set_yticks([])  # hide y-axis ticks
    ax_rect.set_title("Beamforming Rectangle Only (Transparent Background)", color="white")

    # Time indicator text for rectangle figure
    time_text_rect = ax_rect.text(
        0.02, 0.95, "Time: 0.00 s",
        transform=ax_rect.transAxes,
        color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    # -----------------------------------
    # Prepare smoothing buffers
    # -----------------------------------
    az_buffer = deque(maxlen=SMOOTH_LEN)
    el_buffer = deque(maxlen=SMOOTH_LEN)

    # -----------------------------------
    # FFmpeg writers for both figures
    # -----------------------------------
    metadata = {
        "title":   "Beamforming Outputs",
        "artist":  "SSL pipeline",
        "comment": "Heatmap & Rectangle exports with alpha channel"
    }
    writer_hm   = FFMpegWriter(
        fps=fps,
        metadata=metadata,
        codec="qtrle",           # Apple RLE codec for .mov with alpha
        extra_args=["-pix_fmt", "rgba"]
    )
    writer_rect = FFMpegWriter(
        fps=fps,
        metadata=metadata,
        codec="qtrle",
        extra_args=["-pix_fmt", "rgba"]
    )

    # -----------------------------------
    # Concurrently save both outputs
    # -----------------------------------
    with writer_hm.saving(fig_hm, heatmap_mov, dpi=dpi), \
         writer_rect.saving(fig_rect, rect_mov, dpi=dpi):

        blocks_processed = 0

        while blocks_processed < max_blocks:
            frames = wf.readframes(CHUNK)
            if len(frames) < CHUNK * CHANNELS * 2:
                break  # not enough frames → stop early

            # Convert raw bytes → int16 array shaped (CHUNK, CHANNELS)
            audio = np.frombuffer(frames, dtype=np.int16).reshape((-1, CHANNELS))

            # Band-pass filter and normalize
            filtered = apply_bandpass_filter(audio, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)
            max_val = np.abs(filtered).max()
            if max_val != 0:
                filtered = filtered / max_val

            # Compute raw energy map (azimuth × elevation)
            energy = np.zeros((len(azimuth_range), len(elevation_range)))
            for i in range(len(azimuth_range)):
                for j in range(len(elevation_range)):
                    y = apply_beamforming(filtered, precomputed_delays[i, j, :])
                    energy[i, j] = np.sum(y ** 2) / CHUNK

            # Find indices of max energy
            idx_max = np.unravel_index(np.argmax(energy), energy.shape)
            raw_az = azimuth_range[idx_max[0]]
            raw_el = elevation_range[idx_max[1]]

            # Update moving-average buffers
            az_buffer.append(raw_az)
            el_buffer.append(raw_el)
            smooth_az = float(np.mean(az_buffer))
            smooth_el = float(np.mean(el_buffer))

            # =========================
            # Update Heatmap Figure
            # =========================
            # Determine color limits
            vmin = np.percentile(energy, 50)  # 50th percentile
            vmax = energy.max()
            norm       = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap       = plt.cm.Blues_r
            normalized = norm(energy)                  # shape: (len(az), len(el))
            normalized = np.clip(normalized, 0.0, 1.0)
            rgba_img   = cmap(normalized.T)            # transpose → (elevation, azimuth, 4)
            rgba_img[..., 3] = 1.0 - normalized.T       # override alpha (max energy → transparent)

            plt.figure(fig_hm.number)
            heatmap.set_data(rgba_img)

            current_time = start_time + blocks_processed * (CHUNK / RATE)
            time_text_hm.set_text(f"Time: {current_time:.2f} s")

            # Explicitly draw updated artists
            ax_hm.draw_artist(heatmap)
            ax_hm.draw_artist(time_text_hm)
            fig_hm.canvas.draw()
            fig_hm.canvas.flush_events()

            writer_hm.grab_frame()  # record this frame for the heatmap

            # =========================
            # Update Rectangle Figure
            # =========================
            plt.figure(fig_rect.number)

            # Position rectangle centered at smoothed az/el
            rect_x = smooth_az - (RECT_WIDTH / 2.0)
            rect_y = smooth_el - (RECT_HEIGHT / 2.0)
            rect.set_xy((rect_x, rect_y))

            time_text_rect.set_text(f"Time: {current_time:.2f} s")

            ax_rect.draw_artist(rect)
            ax_rect.draw_artist(time_text_rect)
            fig_rect.canvas.draw()
            fig_rect.canvas.flush_events()

            writer_rect.grab_frame()  # record this frame for the rectangle

            blocks_processed += 1
            time.sleep(CHUNK / RATE)  # maintain near real-time pacing

    wf.close()
    plt.close(fig_hm)
    plt.close(fig_rect)
    print(f"✅ Heatmap-only .mov exported → {heatmap_mov}")
    print(f"✅ Rectangle-only .mov exported → {rect_mov}")


# -------------------------------------------------------------------------
# Example usage (adjust paths as needed)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    input_wav     =         "/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/30_05_25/2_Right/Macarthur/right.wav"
    heatmap_output = input_wav.rsplit(".", 1)[0] + "_heatmap.mov"
    rect_output    = input_wav.rsplit(".", 1)[0] + "_rectangle.mov"

    process_audio_and_export_two_plots(
        wav_filename = input_wav,
        heatmap_mov  = heatmap_output,
        rect_mov     = rect_output,
        fps          = int(1 / 0.1),  # 10 fps for 100 ms blocks
        start_time   = 3420.0,        # start at 3420.0 s
        end_time     = 3498.0         # end at 3498.0 s
    )
