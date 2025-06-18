import numpy as np
import matplotlib.pyplot as plt
import wave
import time
from matplotlib.patches import Rectangle
from collections import deque
from matplotlib.animation import FFMpegWriter

# -------------------------------------------------------------------------
# External DSP helpers (imported from your Utilities module)
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
FILTER_ORDER = 5         # Butterworth order
c = 343                  # Speed of sound in air (m/s)

# Azimuth from -180° … +180°, Elevation from 0° … 90°, both at 1° resolution
azimuth_range = np.arange(-180, 181, 1)   # [−180, -179, …, 179, 180]
elevation_range = np.arange(0, 91, 1)     # [0, 1, 2, …, 90]

# Microphone geometry
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
RECT_WIDTH = 0.1 * az_span   # 10% of azimuth range
RECT_HEIGHT = 0.1 * el_span  # 10% of elevation range

# -------------------------------------------------------------------------
# Real-time plotting + MOV export with data-dependent alpha (max energy → transparent)
# -------------------------------------------------------------------------
def process_audio_segment_to_mov(
    wav_filename: str,
    mov_filename: str = "beamforming_overlay.mov",
    fps: int = 10,
    dpi: int = 200,
    start_time: float = 0.0,
    end_time: float = None,
):
    """
    Read a multichannel WAV, run delay‐and‐sum beamforming over 100 ms blocks,
    smooth the az/el estimates with a moving average (SMOOTH_LEN), display a heatmap
    whose alpha channel is inversely proportional to energy (highest-energy pixels
    are fully transparent), draw a red rectangle outline at the smoothed max-energy location,
    and write out a .mov file with alpha channel (qtrle codec) so you can overlay
    this on another video.
    """
    # Open WAV and sanity‐check
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

    # Prepare interactive plot with transparent background
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Prepare an initial RGBA image of zeros (fully transparent)
    initial_rgba = np.zeros((len(elevation_range), len(azimuth_range), 4), dtype=float)
    heatmap = ax.imshow(
        initial_rgba,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )
    cbar = fig.colorbar(heatmap, ax=ax, label="Normalized Energy")
    cbar.ax.patch.set_alpha(0.0)
    cbar.outline.set_alpha(0.7)

    ax.set_xlim(azimuth_range[0], azimuth_range[-1])
    ax.set_ylim(elevation_range[0], elevation_range[-1])
    ax.set_xlabel("Azimuth (deg)", color="white")
    ax.set_ylabel("Elevation (deg)", color="white")
    ax.set_title("Beamforming Energy Map (Data-Dependent Transparency)", color="white")
    ax.tick_params(colors="white")

    # Create a red rectangle patch outline (10% × 10% of axes span), no fill
    rect = Rectangle(
        (azimuth_range[0], elevation_range[0]),  # initial position off‐grid
        RECT_WIDTH,
        RECT_HEIGHT,
        facecolor="none",   # no fill (fully transparent interior)
        edgecolor="red",    # red border
        linewidth=2.0,      # thickness of the border
        alpha=0.8           # border opacity
    )
    ax.add_patch(rect)

    # Time indicator text (top‐left, semi‐transparent box)
    time_text = ax.text(
        0.02, 0.95, "Time: 0.00 s",
        transform=ax.transAxes,
        color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    # Set up deques for moving average
    az_buffer = deque(maxlen=SMOOTH_LEN)
    el_buffer = deque(maxlen=SMOOTH_LEN)

    # FFmpeg writer with qtrle codec for alpha in .mov
    metadata = {
        "title": "Beamforming Energy Map Overlay",
        "artist": "SSL pipeline",
        "comment": "Delay‐and‐sum beamforming with alpha channel",
    }
    writer = FFMpegWriter(
        fps=fps,
        metadata=metadata,
        codec="qtrle",           # Apple RLE codec for .mov with alpha
        extra_args=["-pix_fmt", "rgba"]  # ensure RGBA pixel format
    )

    blocks_processed = 0
    with writer.saving(fig, mov_filename, dpi=dpi):
        while blocks_processed < max_blocks:
            frames = wf.readframes(CHUNK)
            # If we don't have a full CHUNK worth of frames, break
            if len(frames) < CHUNK * CHANNELS * 2:
                break

            # Convert to NumPy array and normalize
            audio = np.frombuffer(frames, dtype=np.int16).reshape((-1, CHANNELS))
            filtered = apply_bandpass_filter(audio, LOWCUT, HIGHCUT, RATE,
                                             order=FILTER_ORDER)
            max_val = np.abs(filtered).max()
            if max_val != 0:
                filtered = filtered / max_val

            # Compute raw energy map
            energy = np.zeros((len(azimuth_range), len(elevation_range)))
            for i in range(len(azimuth_range)):
                for j in range(len(elevation_range)):
                    y = apply_beamforming(filtered, precomputed_delays[i, j, :])
                    energy[i, j] = np.sum(y ** 2) / CHUNK

            # Find raw max‐energy indices
            idx_max = np.unravel_index(np.argmax(energy), energy.shape)
            raw_az = azimuth_range[idx_max[0]]
            raw_el = elevation_range[idx_max[1]]

            # Update moving‐average buffers
            az_buffer.append(raw_az)
            el_buffer.append(raw_el)
            smooth_az = float(np.mean(az_buffer))
            smooth_el = float(np.mean(el_buffer))

            # Determine color limits for normalization
            vmin = np.percentile(energy, 50)   # 50th percentile
            vmax = energy.max()

            # Create a Normalize instance
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            # Get the colormap (without built-in alpha)
            cmap = plt.cm.Blues_r

            # Map energy to RGBA, then override alpha channel:
            #   alpha = 1 - normalized_energy → max energy → alpha=0 (fully transparent)
            normalized = norm(energy)                     # shape: (len(az), len(el))
            normalized = np.clip(normalized, 0.0, 1.0)
            rgba_img = cmap(normalized.T)                 # transpose to match imshow shape: (el, az, 4)
            rgba_img[..., 3] = 1.0 - normalized.T         # override alpha channel

            # Update heatmap with the new RGBA image
            heatmap.set_data(rgba_img)

            # Update rectangle position (centered at the smoothed az/el)
            rect_x = smooth_az - (RECT_WIDTH / 2.0)
            rect_y = smooth_el - (RECT_HEIGHT / 2.0)
            rect.set_xy((rect_x, rect_y))

            # Update time indicator
            current_time = start_time + blocks_processed * (CHUNK / RATE)
            time_text.set_text(f"Time: {current_time:.2f} s")

            # Draw / flush
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Grab this frame (with alpha) for the .mov
            writer.grab_frame()

            blocks_processed += 1
            time.sleep(CHUNK / RATE)  # maintain real‐time pacing

    wf.close()
    plt.ioff()
    plt.close(fig)
    print(f"✅ .mov exported → {mov_filename}")

# -------------------------------------------------------------------------
# Script entry‐point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    wav_files = [
        "/Users/30068385/OneDrive - Western Sydney University/recordings/"
        "Helicop/30_05_25/1_Left/Macarthur/left.wav"
    ]
    for wav_path in wav_files:
        mov_output = (
            wav_path.rsplit(".", 1)[0]
            + f"_{LOWCUT}_{HIGHCUT}3.mov"
        )
        process_audio_segment_to_mov(
            wav_filename=wav_path,
            mov_filename=mov_output,
            fps=int(1 / 0.1),  # 10 fps for 100 ms blocks
            start_time=3420.0,   # start at 3420.0 s
            end_time=3498.0      # end at 3498.0 s
        )
