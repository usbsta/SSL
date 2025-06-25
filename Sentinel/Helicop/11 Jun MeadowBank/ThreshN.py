import numpy as np
import matplotlib.pyplot as plt
import wave
import time
from matplotlib.animation import FFMpegWriter
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

from Utilities.mic_geo import (
    mic_6_N_black_thin,
    mic_6_S_orange,
    mic_6_E_orange,
    mic_6_W_black,
)

# ── Global parameters ──
RATE = 48_000                    # Sampling rate (Hz)
CHUNK = int(0.1 * RATE)          # 100 ms per block
LOWCUT = 100.0                   # Band-pass filter low cut (Hz)
HIGHCUT = 2000.0                 # Band-pass filter high cut (Hz)
FILTER_ORDER = 5                 # Butterworth order
c = 343                          # Speed of sound (m/s)

# Threshold band: center 500 Hz ±400 Hz → [100 Hz…900 Hz]
DETECT_F = 500.0                 # Center frequency for threshold (Hz)
BW = 200.0                       # Bandwidth (Hz)
THRESH_DB = 33.0                # Threshold in dB

# Azimuth/elevation grid for beamforming
azimuth_range = np.arange(-180, 181, 40)   # –180…+180 in 1° steps
elevation_range = np.arange(0, 91, 40)     # 0…90 in 1° steps

# Microphone geometry
mic_positions = mic_6_N_black_thin()
CHANNELS = mic_positions.shape[0]

# ── Precompute delays ──
precomputed_delays = np.empty(
    (len(azimuth_range), len(elevation_range), CHANNELS),
    dtype=np.int32
)
print("Pre-computing delays …")
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, c
        )

# ── Helper to compute RMS in [DETECT_F ± BW/2] band ──
def band_rms(chunk: np.ndarray) -> float:
    """
    Compute RMS of the mean signal across all channels,
    restricted to frequencies in [DETECT_F - BW/2 … DETECT_F + BW/2].
    The 'chunk' should already be band-pass filtered 180–2000 Hz.
    """
    window = np.hanning(chunk.shape[0])
    avg_signal = chunk.mean(axis=1) * window

    X = np.fft.rfft(avg_signal)
    freqs = np.fft.rfftfreq(chunk.shape[0], d=1.0 / RATE)

    fmin = DETECT_F - (BW / 2)
    fmax = DETECT_F + (BW / 2)
    mask = (freqs >= fmin) & (freqs <= fmax)

    return np.sqrt(np.mean(np.abs(X[mask])**2))


# ── Main processing function ──
def process_airplane_with_threshold(
    wav_filename: str,
    video_filename: str = "airplane_threshold_video.mp4",
    fps: int = 10,
    dpi: int = 200,
    start_time: float = 0.0,
    end_time: float = None,
):
    """
    Read a multichannel WAV, apply 180–2000 Hz band-pass first, then compute
    RMS in [100–900 Hz] band (bottom plot), and—when that exceeds THRESH_DB—
    overlay the beamforming energy map (top). The bottom subplot uses relative
    time (0…duration) and auto-adjusts its Y-limits within [-100, 50] dB so
    that the entire blue trace is always visible.
    """
    wf = wave.open(wav_filename, "rb")
    if wf.getnchannels() != CHANNELS:
        raise ValueError(f"Expected {CHANNELS} channels; got {wf.getnchannels()}")
    if wf.getsampwidth() != 2:
        raise ValueError("WAV sample width must be 16-bit.")
    if wf.getframerate() != RATE:
        raise ValueError(f"WAV sampling rate must be {RATE} Hz.")

    total_frames = wf.getnframes()
    # Compute start/end frame indices
    start_frame = int(start_time * RATE)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError("start_time out of range.")
    wf.setpos(start_frame)

    if end_time is not None:
        end_frame = int(end_time * RATE)
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError("end_time out of range or ≤ start_time.")
        max_blocks = (end_frame - start_frame) // CHUNK
    else:
        max_blocks = (total_frames - start_frame) // CHUNK

    # ── Set up figure with two rows: beamforming map on top, threshold plot below ──
    plt.ion()
    fig, (ax_map, ax_thresh) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw=dict(height_ratios=[3, 1])
    )

    # Top: Beamforming energy map (initially hidden)
    heatmap = ax_map.imshow(
        np.zeros((len(azimuth_range), len(elevation_range))).T,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="Blues_r",
        visible=False
    )
    fig.colorbar(heatmap, ax=ax_map, label="Beamforming Energy")
    ax_map.set_xlabel("Azimuth (deg)")
    ax_map.set_ylabel("Elevation (deg)")
    ax_map.set_title("Beamforming Energy Map (displayed only when signal ≥ threshold)")
    ax_map.grid(True)

    max_marker, = ax_map.plot([], [], "ro", label="Max energy", visible=False)
    ax_map.legend(loc="upper right")

    time_text = ax_map.text(
        0.02, 0.95, "Time: 0.00 s",
        transform=ax_map.transAxes,
        color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    # Bottom: Threshold‐band energy vs. relative time
    thresh_line, = ax_thresh.plot([], [], lw=1.5, color="blue", label=f"|X| @ {DETECT_F:.0f} Hz")
    ax_thresh.axhline(THRESH_DB, color="r", ls="--", lw=0.8, label="Threshold (dB)")
    ax_thresh.set_xlabel("Time (s) [relative]")
    ax_thresh.set_ylabel("Magnitude (dB)")
    ax_thresh.set_title("Threshold‐band Energy Over Time")
    ax_thresh.legend()
    ax_thresh.grid(True)
    # Initial X and Y limits
    ax_thresh.set_xlim(0, max_blocks * (CHUNK / RATE))
    ax_thresh.set_ylim(-100, 50)  # base Y-range; will auto-adjust within these bounds

    # Prepare FFmpeg writer
    meta = {
        "title": "Airplane Detection with Threshold",
        "artist": "SSL pipeline",
        "comment": f"Detection band: {int(DETECT_F - BW/2)}–{int(DETECT_F + BW/2)} Hz",
    }
    writer = FFMpegWriter(fps=fps, metadata=meta)

    t_buffer = []
    db_buffer = []
    blocks_processed = 0

    with writer.saving(fig, video_filename, dpi=dpi):
        while blocks_processed < max_blocks:
            frames = wf.readframes(CHUNK)
            if len(frames) < CHUNK * CHANNELS * 2:
                break

            # Convert raw samples to float32 in [–1, +1]
            audio = np.frombuffer(frames, dtype=np.int16).reshape((-1, CHANNELS)).astype(np.float32) / 32768.0

            # First apply the 180–2000 Hz band-pass
            filtered_full = apply_bandpass_filter(audio, LOWCUT, HIGHCUT, RATE, FILTER_ORDER)
            if filtered_full.max() != 0:
                filtered_full /= filtered_full.max()

            # Compute RMS in 100–900 Hz band on the already-filtered data
            rms_val = band_rms(filtered_full)
            db_val = 20 * np.log10(rms_val + 1e-12)

            # Use relative time (0…duration) on the x-axis
            rel_time = blocks_processed * (CHUNK / RATE)
            t_buffer.append(rel_time)
            db_buffer.append(db_val)
            thresh_line.set_data(t_buffer, db_buffer)

            # Dynamically adjust Y-limits but clamp within [-100, 50]
            min_db = min(db_buffer + [THRESH_DB])
            max_db = max(db_buffer + [THRESH_DB])
            y_min = max(min_db - 5, -100)
            y_max = min(max_db + 5, 50)
            ax_thresh.set_ylim(y_min, y_max)

            # Decide whether to show beamforming map
            if db_val >= THRESH_DB:
                energy = np.zeros((len(azimuth_range), len(elevation_range)), dtype=np.float32)
                for i in range(len(azimuth_range)):
                    for j in range(len(elevation_range)):
                        sig = apply_beamforming(filtered_full, precomputed_delays[i, j, :])
                        energy[i, j] = np.sum(sig**2) / CHUNK

                idx_max = np.unravel_index(np.argmax(energy), energy.shape)
                est_az = azimuth_range[idx_max[0]]
                est_el = elevation_range[idx_max[1]]

                heatmap.set_data(energy.T)
                heatmap.set_clim(vmin=np.percentile(energy, 50), vmax=energy.max())
                heatmap.set_visible(True)

                max_marker.set_data([est_az], [est_el])
                max_marker.set_visible(True)
            else:
                heatmap.set_visible(False)
                max_marker.set_visible(False)

            # Update the time label on the top subplot with absolute time
            abs_time = start_time + blocks_processed * (CHUNK / RATE)
            time_text.set_text(f"Time: {abs_time:.2f} s")

            # Draw and capture frame
            fig.canvas.draw()
            fig.canvas.flush_events()
            writer.grab_frame()

            time.sleep(CHUNK / RATE)
            blocks_processed += 1

    wf.close()
    plt.ioff()
    plt.close(fig)
    print(f"✅ Video exported → {video_filename}")


# ── Script entry‐point ──
if __name__ == "__main__":
    wav_files = ["/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25/N.wav"]
    for wav in wav_files:
        process_airplane_with_threshold(
            wav_filename=wav,
            video_filename=f"{wav.rsplit('.', 1)[0]}_thresh_{int(DETECT_F - BW/2)}to{int(DETECT_F + BW/2)}_Full_Hz.mp4",
            fps=int(1 / 0.1),   # 10 fps for 100 ms chunks
            start_time=0.0,
            end_time = 104.0
        )
