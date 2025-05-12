import numpy as np
import soundfile as sf

def make_wave_loopable(wav_data, crossfade_samples=2048):
    """
    Creates a new array that is loopable by crossfading the end and the beginning.

    Args:
        wav_data (np.ndarray): Audio samples (1D).
        crossfade_samples (int): Number of samples for the crossfade region.

    Returns:
        np.ndarray: A new array that loops smoothly.
    """

    N = len(wav_data)
    if crossfade_samples >= N // 2:
        raise ValueError("crossfade_samples is too large compared to length of WAV.")

    # 1) Get head (start) and tail (end)
    head = wav_data[:crossfade_samples]
    tail = wav_data[-crossfade_samples:]

    # 2) Create crossfaded region
    crossfaded = np.zeros(crossfade_samples, dtype=wav_data.dtype)
    for i in range(crossfade_samples):
        frac = i / float(crossfade_samples - 1)
        # fade out tail, fade in head
        s_tail = tail[i] * (1.0 - frac)
        s_head = head[i] * frac
        crossfaded[i] = s_tail + s_head

    # 3) Combine the middle and the crossfaded region
    #    We cut off crossfade_samples from the start and from the end,
    #    then append the crossfaded portion.
    #    This ensures final samples match the initial samples for a continuous loop.
    new_wav = np.concatenate([
        wav_data[crossfade_samples:-crossfade_samples],
        crossfaded
    ])
    return new_wav

input_wav = "/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20seg.wav"
wav_data, sr = sf.read(input_wav)

crossfade_samples = 2048  # ~ 42ms a 48kHz
looped_data = make_wave_loopable(wav_data, crossfade_samples=crossfade_samples)

output_wav = "/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20seg_loopable.wav"
sf.write(output_wav, looped_data, sr)
print(f"Loopable WAV saved to {output_wav}")