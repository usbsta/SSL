import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys
from scipy.signal import butter, filtfilt

FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
RECORD_SECONDS = 12000
c = 343
TARGET_DIFFERENCE = 200e-6
peak_threshold = 0.5e8

device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(10, 91, 5)

lowcut = 400.0
highcut = 8000.0


a = [0, -120, -240]

# config 1 equidistance
h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

# config 2 augmented
h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]

audio = pyaudio.PyAudio()

output_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']
output_filenames2 = ['device_1_nosync.wav', 'device_2_nosync.wav', 'device_3_nosync.wav']
"""
mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],  # Mic 6
    [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],  # Mic 7
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],  # Mic 8
    [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]],  # Mic 9
    [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]],  # Mic 10
    [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]],  # Mic 11
    [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],  # Mic 12
    [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],  # Mic 13
    [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]],  # Mic 14
    [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]],  # Mic 15
    [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]],  # Mic 16
    [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],  # Mic 17
    [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]]]) # Mic 18])
"""

# config 3 rand
h = [1.12, 0.97, 0.98, 0.97, 0.81, 0.88, 0.76, 0.68, 0.67, 0.65, 0.63, 0.49, 0.42, 0.02, 0.27, 0.34]
r = [0.1, 0.18, 0.15, 0.18, 0.26, 0.24, 0.29, 0.3, 0.33, 0.35, 0.36, 0.37, 0.44, 0.63, 0.48, 0.46]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],  # Mic 5
    [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],  # Mic 6
    [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],  # Mic 7
    [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],  # Mic 8
    [r[6] * np.cos(np.radians(a[2])), r[6] * np.sin(np.radians(a[2])), h[6]],  # Mic 9
    [r[7] * np.cos(np.radians(a[0])), r[7] * np.sin(np.radians(a[0])), h[7]],  # Mic 10
    [r[8] * np.cos(np.radians(a[1])), r[8] * np.sin(np.radians(a[1])), h[8]],  # Mic 11
    [r[9] * np.cos(np.radians(a[2])), r[9] * np.sin(np.radians(a[2])), h[9]],  # Mic 12
    [r[10] * np.cos(np.radians(a[0])), r[10] * np.sin(np.radians(a[0])), h[10]],  # Mic 13
    [r[11] * np.cos(np.radians(a[1])), r[11] * np.sin(np.radians(a[1])), h[11]],  # Mic 14
    [r[12] * np.cos(np.radians(a[2])), r[12] * np.sin(np.radians(a[2])), h[12]],  # Mic 15
    [r[13] * np.cos(np.radians(a[0])), r[13] * np.sin(np.radians(a[0])), h[13]],  # Mic 16
    [r[14] * np.cos(np.radians(a[1])), r[14] * np.sin(np.radians(a[1])), h[14]],  # Mic 17
    [r[15] * np.cos(np.radians(a[2])), r[15] * np.sin(np.radians(a[2])), h[15]] # Mic 18
])


# buffers preparation to store audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)] # syncro
buffers2 = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]
frames = [[], [], []]  # frames for 3 devices
frames2 = [[], [], []]  # frames for 3 devices

correction_12 = None  # correction between 1 and 2
correction_13 = None

total_correction_12 = 0
total_correction_13 = 0

synced_12 = False
synced_13 = False

window_size = int(0.05 * RATE)

buffer_lock = threading.Lock()

stop_event = threading.Event()


def shift_signal(signal, shift_amount):
    if shift_amount > 0:  # shift to right "Retrasando"
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # shift to left "Avanzando"
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal
    return shifted_signal

def plot_peaks(signals, peaks_indices, labels):
    plt.figure()
    time_axis = np.linspace(0, 50, window_size)
    for signal, peak_idx, label in zip(signals, peaks_indices, labels):
        plt.plot(time_axis, signal, label=label)
        if peak_idx is not None and 0 <= peak_idx < len(signal):  # Verificar que el índice del pico esté dentro de los límites
            plt.plot(time_axis[peak_idx], signal[peak_idx], 'ro')
            plt.text(time_axis[peak_idx], signal[peak_idx], f'Peak {peak_idx}', color='red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Detected Peaks in 50 ms Window')
    plt.show()


# beamforming
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # 3D direction vector
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            delays = np.dot(mic_positions, direction_vector) / c

            # aplying delays
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np.roll(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1] # normalize amplitud with num of mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

# band pass design
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)  # Aplicar filtro a lo largo de la señal en cada canal
    return filtered_signal

def record_device(device_index, buffer_index, stop_event):
    global correction_12, correction_13, total_correction_12, total_correction_13

    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)

        print(f"Recording from device {device_index}...")

        while not stop_event.is_set():
            data = stream.read(CHUNK)
            signal_data = np.frombuffer(data, dtype=np.int32)

            with buffer_lock:
                buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))
                buffers2[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

                # Apply cum correctoions
                if buffer_index == 1 and total_correction_12 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_12)

                if buffer_index == 2 and total_correction_13 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_13)

                frames[buffer_index].append(buffers[buffer_index].tobytes())
                frames2[buffer_index].append(buffers2[buffer_index].tobytes())

    except Exception as e:
        print(f"Error recording from device {device_index}: {e}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print(f"Stream closed for device {device_index}")


def stop_recording_and_save():
    stop_event.set()

    # wait end of threads
    for thread in threads:
        thread.join()

    print("Stopping recording...")

    for i in range(3):
        with wave.open(output_filenames[i], 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames[i]))

        with wave.open(output_filenames2[i], 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames2[i]))

    audio.terminate()
    print("Recording saved and audio terminated.")


def signal_handler(sig, frame):
    print("Signal received, stopping recording...")
    stop_recording_and_save()
    sys.exit(0)


# (Ctrl + C) stop process
signal.signal(signal.SIGINT, signal_handler)

threads = []
threads.append(threading.Thread(target=record_device, args=(device_index_1, 0, stop_event)))
threads.append(threading.Thread(target=record_device, args=(device_index_2, 1, stop_event)))
threads.append(threading.Thread(target=record_device, args=(device_index_3, 2, stop_event)))

for thread in threads:
    thread.start()


# real time plot
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('beamforming enery')

# max energy point
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    combined_signal = np.hstack(buffers)
    combined_signal2 = np.hstack(buffers2)
    num_samples = combined_signal.shape[0]

    peaks_1, _ = find_peaks(buffers[0][:, 3], height=peak_threshold)
    peaks_2, _ = find_peaks(buffers[1][:, 0], height=peak_threshold)
    peaks_4, _ = find_peaks(buffers[2][:, 3], height=peak_threshold)

    if peaks_1.size > 0 and peaks_2.size > 0 and peaks_4.size > 0:
        # Extraer una ventana de 3 ms alrededor de cada pico para graficar
        start_idx = max(peaks_1[0] - window_size // 2, 0)
        end_idx = start_idx + window_size

        # Asegurarse de que end_idx no exceda la longitud del buffer
        if end_idx > buffers[0].shape[0]:
            end_idx = buffers[0].shape[0]
            start_idx = end_idx - window_size

        signal_1_ch5 = buffers[0][start_idx:end_idx, 3] #chanel 4 device 1
        signal_2_ch1 = buffers[1][start_idx:end_idx, 0] #chanel 1 device 2
        signal_3_ch1 = buffers[2][start_idx:end_idx, 3] #chanel 4 device 3

        signals = [signal_1_ch5, signal_2_ch1, signal_3_ch1]
        peaks_indices = [peaks_1[0] - start_idx, peaks_2[0] - start_idx, peaks_4[0] - start_idx]
        labels = ['Device 1 Channel 4', 'Device 2 Channel 1', 'Device 3 Channel 4']
        plot_peaks(signals, peaks_indices, labels)

    # synchronization
    if peaks_1.size > 0 and peaks_2.size > 0 and not synced_12:
        peak_time_1 = peaks_1[0] / RATE
        peak_time_2 = peaks_2[0] / RATE
        time_difference_12 = peak_time_2 - peak_time_1
        print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds")

        sample_difference_12 = int(time_difference_12 * RATE)

        if abs(time_difference_12) > TARGET_DIFFERENCE:
            correction_12 = -sample_difference_12
            total_correction_12 += correction_12
            buffers[1] = shift_signal(buffers[1], correction_12)
            print(f"Applying correction of {correction_12} samples to synchronize devices 1 and 2.")
            synced_12 = True  # flag after synch 1 and 2
            print("synchronized 1 and 2")

        else:
            print(
                f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds (less than TARGET_DIFFERENCE, no further adjustment)")

    if peaks_1.size > 0 and peaks_4.size > 0 and not synced_13:
        peak_time_1 = peaks_1[0] / RATE
        peak_time_4 = peaks_4[0] / RATE
        time_difference_13 = peak_time_4 - peak_time_1
        print(f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds")

        sample_difference_13 = int(time_difference_13 * RATE)

        if abs(time_difference_13) > TARGET_DIFFERENCE:
            correction_13 = -sample_difference_13
            total_correction_13 += correction_13
            buffers[2] = shift_signal(buffers[2], correction_13)
            print(f"Applying correction of {correction_13} samples to synchronize devices 1 and 3.")
            synced_13 = True
            print("synchronized 1 and 3")

        else:
            print(
                f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds (less than TARGET_DIFFERENCE, no further adjustment)")


    # filtering
    filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)
    num_samples = filtered_signal.shape[0]

    #energy = beamform_time(combined_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
    energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)

    max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[0]]
    estimated_elevation = elevation_range[max_energy_idx[1]]
    print(f"Estimated angle: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")
    #print(np.min(energy),np.max(energy))

    # Actualizar los datos del mapa de calor
    cax.set_data(energy.T)
    cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color

    # Actualizar la posición del marcador de máxima energía
    max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

    # Actualizar la posición del texto con las coordenadas
    max_energy_text.set_position((estimated_azimuth, estimated_elevation))
    max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

    fig.canvas.draw()
    fig.canvas.flush_events()

