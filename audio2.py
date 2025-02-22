import socket
import json
from datetime import datetime
import threading
import csv
import struct
import time
import queue
import numpy as np
from scipy.signal import butter, filtfilt

# =============================================================================
# GLOBALS AND AUDIO SETTINGS
# =============================================================================
# Global processing queue for real-time audio processing
audio_processing_queue = queue.Queue(maxsize=50)  # NEW

fileNumberFromPythonScript = 0  # init
placeHolder = 1
"""Number of Devices connected"""
numberOfDevicesConnected = 1  # 1 master device only, 2 master and slave device
DEVICE_0_NUM = 0  # ALWAYS MASTER DEVICE!!
DEVICE_1_NUM = 1
DEVICE_2_NUM = 2

# -------------------------------
# AUDIO SETTINGS
# -------------------------------
audioSamplingRate = 48000
RATE = audioSamplingRate
RECORDING_DURATION = 1800
numberOfFilesToRecord = 1
audioDigitalVolume = 0
audioHighPassFilter = 1
audioGainCalibration = 0

# -------------------------------
# SENSOR SETTINGS
# -------------------------------
sensorDataBatchSize = 200
accelDataRate = 104
accelDataRange = 2
gyroDataRate = 104
gyroDataRange = 125

# =============================================================================
# NETWORK VARIABLES
# =============================================================================
MASTER_DEVICE_MACRO = 1
SLAVE_DEVICE_MACRO = 2
DEVICE_0 = MASTER_DEVICE_MACRO
DEVICE_1 = SLAVE_DEVICE_MACRO
DEVICE_0_ETHERNET_IP = "192.168.0.10" + str(DEVICE_0_NUM)
DEVICE_1_ETHERNET_IP = "192.168.0.10" + str(DEVICE_1_NUM)
DEVICE_2_ETHERNET_IP = "192.168.0.10" + str(DEVICE_2_NUM)
DEVICE_0_WIFI_IP = "192.168.0.20" + str(DEVICE_0_NUM)
DEVICE_1_WIFI_IP = "192.168.0.20" + str(DEVICE_1_NUM)
DEVICE_2_WIFI_IP = "192.168.0.20" + str(DEVICE_2_NUM)
DEVICE_0_PORT = 80
DEVICE_1_PORT = 80
DEVICE_2_PORT = 80

# =============================================================================
# BUFFER AND STATE VARIABLES
# =============================================================================
AUDIO_BUFFER_SIZE = 32768

masterDeviceEthernet = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
slaveDeviceEthernet = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
slaveDeviceEthernetDeviceNumber2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
masterDeviceWiFi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
slaveDeviceWiFi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
slaveDeviceWiFiDeviceNumber2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

audioBuffers = [
    [bytearray(AUDIO_BUFFER_SIZE), bytearray(AUDIO_BUFFER_SIZE)],
    [bytearray(AUDIO_BUFFER_SIZE), bytearray(AUDIO_BUFFER_SIZE)],
    [bytearray(AUDIO_BUFFER_SIZE), bytearray(AUDIO_BUFFER_SIZE)]
]

audioBufferLock = [threading.Lock(), threading.Lock(), threading.Lock()]
audioBufferReady = [threading.Condition(audioBufferLock[0]),
                    threading.Condition(audioBufferLock[1]),
                    threading.Condition(audioBufferLock[2])]

audioBufferStates = [[False, False], [False, False], [False, False]]
audioBufferDataSizes = [[0, 0], [0, 0], [0, 0]]

doneReceivingAudio = [False, False, False]
audio_data_size = [0, 0, 0]
total_samples = [0, 0, 0]

bits_per_sample = 16
total_samples_to_receive = audioSamplingRate * RECORDING_DURATION

if audioSamplingRate > 48000:
    num_channels = 4
else:
    num_channels = 8


# =============================================================================
# AUDIO RECEPTION MODIFICATION
# =============================================================================
def receiveAudioData(sock, device_type):
    """
    Receives audio data from a socket, stores it in buffers, and deposits the raw data
    into a processing queue for real-time analysis.
    """
    try:
        while total_samples[device_type] < total_samples_to_receive:
            with audioBufferLock[device_type]:
                # Wait until an available buffer is free
                while all(audioBufferStates[device_type]):
                    audioBufferReady[device_type].wait()

                # Find an available buffer index
                audioBufferIndex = audioBufferStates[device_type].index(False)

                # Receive data from the socket
                data = sock.recv(AUDIO_BUFFER_SIZE)
                if not data:
                    doneReceivingAudio[device_type] = True
                    audioBufferReady[device_type].notify_all()
                    break

                # Store the received data into the chosen buffer
                audioBuffers[device_type][audioBufferIndex][:len(data)] = data
                audioBufferDataSizes[device_type][audioBufferIndex] = len(data)
                audioBufferStates[device_type][audioBufferIndex] = True

                # Update the total samples received
                samples_received = len(data) * 8 // (num_channels * bits_per_sample)
                total_samples[device_type] += samples_received

                # Notify the writing thread that this buffer is now ready
                audioBufferReady[device_type].notify_all()

                # NEW: Deposit the data into the processing queue
                try:
                    audio_processing_queue.put_nowait(data)
                except queue.Full:
                    # If the queue is full, we can choose to discard the chunk or block (here we discard)
                    pass
    except Exception as e:
        print(f"An error occurred while receiving data: {e}")


# =============================================================================
# REAL-TIME PROCESSING FUNCTION
# =============================================================================
# Beamforming (delay-and-sum) function
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, rate, c=343):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))
    for az_idx, theta in enumerate(azimuth_range):
        az_rad = np.radians(theta)
        for el_idx, phi in enumerate(elevation_range):
            el_rad = np.radians(phi)
            direction = np.array([np.cos(el_rad) * np.cos(az_rad),
                                  np.cos(el_rad) * np.sin(az_rad),
                                  np.sin(el_rad)])
            delays = np.dot(mic_positions, direction) / c
            summed = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                shift = int(np.round(delay * rate))
                summed += np.roll(signal_data[:, i], shift)
            summed /= signal_data.shape[1]
            energy[az_idx, el_idx] = np.sum(summed ** 2)
    return energy


# Bandpass filtering functions
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut=400.0, highcut=18000.0, rate=RATE, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    return filtfilt(b, a, data, axis=0)


def process_audio_from_queue():
    # Define beamforming parameters
    azimuth_range = np.arange(-180, 181, 5)
    elevation_range = np.arange(0, 91, 5)
    radius = 0.3  # radius in meters
    angles = np.linspace(0, 360, 8, endpoint=False)
    mic_positions = np.array([[radius * np.cos(np.radians(a)),
                               radius * np.sin(np.radians(a)),
                               0] for a in angles])

    while True:
        try:
            # Get an audio chunk from the processing queue (wait up to 1 second)
            data = audio_processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            # Convert raw bytes to a NumPy array (assuming int16 and 8 channels)
            samples = np.frombuffer(data, dtype=np.int16)
            sample_count = len(samples) // 8
            if sample_count < 33:  # Skip if the chunk is too short for filtering
                audio_processing_queue.task_done()
                continue
            samples = samples[:sample_count * 8].reshape((sample_count, 8))
        except Exception as e:
            print(f"Conversion error: {e}")
            audio_processing_queue.task_done()
            continue

        try:
            # Apply bandpass filtering (this might fail if the chunk is too short)
            filtered = apply_bandpass_filter(samples, lowcut=400.0, highcut=18000.0, rate=RATE, order=5)
        except ValueError as e:
            print("Skipping chunk due to filtering error:", e)
            audio_processing_queue.task_done()
            continue

        # Apply beamforming to get the energy map
        energy = beamform_time(filtered, mic_positions, azimuth_range, elevation_range, RATE, c=343)

        # Estimate the direction by finding the maximum energy point
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]
        print(f"Estimated Direction: Azimuth {estimated_azimuth}°, Elevation {estimated_elevation}°")

        audio_processing_queue.task_done()


# ================================================================================================================#
""" @brief  Function to handle writing data to the file
    @param  deviceType - either master or slave device
    @retval none
"""


def write_data(file, deviceType):
    global doneReceivingAudio
    global audio_data_size
    try:
        while True:
            with audioBufferLock[deviceType]:
                # Wait for a buffer that has data to be written
                while (
                        not any(audioBufferStates[deviceType])
                        and not doneReceivingAudio[deviceType]
                ):
                    audioBufferReady[deviceType].wait()

                # If done receiving and no data left to write, exit the loop
                if doneReceivingAudio[deviceType] and not any(
                        audioBufferStates[deviceType]
                ):
                    break

                # Determine which buffer has data to be written
                audioBufferIndex = audioBufferStates[deviceType].index(True)

                # Write the data from the selected buffer to the file
                data_to_write = audioBuffers[deviceType][audioBufferIndex][
                                : audioBufferDataSizes[deviceType][audioBufferIndex]
                                ]
                file.write(data_to_write)
                file.flush()

                # Update the total audio data size
                audio_data_size[deviceType] += len(data_to_write)

                # Mark the buffer as available again
                audioBufferStates[deviceType][audioBufferIndex] = False

                # Notify the receiving thread that the buffer is free
                audioBufferReady[deviceType].notify_all()

    except Exception as e:
        print(f"An error occurred while writing data: {e}")


# ================================================================================================================#
""" @brief Function to write the WAV header
"""


def write_wav_header(file, num_channels, audioSamplingRate, bits_per_sample):
    byte_rate = audioSamplingRate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Write the WAV header
    file.write(b"RIFF")  # Chunk ID
    file.write(struct.pack("<I", 0))  # Chunk size (to be filled later)
    file.write(b"WAVE")  # Format
    file.write(b"fmt ")  # Subchunk1 ID
    file.write(struct.pack("<I", 16))  # Subchunk1 size (16 for PCM)
    file.write(struct.pack("<H", 1))  # Audio format (1 for PCM)
    file.write(struct.pack("<H", num_channels))  # Number of channels
    file.write(struct.pack("<I", audioSamplingRate))  # Sample rate
    file.write(struct.pack("<I", byte_rate))  # Byte rate
    file.write(struct.pack("<H", block_align))  # Block align
    file.write(struct.pack("<H", bits_per_sample))  # Bits per sample
    file.write(b"data")  # Subchunk2 ID
    file.write(struct.pack("<I", 0))  # Subchunk2 size (to be filled later)


# ================================================================================================================#
""" @brief Function that updates the wav header with the correct size
"""


def update_wav_header(file, deviceType):
    file.seek(4)
    # Update Chunk size
    chunk_size = 36 + audio_data_size[deviceType]
    file.write(struct.pack("<I", chunk_size))

    file.seek(40)
    # Update Subchunk2 size
    file.write(struct.pack("<I", audio_data_size[deviceType]))


# ================================================================================================================#
""" @brief Function to compile audio data into a wav file
    @param file_number: keeps track of what number the file is
"""


def compileAudioDataIntoWavFile(temp, file_number, file_name, deviceType):
    # Generate the filename based on the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time}_File{file_number}_{file_name}_device.wav"

    # Open a file to save the received audio data
    with open(filename, "wb") as audio_file:
        print(f"Receiving audio data and saving as '{filename}'...\n")

        # Write the WAV header
        write_wav_header(audio_file, num_channels, audioSamplingRate, bits_per_sample)

        # Create and start threads for receiving and writing data
        receive_thread = threading.Thread(
            target=receiveAudioData, args=(temp, deviceType)
        )
        write_thread = threading.Thread(
            target=write_data, args=(audio_file, deviceType)
        )

        receive_thread.start()
        write_thread.start()

        # Wait for both threads to finish
        receive_thread.join()
        write_thread.join()

        # Update the WAV file header with correct sizes
        audio_file.seek(0)
        update_wav_header(audio_file, deviceType)

        print(f"Audio data received and saved as '{filename}'.")

    return filename


# ================================================================================================================#
""" @brief Function to sends the appropriate sampling rate
    @param tempAudioSamplingRate          : audio sampling rate set
    @retval audioSamplingRateCommandValue : 1 - 16KHz
                                          : 2 - 24KHz
                                          : 3 - 32KHz
                                          : 4 - 48KHz
                                          : 5 - 96KHz
                                          : 6 - 192KHz
                                          : 7 - 384KHz
                                          : 8 - 768KHz
"""


def sendSamplingRateCommand(tempAudioSamplingRate):
    if tempAudioSamplingRate == 16000:  # 16KHz
        audioSamplingRateCommandValue = 1
    elif tempAudioSamplingRate == 24000:  # 24 KHz
        audioSamplingRateCommandValue = 2
    elif tempAudioSamplingRate == 32000:  # 32KHz
        audioSamplingRateCommandValue = 3
    elif tempAudioSamplingRate == 48000:  # 48KHz
        audioSamplingRateCommandValue = 4
    elif tempAudioSamplingRate == 96000:  # 96KHz
        audioSamplingRateCommandValue = 5
    elif tempAudioSamplingRate == 192000:  # 192KHz
        audioSamplingRateCommandValue = 6
    elif tempAudioSamplingRate == 384000:  # 384KHz
        audioSamplingRateCommandValue = 7
    elif tempAudioSamplingRate == 768000:  # 768KHz
        audioSamplingRateCommandValue = 8
    else:
        print("Invalid sampling rate")
    return audioSamplingRateCommandValue


# ================================================================================================================#
"""   @brief  Function that the audio digital volume
      @param  - tempAudioDigitalVolume
      @retval - audioDigitalVolumeCommandValue  : 1 - muted
                                                : 2 - -100dB
                                                : 3 - -99.5dB   
                                                : 4 - 0dB
                                                : 5 - 0.5dB
                                                : 6 - 26.5dB
                                                : 7 - 27dB

      @note The audio digital volume is an internal codec setting
"""


def sendAudioDigitalVolumeCommand(tempAudioDigitalVolume):
    if tempAudioDigitalVolume == 1:  # muted
        audioDigitalVolumeCommandValue = 1
    elif tempAudioDigitalVolume == -100:  # -100dB
        audioDigitalVolumeCommandValue = 2
    elif tempAudioDigitalVolume == -99.5:  # -99.5dB
        audioDigitalVolumeCommandValue = 3
    elif tempAudioDigitalVolume == 0:  # 0dB
        audioDigitalVolumeCommandValue = 4
    elif tempAudioDigitalVolume == 0.5:  # 0.5dB
        audioDigitalVolumeCommandValue = 5
    elif tempAudioDigitalVolume == 26.5:  # 26.5dB
        audioDigitalVolumeCommandValue = 6
    elif tempAudioDigitalVolume == 27.5:  # 26.5dB
        audioDigitalVolumeCommandValue = 7
    else:
        print("Invalid digital volume")
    return audioDigitalVolumeCommandValue


# ================================================================================================================#
""" @brief Function that sets the audio high pass filter
    @retval audioHighPassFilterValue  : 1 - default  
                                      : 2 - 0.00025fs
                                      : 3 - 0.002fs  
                                      : 4 - 0.008fs  
    @note The audio high pass filter is an internal codec setting
"""


def sendAudioHighPassFilterCommand(tempAudioHighPassFilter):
    if tempAudioHighPassFilter == 1:  # default
        audioHighPassFilterCommandValue = 1
    elif tempAudioHighPassFilter == 0.00025:  # 0.00025fs
        audioHighPassFilterCommandValue = 2
    elif tempAudioHighPassFilter == 0.002:  # 0.002fs
        audioHighPassFilterCommandValue = 3
    elif tempAudioHighPassFilter == 0.008:  # 0.008fs
        audioHighPassFilterCommandValue = 4
    else:
        print("Invalid High Pass Filter setting")
    return audioHighPassFilterCommandValue


# ================================================================================================================#
""" @brief Function that sets the audio gain calibration
    @retval audioGainCalibrationCommandValue  : 1 - -0.8
                                              : 2 - -0.7
                                              : 3 - -0.6
                                              : 4 - 0
                                              : 5 - 0.1
                                              : 6 - 0.6
                                              : 7 - 0.7
    @note The audio gain calibration is an internal codec setting
"""


def sendAudioGainCalibrationCommand(tempAudioGainCalibration):
    if tempAudioGainCalibration == -0.8:
        audioGainCalibrationCommandValue = 1
    elif tempAudioGainCalibration == -0.7:
        audioGainCalibrationCommandValue = 2
    elif tempAudioGainCalibration == -0.6:
        audioGainCalibrationCommandValue = 3
    elif tempAudioGainCalibration == 0:
        audioGainCalibrationCommandValue = 4
    elif tempAudioGainCalibration == 0.1:
        audioGainCalibrationCommandValue = 5
    elif tempAudioGainCalibration == 0.6:
        audioGainCalibrationCommandValue = 6
    elif tempAudioGainCalibration == 0.7:
        audioGainCalibrationCommandValue = 7
    else:
        print("Invalid Audio gain Filter setting")
    return audioGainCalibrationCommandValue


# ================================================================================================================#
#                                          SENSOR FUNCTIONS END                                                   #
# ----------------------------------------------------------------------------------------------------------------#
"""@brief receives sensor data
   @param sensorTemp        - the socket that sensors will use to receive data
   @param sensorsFileNumber - the file number being downloaded
   @param deviceName        - name of the device, master device or slave device
   @retval                  - none
"""


def receiveSensorData(sensorTemp, sensorsFileNumber, deviceName):
    # CSV file names
    currentTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    accelerometer_file = f"{currentTime}_accelerometer_File_{sensorsFileNumber}_{deviceName}_device.csv"
    gyroscope_file = f"{currentTime}_gyroscope_File_{sensorsFileNumber}_{deviceName}_device.csv"
    magnetometer_file = f"{currentTime}_magnetometer_File_{sensorsFileNumber}_{deviceName}_device.csv"
    temperature_file = f"{currentTime}_temperature_File_{sensorsFileNumber}_{deviceName}_device.csv"

    # Open the files for writing
    with open(accelerometer_file, mode="w", newline="") as acc_file, open(gyroscope_file, mode="w",
                                                                          newline="") as gyro_file, open(
            magnetometer_file, mode="w", newline="") as mag_file, open(temperature_file, mode="w",
                                                                       newline="") as temp_file:
        # Create CSV writers
        acc_writer = csv.writer(acc_file)
        gyro_writer = csv.writer(gyro_file)
        mag_writer = csv.writer(mag_file)
        temp_writer = csv.writer(temp_file)

        # Write headers
        acc_writer.writerow(["accX", "accY", "accZ"])
        gyro_writer.writerow(["gyroX", "gyroY", "gyroZ"])
        mag_writer.writerow(["magX", "magY", "magZ"])
        temp_writer.writerow(["tempDegC"])

        try:
            buffer = ""  # Initialize an empty buffer
            while True:
                # Receive data from the ESP32
                chunk = sensorTemp.recv(4096)
                if not chunk:  # No more data
                    break

                # Add the received chunk to the buffer
                buffer += chunk.decode("utf-8")

                # Process complete JSON objects in the buffer
                while True:
                    try:
                        # Attempt to parse the first JSON object in the buffer
                        sensor_data_list, index = json.JSONDecoder().raw_decode(buffer)
                        # Remove the processed JSON from the buffer
                        buffer = buffer[index:].strip()

                        # Ensure sensor_data_list is a list
                        if isinstance(sensor_data_list, list):
                            for sensor_data in sensor_data_list:
                                # Write only relevant data to each CSV file
                                if "accX" in sensor_data and "accY" in sensor_data and "accZ" in sensor_data:
                                    acc_writer.writerow(
                                        [sensor_data["accX"], sensor_data["accY"], sensor_data["accZ"], ])

                                if "gyroX" in sensor_data and "gyroY" in sensor_data and "gyroZ" in sensor_data:
                                    gyro_writer.writerow(
                                        [sensor_data["gyroX"], sensor_data["gyroY"], sensor_data["gyroZ"], ])

                                if ("magX" in sensor_data and "magY" in sensor_data and "magZ" in sensor_data):
                                    mag_writer.writerow(
                                        [sensor_data["magX"], sensor_data["magY"], sensor_data["magZ"], ])

                                if "tempDegC" in sensor_data:
                                    temp_writer.writerow([sensor_data["tempDegC"]])

                        else:
                            print("Unexpected data format: JSON object is not a list")
                            break

                    except json.JSONDecodeError:
                        # Incomplete JSON object, wait for more data
                        break

            print(
                f"Data successfully written to:\n  {accelerometer_file}\n  {gyroscope_file}\n  {magnetometer_file}\n  {temperature_file}")
        except Exception as e:
            print(f"An error occurred: {e}")
        # finally:
        #     sensorTemp.close()


# ================================================================================================================#
def accelDataRateValue(accelDataRateCommand):
    if accelDataRateCommand == 0:  # SHUTDOWN
        accelDataRateCommandValue = 1
    elif accelDataRateCommand == 12.5:  # 12.5Hz
        accelDataRateCommandValue = 2
    elif accelDataRateCommand == 26:  # 26Hz
        accelDataRateCommandValue = 3
    elif accelDataRateCommand == 52:  # 52Hz
        accelDataRateCommandValue = 4
    elif accelDataRateCommand == 104:  # 104Hz
        accelDataRateCommandValue = 5
    elif accelDataRateCommand == 208:  # 208Hz
        accelDataRateCommandValue = 6
    elif accelDataRateCommand == 416:  # 416Hz
        accelDataRateCommandValue = 7
    elif accelDataRateCommand == 833:  # 833Hz
        accelDataRateCommandValue = 8
    elif accelDataRateCommand == 1.66:  # 1.66KHz
        accelDataRateCommandValue = 9
    elif accelDataRateCommand == 3.33:  # 3.33KHz
        accelDataRateCommandValue = 10
    elif accelDataRateCommand == 6.66:  # 6.66KHz
        accelDataRateCommandValue = 11
    else:
        print("Invalid Input!")
    return accelDataRateCommandValue


# ================================================================================================================#
def accelDataRangeValue(accelDataRangeCommand):
    if accelDataRangeCommand == 2:  # 2G
        accelDataRateCommandValue = 1
    elif accelDataRangeCommand == 4:  # 4G
        accelDataRateCommandValue = 2
    elif accelDataRangeCommand == 8:  # 8G
        accelDataRateCommandValue = 3
    elif accelDataRangeCommand == 16:  # 16G
        accelDataRateCommandValue = 4
    else:
        print("Invalid Input!")
    return accelDataRateCommandValue


# ================================================================================================================#
def gyroDataRateValue(gyroDataRateCommand):
    if gyroDataRateCommand == 0:  # SHUTDOWN
        gyroDataRateCommandValue = 1
    elif gyroDataRateCommand == 12.5:  # 12.5Hz
        gyroDataRateCommandValue = 2
    elif gyroDataRateCommand == 26:  # 26Hz
        gyroDataRateCommandValue = 3
    elif gyroDataRateCommand == 52:  # 52Hz
        gyroDataRateCommandValue = 4
    elif gyroDataRateCommand == 104:  # 104Hz
        gyroDataRateCommandValue = 5
    elif gyroDataRateCommand == 208:  # 208Hz
        gyroDataRateCommandValue = 6
    elif gyroDataRateCommand == 416:  # 416Hz
        gyroDataRateCommandValue = 7
    elif gyroDataRateCommand == 833:  # 833Hz
        gyroDataRateCommandValue = 8
    elif gyroDataRateCommand == 1.66:  # 1.66KHz
        gyroDataRateCommandValue = 9
    elif gyroDataRateCommand == 3.33:  # 3.33KHz
        gyroDataRateCommandValue = 10
    elif gyroDataRateCommand == 6.66:  # 6.66KHz
        gyroDataRateCommandValue = 11
    else:
        print("Invalid Input!")
    return gyroDataRateCommandValue


# ================================================================================================================#
def gyroDataRangeValue(gyroDataRangeCommand):
    if gyroDataRangeCommand == 125:  # LSM6DS_GYRO_RANGE_125_DPS
        gyroDataRangeCommandValue = 1
    elif gyroDataRangeCommand == 250:  # LSM6DS_GYRO_RANGE_250_DPS
        gyroDataRangeCommandValue = 2
    elif gyroDataRangeCommand == 500:  # LSM6DS_GYRO_RANGE_500_DPS
        gyroDataRangeCommandValue = 3
    elif gyroDataRangeCommand == 1000:  # LSM6DS_GYRO_RANGE_1000_DPS
        gyroDataRangeCommandValue = 4
    elif gyroDataRangeCommand == 2000:  # LSM6DS_GYRO_RANGE_2000_DPS
        gyroDataRangeCommandValue = 5
    elif gyroDataRangeCommand == 4000:  # ISM330DHCX_GYRO_RANGE_4000_DPS
        gyroDataRangeCommandValue = 6
    else:
        print("Invalid Input!")
    return gyroDataRangeCommandValue


# ================================================================================================================#
#                                                USER INTERFACE                                                   #
# ----------------------------------------------------------------------------------------------------------------#
""" @brief UI Main menu function displays the main menu
    @retval : mainMenuFunctionToPerform    : 1 - record data
                                           : 2 - access sd card

            : recordingMethodValue         : 1 - Ethernet 
                                           : 2 - WiFi 
                                           : 3 - SD Card 
                                           : 4 - SD Card & Download Through WiFi

            : dataToRecordInputValue       : 1. Record Audio Data 
                                           : 2 - Record Sensors Data 
                                           : 3 - Record Both Audio & Sensor Data

            : sdCardFunctionToPerformValue : 1 - List files 
                                           : 2 - Download File 
                                           : 3 - Upload file 
                                           : 4 - Delete File 
                                           : 5 - Exit
"""


def mainMenu():
    # main Menu
    while True:
        print(
            "----------------------------------\n         Sound Sentinel\n----------------------------------"
        )
        # Prompt the user to enter setting
        mainMenuFunctionToPerform = input(
            "Select function to perform: \n1. Record Data \n2. Access SD card\n"
        )
        if mainMenuFunctionToPerform == "1":
            sdCardFunctionToPerformValue = 0  # initialize
            recordingMethodValue, dataToRecordInputValue = recordingMethod()
            break
        elif mainMenuFunctionToPerform == "2":
            recordingMethodValue = 0  # initialize
            dataToRecordInputValue = 0  # init
            sdCardFunctionToPerformValue = sdCardFunctionToPerform()
            break
        else:
            print("Invalid Selection")
    return (
        mainMenuFunctionToPerform,
        recordingMethodValue,
        dataToRecordInputValue,
        sdCardFunctionToPerformValue,
    )


# ================================================================================================================#
""" @brief selects the recording method to be used when recording data
    @retval : recordingMethodInput : 1. Ethernet 
                                   : 2 - WiFi 
                                   : 3 - SD Card 
                                   : 4 - SD Card & Download Through WiFi

            : dataToRecordValue    : 1 - Record Audio Data 
                                   : 2 - Record Sensors Data 
                                   : 3 - Record Both Audio and Sensor Data

"""


def recordingMethod():
    while True:
        recordingMethodInput = input(
            "Select Recording Method:\n1. Ethernet \n2. WiFi \n3. SD Card \n4. SD Card & Download Through WiFi\n"
        )
        if (
                recordingMethodInput == "1"
                or recordingMethodInput == "2"
                or recordingMethodInput == "3"
                or recordingMethodInput == "4"
        ):
            dataToRecordValue = dataToRecord()
            break
        else:
            print("Invalid Input!!!")
    return recordingMethodInput, dataToRecordValue


# ================================================================================================================#
""" @brief takes input of what data to record
    @retval dataToRecordInput 1 - Record audio data
                              2 - Record sensor data
                              3 - Record both audio and sensor data
"""


def dataToRecord():
    dataToRecordInput = input(
        "Select data to record:\n1. Record Audio Data\n2. Record Sensors Data \n3. Record Both Audio & Sensor Data\n"
    )
    return dataToRecordInput


# ================================================================================================================#
""" @brief takes input of what sd card function to perform
    @retval : sdCardFunctionToPerformInput : 1 - List files 
                                           : 2 - Download File 
                                           : 3 - Upload file 
                                           : 4 - Delete File 
                                           : 5 - Exit
"""


def sdCardFunctionToPerform():
    sdCardFunctionToPerformInput = input(
        "Select SD Card Function To Perform\n1. List files \n2. Download File \n3. Download all files \n4. Delete all files"
    )
    return sdCardFunctionToPerformInput


# ================================================================================================================#
""" @brief Records audio data over network, either wifi or ethernet
    @param masterDeviceNetwork can be either masterDeviceEthernet or masterDeviceWiFi
    @parma slaveDeviceNetwork can be either slaveDeviceEthernet or slaveDeviceWiFi
"""


def recordAudioOverNetwork(masterDeviceNetworkForAudio, slaveDeviceNetworkForAudio, slaveDeviceNetworkForAudioDevNum2,
                           numberOfAudioFilesToBeRecorded, masterDeviceIPForAudio, slaveDeviceIPForAudio,
                           slaveDeviceIPForAudioDevNum2):
    currentTimeOfSendingCommand = datetime.now().strftime("%Y%m%d_%H%M%S")
    generateJson(masterDeviceNetworkForAudio, slaveDeviceNetworkForAudio, slaveDeviceNetworkForAudioDevNum2,
                 masterDeviceIPForAudio, slaveDeviceIPForAudio, slaveDeviceIPForAudioDevNum2, placeHolder,
                 currentTimeOfSendingCommand)
    print("Recording Audio Over Network!")
    # for i in range(1, numberOfFilesToRecord + 1):
    # Reset state variables for each file

    doneReceivingAudio[0] = False
    doneReceivingAudio[1] = False
    doneReceivingAudio[2] = False
    total_samples[0] = 0
    total_samples[1] = 0
    total_samples[2] = 0
    audio_data_size[0] = 0
    audio_data_size[1] = 0
    audio_data_size[2] = 0

    if numberOfDevicesConnected == 1:  # master device only
        # create thread to send audio
        threadTocompileAudioDataIntoWavFileMasterDevice = threading.Thread(target=compileAudioDataIntoWavFile, args=(
        masterDeviceNetworkForAudio, numberOfAudioFilesToBeRecorded, "Master", 0), )
        threadTocompileAudioDataIntoWavFileMasterDevice.start()
        threadTocompileAudioDataIntoWavFileMasterDevice.join()
        # masterDeviceNetworkForAudio.close()
    elif numberOfDevicesConnected == 2:  # master device and slave device
        # create thread to send audio
        threadTocompileAudioDataIntoWavFileMasterDevice = threading.Thread(target=compileAudioDataIntoWavFile, args=(
        masterDeviceNetworkForAudio, numberOfAudioFilesToBeRecorded, "Master", 0), )
        threadTocompileAudioDataIntoWavFileSlaveDevice = threading.Thread(target=compileAudioDataIntoWavFile, args=(
        slaveDeviceNetworkForAudio, numberOfAudioFilesToBeRecorded, "Slave", 1), )
        threadTocompileAudioDataIntoWavFileSlaveDevice.start()
        threadTocompileAudioDataIntoWavFileMasterDevice.start()
        threadTocompileAudioDataIntoWavFileSlaveDevice.join()
        threadTocompileAudioDataIntoWavFileMasterDevice.join()
        # masterDeviceNetworkForAudio.close()
        # slaveDeviceNetworkForAudio.close()
    else:  # master device and slave device
        # create thread to send audio
        threadTocompileAudioDataIntoWavFileMasterDevice = threading.Thread(target=compileAudioDataIntoWavFile, args=(
        masterDeviceNetworkForAudio, numberOfAudioFilesToBeRecorded, "Master", 0), )
        threadTocompileAudioDataIntoWavFileSlaveDevice = threading.Thread(target=compileAudioDataIntoWavFile, args=(
        slaveDeviceNetworkForAudio, numberOfAudioFilesToBeRecorded, "Slave", 1), )
        threadTocompileAudioDataIntoWavFileSlaveDeviceNum2 = threading.Thread(target=compileAudioDataIntoWavFile, args=(
        slaveDeviceNetworkForAudioDevNum2, numberOfAudioFilesToBeRecorded, "Slave_device_number_2", 2), )
        threadTocompileAudioDataIntoWavFileSlaveDevice.start()
        threadTocompileAudioDataIntoWavFileSlaveDeviceNum2.start()
        threadTocompileAudioDataIntoWavFileMasterDevice.start()
        threadTocompileAudioDataIntoWavFileSlaveDevice.join()
        threadTocompileAudioDataIntoWavFileSlaveDeviceNum2.join()
        threadTocompileAudioDataIntoWavFileMasterDevice.join()
        # masterDeviceNetworkForAudio.close()
        # slaveDeviceNetworkForAudio.close()


# ================================================================================================================#
""" @brief Records sensor data over network, either wifi or ethernet
    @param masterDeviceNetwork can be either masterDeviceEthernet or masterDeviceWiFi
    @parma slaveDeviceNetwork can be either slaveDeviceEthernet or slaveDeviceWiFi
"""


def recordSensorDataOverNetwork(masterDeviceNetworkForSensor, slaveDeviceNetworkForSensor,
                                slaveDeviceNetworkForSensorDevNum2, numberOfSensorFilesToBeRecorded,
                                masterDeviceIPForSensors, slaveDeviceIPForSensors, slaveDeviceIPForSensorsDevNum2):
    currentTimeOfSendingCommand = datetime.now().strftime("%Y%m%d_%H%M%S")
    generateJson(masterDeviceNetworkForSensor, slaveDeviceNetworkForSensor, slaveDeviceNetworkForSensorDevNum2,
                 masterDeviceIPForSensors, slaveDeviceIPForSensors, slaveDeviceIPForSensorsDevNum2, placeHolder,
                 currentTimeOfSendingCommand)
    if numberOfDevicesConnected == 1:  # master device only
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice = threading.Thread(target=receiveSensorData, args=(
        masterDeviceNetworkForSensor, numberOfSensorFilesToBeRecorded, "Master",), )
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice.start()
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice.join()
        masterDeviceNetworkForSensor.close()

    elif numberOfDevicesConnected == 2:  # Device 0(Master Device), Device 1
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice = threading.Thread(target=receiveSensorData, args=(
        masterDeviceNetworkForSensor, numberOfSensorFilesToBeRecorded, "Master",), )
        threadToReceiveSensorDataAndRecordToCSVonSlaveDevice = threading.Thread(target=receiveSensorData, args=(
        slaveDeviceNetworkForSensor, numberOfSensorFilesToBeRecorded, "Slave",), )
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice.start()
        threadToReceiveSensorDataAndRecordToCSVonSlaveDevice.start()
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice.join()
        threadToReceiveSensorDataAndRecordToCSVonSlaveDevice.join()
        masterDeviceNetworkForSensor.close()
        slaveDeviceNetworkForSensor.close()

    else:  # Device 0(Master Device), Device 1, Device 2
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice = threading.Thread(target=receiveSensorData, args=(
        masterDeviceNetworkForSensor, numberOfSensorFilesToBeRecorded, "Master",), )
        threadToReceiveSensorDataAndRecordToCSVonSlaveDevice = threading.Thread(target=receiveSensorData, args=(
        slaveDeviceNetworkForSensor, numberOfSensorFilesToBeRecorded, "Slave",), )
        threadToReceiveSensorDataAndRecordToCSVonSlaveDeviceNum2 = threading.Thread(target=receiveSensorData, args=(
        slaveDeviceNetworkForSensorDevNum2, numberOfSensorFilesToBeRecorded, "Slave_device_number_2",), )
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice.start()
        threadToReceiveSensorDataAndRecordToCSVonSlaveDevice.start()
        threadToReceiveSensorDataAndRecordToCSVonSlaveDeviceNum2.start()
        threadToReceiveSensorDataAndRecordToCSVonMasterDevice.join()
        threadToReceiveSensorDataAndRecordToCSVonSlaveDevice.join()
        threadToReceiveSensorDataAndRecordToCSVonSlaveDeviceNum2.join()
        masterDeviceNetworkForSensor.close()
        slaveDeviceNetworkForSensor.close()
        slaveDeviceNetworkForSensorDevNum2.close()


# ================================================================================================================#
def restartEthernetConnection(delayInSecondsEthernet):
    global masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2
    masterDeviceEthernet.close()
    slaveDeviceEthernet.close()
    slaveDeviceEthernetDeviceNumber2.close()
    time.sleep(delayInSecondsEthernet)
    masterDeviceEthernet = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Master device web socket
    slaveDeviceEthernet = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Slave device websocket
    slaveDeviceEthernetDeviceNumber2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Slave device websocket


# ================================================================================================================#
def restartWiFiConnection(delayInSecondsWiFi):
    global masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2
    masterDeviceWiFi.close()
    slaveDeviceWiFi.close()
    slaveDeviceWiFiDeviceNumber2.close()
    time.sleep(delayInSecondsWiFi)
    masterDeviceWiFi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Wifi device websocket
    slaveDeviceWiFi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Wifi device websocket
    slaveDeviceWiFiDeviceNumber2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Wifi device websocket


# ================================================================================================================#
""" @brief handles all data recording functions
    @retval none
"""


def recordData(tempSelectedRecordingMethodValue, tempSelectedDataToRecordValue):
    global selectedDataToRecordValue, masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, selectedRecordingMethodValue
    currentTimeOfSendingCommandForSDrecording = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Recording Data!")
    # ======================================================================================================================#
    #                                           Ethernet Data Recording                                                    #
    # ----------------------------------------------------------------------------------------------------------------------#
    if tempSelectedRecordingMethodValue == "1" and tempSelectedDataToRecordValue == "1":
        print("1. Record Data -> 1. Ethernet -> 1. Record Audio")
        for i in range(numberOfFilesToRecord):
            recordAudioOverNetwork(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, i,
                                   DEVICE_0_ETHERNET_IP, DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP)
            restartEthernetConnection(2)

    elif tempSelectedRecordingMethodValue == "1" and tempSelectedDataToRecordValue == "2":
        print("1. Record Data -> 1. Ethernet -> 2. Record Sensors")
        for i in range(numberOfFilesToRecord):
            recordSensorDataOverNetwork(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, i,
                                        DEVICE_0_ETHERNET_IP, DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP)
            restartEthernetConnection(2)

    elif tempSelectedRecordingMethodValue == "1" and tempSelectedDataToRecordValue == "3":
        print("1. Record Data -> 1. Ethernet -> 3. Record Both(Audio & Sensors)")
        for i in range(numberOfFilesToRecord):
            selectedDataToRecordValue = 1
            recordAudioOverNetwork(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, i,
                                   DEVICE_0_ETHERNET_IP, DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP)
            selectedDataToRecordValue = 2
            restartEthernetConnection(2)
            recordSensorDataOverNetwork(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, i,
                                        DEVICE_0_ETHERNET_IP, DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP)
            restartEthernetConnection(2)

    # ======================================================================================================================#
    #                                           WiFi Data Recording                                                         #
    # ----------------------------------------------------------------------------------------------------------------------#
    elif tempSelectedRecordingMethodValue == "2" and tempSelectedDataToRecordValue == "1":
        print("1. Record Data -> 2. WiFi -> 1. Record Audio")
        for i in range(numberOfFilesToRecord):
            recordAudioOverNetwork(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, i, DEVICE_0_WIFI_IP,
                                   DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP)
            restartWiFiConnection(2)

    elif tempSelectedRecordingMethodValue == "2" and tempSelectedDataToRecordValue == "2":
        print("1. Record Data -> 2. WiFi -> 2. Record Sensors")
        for i in range(numberOfFilesToRecord):
            recordSensorDataOverNetwork(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, i,
                                        DEVICE_0_WIFI_IP, DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP)
            restartWiFiConnection(2)

    elif tempSelectedRecordingMethodValue == "2" and tempSelectedDataToRecordValue == "3":
        print("1. Record Data -> 2. WiFi -> 3. Record Both(Audio & Sensors)")
        for i in range(numberOfFilesToRecord):
            selectedDataToRecordValue = 1
            recordAudioOverNetwork(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, i, DEVICE_0_WIFI_IP,
                                   DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP)
            selectedDataToRecordValue = 2
            restartWiFiConnection(2)
            recordSensorDataOverNetwork(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, i,
                                        DEVICE_0_WIFI_IP, DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP)
            restartWiFiConnection(2)
    # ======================================================================================================================#
    #                                           SD Card Data Recording                                                      #
    # ----------------------------------------------------------------------------------------------------------------------#
    elif tempSelectedRecordingMethodValue == "3" and tempSelectedDataToRecordValue == "1":
        print("1. Record Data -> 3. SD Card -> 1. Record Audio")
        generateJson(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, DEVICE_0_ETHERNET_IP,
                     DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP, placeHolder, currentTimeOfSendingCommandForSDrecording)

    elif tempSelectedRecordingMethodValue == "3" and tempSelectedDataToRecordValue == "2":
        print("1. Record Data -> 3. SD Card -> 2. Record Sensors")
        generateJson(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, DEVICE_0_ETHERNET_IP,
                     DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP, placeHolder, currentTimeOfSendingCommandForSDrecording)

    elif tempSelectedRecordingMethodValue == "3" and tempSelectedDataToRecordValue == "3":
        print("1. Record Data -> 3. SD Card -> 3. Record Both(Audio & Sensors)")
        generateJson(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2, DEVICE_0_ETHERNET_IP,
                     DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP, placeHolder, currentTimeOfSendingCommandForSDrecording)

    # ======================================================================================================================#
    #                                       SD Card Data and Download Recording                                             #
    # ----------------------------------------------------------------------------------------------------------------------#
    elif tempSelectedRecordingMethodValue == "4" and tempSelectedDataToRecordValue == "1":
        print("1. Record Data -> 3. SD Card Data & Download -> 1. Record Audio")
        # Function to record audio data to sd card and download
        currentTimeOfSendingCommand = datetime.now().strftime("%Y%m%d_%H%M%S")
        if numberOfDevicesConnected == 1:  # master device only
            for i in range(numberOfFilesToRecord):  # Record Audio to SD card
                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 1  # records audio to sd
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)

            for i in range(numberOfFilesToRecord):  # Download recorded audio on SD card
                fileNumberToDownloadString = str(i)
                print("Downloading Audio File Number " + fileNumberToDownloadString)

                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 3  # Download audio
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "master_device",), )

                # start thread
                threadToautodownloadAudioOnSDcardMasterDevice.start()

                # join thread
                threadToautodownloadAudioOnSDcardMasterDevice.join()
                # time.sleep(10)  # wait for download to finish
                restartWiFiConnection(2)

        elif numberOfDevicesConnected == 2:  # Device 0(Master Device), Device 1
            for i in range(numberOfFilesToRecord):  # Download recorded audio on SD card
                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 1  # records audio to sd
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)

            for i in range(numberOfFilesToRecord):  # Download recorded audio on SD card
                fileNumberToDownloadString = str(i)
                print("Downloading Audio File Number " + fileNumberToDownloadString)

                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 3  # Download audio
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "master_device",), )
                threadToautodownloadAudioOnSDcardSlaveDevice = threading.Thread(target=autodownloadAudioOnSDcard, args=(
                slaveDeviceWiFi, fileNumberToDownloadString, "slave_device",), )

                # start thread
                threadToautodownloadAudioOnSDcardMasterDevice.start()
                threadToautodownloadAudioOnSDcardSlaveDevice.start()

                # join thread
                threadToautodownloadAudioOnSDcardMasterDevice.join()
                threadToautodownloadAudioOnSDcardSlaveDevice.join()
                # time.sleep(10)  # wait for download to finish
                restartWiFiConnection(2)


        else:  # Device 0(Master Device), Device 1, Device 2
            for i in range(numberOfFilesToRecord):  # Download recorded audio on SD card
                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 1  # records audio to sd
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)

            for i in range(numberOfFilesToRecord):  # Download recorded audio on SD card
                fileNumberToDownloadString = str(i)
                print("Downloading Audio File Number " + fileNumberToDownloadString)

                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 3  # Download audio
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "master_device",), )
                threadToautodownloadAudioOnSDcardSlaveDevice = threading.Thread(target=autodownloadAudioOnSDcard, args=(
                slaveDeviceWiFi, fileNumberToDownloadString, "slave_device",), )
                threadToautodownloadAudioOnSDcardSlaveDeviceNum2 = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                    args=(slaveDeviceWiFiDeviceNumber2,
                                                                                          fileNumberToDownloadString,
                                                                                          "slave_device_number_2",), )

                # start thread
                threadToautodownloadAudioOnSDcardMasterDevice.start()
                threadToautodownloadAudioOnSDcardSlaveDevice.start()
                threadToautodownloadAudioOnSDcardSlaveDeviceNum2.start()

                # join thread
                threadToautodownloadAudioOnSDcardMasterDevice.join()
                threadToautodownloadAudioOnSDcardSlaveDevice.join()
                threadToautodownloadAudioOnSDcardSlaveDeviceNum2.join()
                # time.sleep(10)  # wait for download to finish
                restartWiFiConnection(2)

    elif tempSelectedRecordingMethodValue == "4" and tempSelectedDataToRecordValue == "2":
        print("1. Record Data -> 3. SD Card Data & Download -> 2. Record Sensors")
        currentTimeOfSendingCommand = datetime.now().strftime("%Y%m%d_%H%M%S")
        if numberOfDevicesConnected == 1:  # master device only
            for i in range(numberOfFilesToRecord):  # Record data to sd card
                selectedRecordingMethodValue = 4  # records to sd
                selectedDataToRecordValue = 2  # records sensor data to SD
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)
                time.sleep(2)

            for i in range(numberOfFilesToRecord):  # download data recorded
                print(fileNumberToDownloadString)
                fileNumberToDownloadString = str(i)

                restartWiFiConnection(2)
                selectedDataToRecordValue = 4  # download accelerometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAccelerometerDataOnSDcard = threading.Thread(target=autodownloadSensorDataOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "accelerometer",
                                                                                       "master_device",), )

                # start thread
                threadToautodownloadAccelerometerDataOnSDcard.start()

                # join thread
                threadToautodownloadAccelerometerDataOnSDcard.join()

                restartWiFiConnection(2)
                selectedDataToRecordValue = 5  # download gyroscope data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadGyroscopeDataOnSDcard = threading.Thread(target=autodownloadSensorDataOnSDcard,
                                                                             args=(masterDeviceWiFi,
                                                                                   fileNumberToDownloadString,
                                                                                   "gyroscope", "master_device",), )

                # start thread
                threadToautodownloadGyroscopeDataOnSDcard.start()

                # join thread
                threadToautodownloadGyroscopeDataOnSDcard.join()

                restartWiFiConnection(2)
                selectedDataToRecordValue = 6  # download magnetometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadMagnetometerDataOnSDcard = threading.Thread(target=autodownloadSensorDataOnSDcard,
                                                                                args=(masterDeviceWiFi,
                                                                                      fileNumberToDownloadString,
                                                                                      "magnetometer",
                                                                                      "master_device",), )

                # start thread
                threadToautodownloadMagnetometerDataOnSDcard.start()

                # join thread
                threadToautodownloadMagnetometerDataOnSDcard.join()

                restartWiFiConnection(2)
                selectedDataToRecordValue = 7  # download temperature data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadTemperatureDataOnSDcard = threading.Thread(target=autodownloadSensorDataOnSDcard,
                                                                               args=(masterDeviceWiFi,
                                                                                     fileNumberToDownloadString,
                                                                                     "temperature", "master_device",), )

                # start thread
                threadToautodownloadTemperatureDataOnSDcard.start()

                # join thread
                threadToautodownloadTemperatureDataOnSDcard.join()

                restartWiFiConnection(2)

        elif numberOfDevicesConnected == 2:  # multiple devices
            for i in range(numberOfFilesToRecord):  # Record data to sd card
                selectedRecordingMethodValue = 4  # records to sd
                selectedDataToRecordValue = 2  # records sensor data to SD
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)
                time.sleep(2)

            for i in range(numberOfFilesToRecord):  # download data recorded
                print(fileNumberToDownloadString)
                restartWiFiConnection(2)

                selectedDataToRecordValue = 4  # download accelerometer data
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "accelerometer", "master_device",), )
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "accelerometer", "slave_device",), )

                # start threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.start()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.join()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 5  # download gyroscope data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "gyroscope", "master_device",), )
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "gyroscope", "slave_device",), )

                # start threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.start()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.join()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 6  # download magnetometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "magnetometer", "master_device",), )
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "magnetometer", "slave_device",), )

                # start start threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.start()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.join()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 7  # download temperature data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "temperature", "master_device",), )
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "temperature", "slave_device",), )

                # start threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.start()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.join()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)



        else:  # multiple devices
            for i in range(numberOfFilesToRecord):  # Record data to sd card
                selectedRecordingMethodValue = 4  # records to sd
                selectedDataToRecordValue = 2  # records sensor data to SD
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)
                time.sleep(2)

            for i in range(numberOfFilesToRecord):  # download data recorded
                print(fileNumberToDownloadString)
                restartWiFiConnection(2)

                selectedDataToRecordValue = 4  # download accelerometer data
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "accelerometer", "master_device",), )
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "accelerometer", "slave_device",), )
                threadToautodownloadAccelerometerDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "accelerometer",
                    "slave_device_number_2",), )

                # start threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.start()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.start()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.join()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.join()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 5  # download gyroscope data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "gyroscope", "master_device",), )
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "gyroscope", "slave_device",), )
                threadToautodownloadGyroscopeDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "gyroscope", "slave_device_number_2",), )

                # start threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.start()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.start()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.join()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.join()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 6  # download magnetometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "magnetometer", "master_device",), )
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "magnetometer", "slave_device",), )
                threadToautodownloadMagnetometerDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "magnetometer",
                    "slave_device_number_2",), )

                # start start threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.start()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.start()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.join()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.join()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 7  # download temperature data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "temperature", "master_device",), )
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "temperature", "slave_device",), )
                threadToautodownloadTemperatureDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "temperature",
                    "slave_device_number_2",), )

                # start threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.start()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.start()
                threadToautodownloadTemperatureDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.join()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.join()
                threadToautodownloadTemperatureDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)




    elif tempSelectedRecordingMethodValue == "4" and tempSelectedDataToRecordValue == "3":
        print("1. Record Data -> 3. SD Card Data & Download -> 3. Record Both(Audio & Sensors)")
        currentTimeOfSendingCommand = datetime.now().strftime("%Y%m%d_%H%M%S")
        if numberOfDevicesConnected == 1:  # master device only
            for i in range(numberOfFilesToRecord):  # Record data to sd card
                restartWiFiConnection(2)
                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 1  # records audio to sd
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)

                # selectedRecordingMethodValue = 4  # records to sd
                selectedDataToRecordValue = 2  # records sensor data to SD
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading

                time.sleep(2)

            for i in range(numberOfFilesToRecord):  # download data recorded
                restartWiFiConnection(2)
                fileNumberToDownloadString = str(i)
                print("Downloading Audio File Number " + fileNumberToDownloadString)

                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 3  # Download audio
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "master_device",), )
                # threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,args=(masterDeviceWiFi,fileNumberToDownloadString,"master_device",),)

                # start thread
                threadToautodownloadAudioOnSDcardMasterDevice.start()

                # join thread
                threadToautodownloadAudioOnSDcardMasterDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 4  # download accelerometer data
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "accelerometer", "master_device",), )

                # start threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.start()

                # join threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 5  # download gyroscope data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "gyroscope", "master_device",), )

                # start threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.start()

                # join threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 6  # download magnetometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "magnetometer", "master_device",), )

                # start start threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.start()

                # join threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 7  # download temperature data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "temperature", "master_device",), )

                # start threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.start()

                # join threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.join()
                # restartWiFiConnection(2)

        elif numberOfDevicesConnected == 2:  # multiple devices
            for i in range(numberOfFilesToRecord):  # Record data to sd card
                restartWiFiConnection(2)
                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 1  # records audio to sd
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)

                # selectedRecordingMethodValue = 4  # records to sd
                selectedDataToRecordValue = 2  # records sensor data to SD
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)
                time.sleep(2)

            for i in range(numberOfFilesToRecord):  # download data recorded
                fileNumberToDownloadString = str(i)
                print("Downloading Audio File Number " + fileNumberToDownloadString)

                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 3  # Download audio
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "master_device",), )
                threadToautodownloadAudioOnSDcardSlaveDevice = threading.Thread(target=autodownloadAudioOnSDcard, args=(
                slaveDeviceWiFi, fileNumberToDownloadString, "slave_device",), )

                # start thread
                threadToautodownloadAudioOnSDcardMasterDevice.start()
                threadToautodownloadAudioOnSDcardSlaveDevice.start()

                # join thread
                threadToautodownloadAudioOnSDcardMasterDevice.join()
                threadToautodownloadAudioOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 4  # download accelerometer data
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "accelerometer", "master_device",), )
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "accelerometer", "slave_device",), )

                # start threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.start()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.join()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 5  # download gyroscope data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "gyroscope", "master_device",), )
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "gyroscope", "slave_device",), )

                # start threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.start()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.join()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 6  # download magnetometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "magnetometer", "master_device",), )
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "magnetometer", "slave_device",), )

                # start start threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.start()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.join()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 7  # download temperature data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "temperature", "master_device",), )
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "temperature", "slave_device",), )

                # start threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.start()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.start()

                # join threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.join()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.join()
                restartWiFiConnection(2)



        else:  # multiple devices
            for i in range(numberOfFilesToRecord):  # Record data to sd card
                restartWiFiConnection(2)
                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 1  # records audio to sd
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)

                # selectedRecordingMethodValue = 4  # records to sd
                selectedDataToRecordValue = 2  # records sensor data to SD
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)
                time.sleep(RECORDING_DURATION)  # wait for sd card to finish recording audio before downloading
                restartWiFiConnection(2)
                time.sleep(2)

            for i in range(numberOfFilesToRecord):  # download data recorded
                fileNumberToDownloadString = str(i)
                print("Downloading Audio File Number " + fileNumberToDownloadString)

                selectedRecordingMethodValue = 4  # Go to SD Card Data and Download Recording
                selectedDataToRecordValue = 3  # Download audio
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create thread
                threadToautodownloadAudioOnSDcardMasterDevice = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                 args=(masterDeviceWiFi,
                                                                                       fileNumberToDownloadString,
                                                                                       "master_device",), )
                threadToautodownloadAudioOnSDcardSlaveDevice = threading.Thread(target=autodownloadAudioOnSDcard, args=(
                slaveDeviceWiFi, fileNumberToDownloadString, "slave_device",), )
                threadToautodownloadAudioOnSDcardSlaveDeviceNumber2 = threading.Thread(target=autodownloadAudioOnSDcard,
                                                                                       args=(
                                                                                       slaveDeviceWiFiDeviceNumber2,
                                                                                       fileNumberToDownloadString,
                                                                                       "slave_device_number_2",), )

                # start thread
                threadToautodownloadAudioOnSDcardMasterDevice.start()
                threadToautodownloadAudioOnSDcardSlaveDevice.start()
                threadToautodownloadAudioOnSDcardSlaveDeviceNumber2.start()

                # join thread
                threadToautodownloadAudioOnSDcardMasterDevice.join()
                threadToautodownloadAudioOnSDcardSlaveDevice.join()
                threadToautodownloadAudioOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 4  # download accelerometer data
                fileNumberToDownloadString = str(i)
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "accelerometer", "master_device",), )
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "accelerometer", "slave_device",), )
                threadToautodownloadAccelerometerDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "accelerometer",
                    "slave_device_number_2",), )

                # start threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.start()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.start()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadAccelerometerDataOnSDcardMasterDevice.join()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDevice.join()
                threadToautodownloadAccelerometerDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 5  # download gyroscope data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "gyroscope", "master_device",), )
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "gyroscope", "slave_device",), )
                threadToautodownloadGyroscopeDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "gyroscope", "slave_device_number_2",), )

                # start threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.start()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.start()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadGyroscopeDataOnSDcardMasterDevice.join()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDevice.join()
                threadToautodownloadGyroscopeDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 6  # download magnetometer data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "magnetometer", "master_device",), )
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "magnetometer", "slave_device",), )
                threadToautodownloadMagnetometerDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "magnetometer",
                    "slave_device_number_2",), )

                # start start threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.start()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.start()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadMagnetometerDataOnSDcardMasterDevice.join()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDevice.join()
                threadToautodownloadMagnetometerDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

                selectedDataToRecordValue = 7  # download temperature data
                generateJson(masterDeviceWiFi, slaveDeviceWiFi, slaveDeviceWiFiDeviceNumber2, DEVICE_0_WIFI_IP,
                             DEVICE_1_WIFI_IP, DEVICE_2_WIFI_IP, i, currentTimeOfSendingCommand)

                # create threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(masterDeviceWiFi, fileNumberToDownloadString, "temperature", "master_device",), )
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice = threading.Thread(
                    target=autodownloadSensorDataOnSDcard,
                    args=(slaveDeviceWiFi, fileNumberToDownloadString, "temperature", "slave_device",), )
                threadToautodownloadTemperatureDataOnSDcardSlaveDeviceNumber2 = threading.Thread(
                    target=autodownloadSensorDataOnSDcard, args=(
                    slaveDeviceWiFiDeviceNumber2, fileNumberToDownloadString, "temperature",
                    "slave_device_number_2",), )

                # start threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.start()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.start()
                threadToautodownloadTemperatureDataOnSDcardSlaveDeviceNumber2.start()

                # join threads
                threadToautodownloadTemperatureDataOnSDcardMasterDevice.join()
                threadToautodownloadTemperatureDataOnSDcardSlaveDevice.join()
                threadToautodownloadTemperatureDataOnSDcardSlaveDeviceNumber2.join()
                restartWiFiConnection(2)

            # ================================================================================================================#


def delete_all_files(clientToDeleteFiles, esp32_ip):
    try:
        # Set up a socket connection to the ESP32
        # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # clientToDeleteFiles.connect((esp32_ip, 5000))  # Connect to ESP32 server (IP and port)

        # Send command to delete all files
        # clientToDeleteFiles.sendall(b"DELETE_ALL_FILES\n")

        # Receive response from the server
        response = clientToDeleteFiles.recv(1024).decode('utf-8')
        print(response)

        clientToDeleteFiles.close()  # Close the connection
    except Exception as e:
        print(f"Error: {e}")


# ================================================================================================================#
def download_all_files(clientToDownloadAllFiles):
    # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect((ESP32_IP, PORT))
    clientToDownloadAllFiles.sendall(b"DOWNLOAD_ALL\n")

    while True:
        # Read the first line which contains the filename
        header = clientToDownloadAllFiles.recv(1024).decode().strip()
        if header.startswith("FILE:"):
            file_name = header[5:]  # Extract the filename
            file_name = file_name.replace("/", "_")  # Replace slashes for local saving

            print(f"Downloading: {file_name}")
            with open(file_name, "wb") as f:
                while True:
                    data = clientToDownloadAllFiles.recv(512)
                    if b"EOF" in data:
                        data = data.replace(b"EOF", b"")  # Remove EOF marker
                        f.write(data)
                        break
                    f.write(data)

        elif header == "DONE":
            print("All files downloaded successfully.")
            break
        else:
            print(header)  # Print any errors received

    clientToDownloadAllFiles.close()


# ================================================================================================================#
""" @brief handles all functions that access sd card"""


def accessSDcard(masterDeviceNetworkForSDaccess, slaveDeviceNetworkForSDaccess,
                 slaveDeviceNetworkForSDaccessDeviceNumber2, masterDeviceIPForSDaccess, slaveDeviceIPForSDaccess,
                 slaveDeviceIPForSDaccessDeviceNumber2, tempSelectedSDcardFunctionToPerformValue):
    print("Accessing SD card")
    timeOfAccessingSDcard = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tempSelectedSDcardFunctionToPerformValue == "1":
        # Function to list files on sd card
        print("2. Access SD Card -> 1. List files")
        generateJson(masterDeviceNetworkForSDaccess, slaveDeviceNetworkForSDaccess,
                     slaveDeviceNetworkForSDaccessDeviceNumber2, masterDeviceIPForSDaccess, slaveDeviceIPForSDaccess,
                     slaveDeviceIPForSDaccessDeviceNumber2, placeHolder, timeOfAccessingSDcard)
        listFilesOnSDcard()
    elif tempSelectedSDcardFunctionToPerformValue == "2":
        # Function to download files on sd card
        print("2. Access SD Card -> 2. Download File")
        generateJson(masterDeviceNetworkForSDaccess, slaveDeviceNetworkForSDaccess,
                     slaveDeviceNetworkForSDaccessDeviceNumber2, masterDeviceIPForSDaccess, slaveDeviceIPForSDaccess,
                     slaveDeviceIPForSDaccessDeviceNumber2, placeHolder, timeOfAccessingSDcard)
        downloadFileOnSDcard()
    elif tempSelectedSDcardFunctionToPerformValue == "3":
        # Function to upload files on sd card
        print("2. Access SD Card -> 3. Download all files")
        generateJson(masterDeviceNetworkForSDaccess, slaveDeviceNetworkForSDaccess,
                     slaveDeviceNetworkForSDaccessDeviceNumber2, masterDeviceIPForSDaccess, slaveDeviceIPForSDaccess,
                     slaveDeviceIPForSDaccessDeviceNumber2, placeHolder, timeOfAccessingSDcard)
        download_all_files(masterDeviceNetworkForSDaccess)
    elif tempSelectedSDcardFunctionToPerformValue == "4":
        # Function to delete files on sd card
        print("2. Access SD Card -> 4. Delete all files on SD card")
        deleteFilesChoice = input("Are you sure you want to delete all files on the SD card?\n1. Yes\n2. No\n")
        if deleteFilesChoice == "1":
            generateJson(masterDeviceNetworkForSDaccess, slaveDeviceNetworkForSDaccess,
                         slaveDeviceNetworkForSDaccessDeviceNumber2, masterDeviceIPForSDaccess,
                         slaveDeviceIPForSDaccess, slaveDeviceIPForSDaccessDeviceNumber2, placeHolder,
                         timeOfAccessingSDcard)
            delete_all_files(masterDeviceNetworkForSDaccess, masterDeviceIPForSDaccess)
        elif deleteFilesChoice == "2":
            print("Exiting...")
        else:
            print("Invalid choice. Exiting...")
    elif tempSelectedSDcardFunctionToPerformValue == "5":
        # Function to exit sd card
        print("2. Access SD Card -> 5. Exit")


# ================================================================================================================#
""" @brief received information on data that is on the sd card
    @note works with master device only at the moment
"""


def receiveSDcardInfo():
    """Send a command to the ESP32 server and return the response."""
    try:
        # Receive the response
        response = []
        while True:
            data = masterDeviceEthernet.recv(512)
            if not data:
                break
            response.append(data.decode())
        return "".join(response)
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        masterDeviceEthernet.close()


# ================================================================================================================#
""" @brief lists files on the sd card
    @note works with the master device only at the moment
"""


def listFilesOnSDcard():
    """List files on the ESP32 SD card."""
    response = receiveSDcardInfo()
    if response:
        print("Files on SD card:")
        print(response)


# ================================================================================================================#
""" @brief main function"""


def downloadFileOnSDcard():
    remote_filename = input("Enter the name of the file to download: ")
    local_filename = input("Enter the name to save the file as: ")
    """Download a file from the ESP32."""
    command = f"DOWNLOAD {remote_filename}"
    try:
        # Connect to ESP32 server
        # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client.connect((DEVICE_0_ETHERNET_IP, DEVICE_0_PORT))
        # sends name of file to download
        masterDeviceEthernet.sendall((command + "\n").encode())

        # Receive file data and save locally
        with open(local_filename, "wb") as file:
            while True:
                data = masterDeviceEthernet.recv(512)
                if not data:
                    break
                file.write(data)
        print(f"File downloaded successfully as {local_filename}")
    except Exception as e:
        print(f"Error: {e}")
    # finally:
    #     masterDeviceEthernet.close()


# ================================================================================================================#
# add logic for multiple devices, also for slave device
def autodownloadAudioOnSDcard(deviceNetworkForAudioOnSDcard, fileNumberToDownload, deviceFileName):
    remote_filename = f"/{timeStamp}_file_number_{fileNumberToDownload}_{deviceFileName}.wav"
    localFilename = f"{timeStamp}_file_number_{fileNumberToDownload}_{deviceFileName}.wav"

    """Download a file from the ESP32."""
    command = f"DOWNLOAD {remote_filename}"
    try:
        # Send the name of the file to download
        deviceNetworkForAudioOnSDcard.sendall((command + "\n").encode())

        # Receive file data and save locally
        with open(localFilename, "wb") as file:
            buffer = b""
            while True:
                data = deviceNetworkForAudioOnSDcard.recv(512)
                if not data:
                    break

                buffer += data

                # Check if "EOF\n" is present in the buffer
                if b"EOF\n" in buffer:
                    file.write(buffer[:-4])  # Remove "EOF\n" before writing
                    break

                file.write(buffer)
                buffer = b""  # Clear buffer after writing

        print(f"File downloaded successfully as {localFilename}")
    except Exception as e:
        print(f"Error: {e}")


# ================================================================================================================#
# add logic for multiple devices, also for slave device
def autodownloadSensorDataOnSDcard(deviceNetworkForSensorDataOnSDcard, fileNumberToDownload, sensorName,
                                   deviceFileName):
    remote_filename = f"/{timeStamp}_file_number_{fileNumberToDownload}_{sensorName}_{deviceFileName}.csv"
    localFilename = f"{timeStamp}_file_number_{fileNumberToDownload}_{sensorName}_{deviceFileName}.csv"

    """Download a file from the ESP32."""
    command = f"DOWNLOAD {remote_filename}"
    try:
        # Send the name of the file to download
        deviceNetworkForSensorDataOnSDcard.sendall((command + "\n").encode())

        # Receive file data and save locally
        with open(localFilename, "wb") as file:
            buffer = b""
            while True:
                data = deviceNetworkForSensorDataOnSDcard.recv(512)
                if not data:
                    break

                buffer += data

                # Check if "EOF\n" is present in the buffer
                if b"EOF\n" in buffer:
                    file.write(buffer[:-4])  # Remove "EOF\n" before writing
                    break

                file.write(buffer)
                buffer = b""  # Clear buffer after writing

        print(f"File downloaded successfully as {localFilename}")
    except Exception as e:
        print(f"Error: {e}")


# ================================================================================================================#
def generateJson(masterDeviceNetworkForJson, slaveDeviceNetworkForJson, slaveDeviceNetworkForJsonDeviceNumber2,
                 masterDeviceIPForJson, slaveDeviceIPForJson, slaveDeviceIPForJsonDeviceNumber2,
                 fileNumberFromPythonScript, timeStampForJson):
    global doneReceivingAudio, total_samples, audio_data_size, timeStamp

    # selectedMainMenuFunction,selectedRecordingMethodValue,selectedDataToRecordValue,selectedSDcardFunctionToPerformValue = mainMenu()
    audioSamplingRateToSend = sendSamplingRateCommand(audioSamplingRate)
    audioDigitalVolumeToSend = sendAudioDigitalVolumeCommand(audioDigitalVolume)
    audioHighPassFilterToSend = sendAudioHighPassFilterCommand(audioHighPassFilter)
    audioGainCalibrationToSend = sendAudioGainCalibrationCommand(audioGainCalibration)

    accelDataRateValueToSend = accelDataRateValue(accelDataRate)
    accelDataRangeValueToSend = accelDataRangeValue(accelDataRange)
    gyroDataRateValueToSend = gyroDataRateValue(gyroDataRate)
    gyroDataRangeValueToSend = gyroDataRangeValue(gyroDataRange)
    # currentTimeOfSendingCommand = datetime.now().strftime("%Y%m%d_%H%M%S")
    # timeStamp = f"{currentTimeOfSendingCommand}"
    timeStamp = f"{timeStampForJson}"

    # JSON data to send
    DEVICE_TYPE = DEVICE_0  # initialise device type variable
    data = {
        "selectedMainMenuFunction": selectedMainMenuFunction,
        "selectedRecordingMethodValue": selectedRecordingMethodValue,
        "selectedDataToRecordValue": selectedDataToRecordValue,
        "sdCardFunctionToPerformValue": selectedSDcardFunctionToPerformValue,
        "audioSamplingRateToSend": audioSamplingRateToSend,
        "RECORDING_DURATION": RECORDING_DURATION,
        "audioDigitalVolumeToSend": audioDigitalVolumeToSend,
        "audioHighPassFilterToSend": audioHighPassFilterToSend,
        "audioGainCalibrationToSend": audioGainCalibrationToSend,
        "accelDataRateValueToSend": accelDataRateValueToSend,
        "accelDataRangeValueToSend": accelDataRangeValueToSend,
        "gyroDataRateValueToSend": gyroDataRateValueToSend,
        "gyroDataRangeValueToSend": gyroDataRangeValueToSend,
        "numberOfFilesToRecord": numberOfFilesToRecord,
        "timeStamp": timeStamp,
        "fileNumberFromPythonScript": fileNumberFromPythonScript,
        "DEVICE_TYPE": DEVICE_TYPE,
        "sensorDataBatchSize": sensorDataBatchSize
    }

    json_data = json.dumps(data)
    if numberOfDevicesConnected == 1:  # one device connected
        masterDeviceNetworkForJson.connect((masterDeviceIPForJson, DEVICE_0_PORT))
        # Send JSON data
        DEVICE_TYPE = DEVICE_0
        masterDeviceNetworkForJson.sendall(json_data.encode("utf-8"))
        print("Data sent to ESP32")
    elif numberOfDevicesConnected == 2:  # multiple devices connected, master and slave device
        masterDeviceNetworkForJson.connect((masterDeviceIPForJson, DEVICE_0_PORT))
        slaveDeviceNetworkForJson.connect((slaveDeviceIPForJson, DEVICE_0_PORT))
        # Send JSON data
        DEVICE_TYPE = DEVICE_0
        data = {
            "selectedMainMenuFunction": selectedMainMenuFunction,
            "selectedRecordingMethodValue": selectedRecordingMethodValue,
            "selectedDataToRecordValue": selectedDataToRecordValue,
            "sdCardFunctionToPerformValue": selectedSDcardFunctionToPerformValue,
            "audioSamplingRateToSend": audioSamplingRateToSend,
            "RECORDING_DURATION": RECORDING_DURATION,
            "audioDigitalVolumeToSend": audioDigitalVolumeToSend,
            "audioHighPassFilterToSend": audioHighPassFilterToSend,
            "audioGainCalibrationToSend": audioGainCalibrationToSend,
            "accelDataRateValueToSend": accelDataRateValueToSend,
            "accelDataRangeValueToSend": accelDataRangeValueToSend,
            "gyroDataRateValueToSend": gyroDataRateValueToSend,
            "gyroDataRangeValueToSend": gyroDataRangeValueToSend,
            "numberOfFilesToRecord": numberOfFilesToRecord,
            "timeStamp": timeStamp,
            "fileNumberFromPythonScript": fileNumberFromPythonScript,
            "DEVICE_TYPE": DEVICE_TYPE,
            "sensorDataBatchSize": sensorDataBatchSize
        }
        json_data = json.dumps(data)
        masterDeviceNetworkForJson.sendall(json_data.encode("utf-8"))

        DEVICE_TYPE = DEVICE_1
        data = {
            "selectedMainMenuFunction": selectedMainMenuFunction,
            "selectedRecordingMethodValue": selectedRecordingMethodValue,
            "selectedDataToRecordValue": selectedDataToRecordValue,
            "sdCardFunctionToPerformValue": selectedSDcardFunctionToPerformValue,
            "audioSamplingRateToSend": audioSamplingRateToSend,
            "RECORDING_DURATION": RECORDING_DURATION,
            "audioDigitalVolumeToSend": audioDigitalVolumeToSend,
            "audioHighPassFilterToSend": audioHighPassFilterToSend,
            "audioGainCalibrationToSend": audioGainCalibrationToSend,
            "accelDataRateValueToSend": accelDataRateValueToSend,
            "accelDataRangeValueToSend": accelDataRangeValueToSend,
            "gyroDataRateValueToSend": gyroDataRateValueToSend,
            "gyroDataRangeValueToSend": gyroDataRangeValueToSend,
            "numberOfFilesToRecord": numberOfFilesToRecord,
            "timeStamp": timeStamp,
            "fileNumberFromPythonScript": fileNumberFromPythonScript,
            "DEVICE_TYPE": DEVICE_TYPE,
            "sensorDataBatchSize": sensorDataBatchSize
        }
        json_data = json.dumps(data)
        slaveDeviceNetworkForJson.sendall(json_data.encode("utf-8"))
        print("Data sent to ESP32")

    else:  # multiple devices connected, master and slave device
        masterDeviceNetworkForJson.connect((masterDeviceIPForJson, DEVICE_0_PORT))
        slaveDeviceNetworkForJson.connect((slaveDeviceIPForJson, DEVICE_0_PORT))
        slaveDeviceNetworkForJsonDeviceNumber2.connect((slaveDeviceIPForJsonDeviceNumber2, DEVICE_0_PORT))
        # Send JSON data
        DEVICE_TYPE = DEVICE_0
        data = {
            "selectedMainMenuFunction": selectedMainMenuFunction,
            "selectedRecordingMethodValue": selectedRecordingMethodValue,
            "selectedDataToRecordValue": selectedDataToRecordValue,
            "sdCardFunctionToPerformValue": selectedSDcardFunctionToPerformValue,
            "audioSamplingRateToSend": audioSamplingRateToSend,
            "RECORDING_DURATION": RECORDING_DURATION,
            "audioDigitalVolumeToSend": audioDigitalVolumeToSend,
            "audioHighPassFilterToSend": audioHighPassFilterToSend,
            "audioGainCalibrationToSend": audioGainCalibrationToSend,
            "accelDataRateValueToSend": accelDataRateValueToSend,
            "accelDataRangeValueToSend": accelDataRangeValueToSend,
            "gyroDataRateValueToSend": gyroDataRateValueToSend,
            "gyroDataRangeValueToSend": gyroDataRangeValueToSend,
            "numberOfFilesToRecord": numberOfFilesToRecord,
            "timeStamp": timeStamp,
            "fileNumberFromPythonScript": fileNumberFromPythonScript,
            "DEVICE_TYPE": DEVICE_TYPE,
            "sensorDataBatchSize": sensorDataBatchSize
        }
        json_data = json.dumps(data)
        masterDeviceNetworkForJson.sendall(json_data.encode("utf-8"))

        DEVICE_TYPE = DEVICE_1
        data = {
            "selectedMainMenuFunction": selectedMainMenuFunction,
            "selectedRecordingMethodValue": selectedRecordingMethodValue,
            "selectedDataToRecordValue": selectedDataToRecordValue,
            "sdCardFunctionToPerformValue": selectedSDcardFunctionToPerformValue,
            "audioSamplingRateToSend": audioSamplingRateToSend,
            "RECORDING_DURATION": RECORDING_DURATION,
            "audioDigitalVolumeToSend": audioDigitalVolumeToSend,
            "audioHighPassFilterToSend": audioHighPassFilterToSend,
            "audioGainCalibrationToSend": audioGainCalibrationToSend,
            "accelDataRateValueToSend": accelDataRateValueToSend,
            "accelDataRangeValueToSend": accelDataRangeValueToSend,
            "gyroDataRateValueToSend": gyroDataRateValueToSend,
            "gyroDataRangeValueToSend": gyroDataRangeValueToSend,
            "numberOfFilesToRecord": numberOfFilesToRecord,
            "timeStamp": timeStamp,
            "fileNumberFromPythonScript": fileNumberFromPythonScript,
            "DEVICE_TYPE": DEVICE_TYPE,
            "sensorDataBatchSize": sensorDataBatchSize
        }
        json_data = json.dumps(data)
        slaveDeviceNetworkForJson.sendall(json_data.encode("utf-8"))
        slaveDeviceNetworkForJsonDeviceNumber2.sendall(json_data.encode("utf-8"))
        print("Data sent to ESP32")

    return json_data


##############################################################################################################
import numpy as np
from scipy.signal import butter, filtfilt

RATE = 48000

# Función de beamforming (delay-and-sum)
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, rate, c=343):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))
    for az_idx, theta in enumerate(azimuth_range):
        az_rad = np.radians(theta)
        for el_idx, phi in enumerate(elevation_range):
            el_rad = np.radians(phi)
            direction = np.array([np.cos(el_rad)*np.cos(az_rad),
                                  np.cos(el_rad)*np.sin(az_rad),
                                  np.sin(el_rad)])
            delays = np.dot(mic_positions, direction) / c
            summed = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                shift = int(np.round(delay * rate))
                # Usamos np.roll para aplicar el desfase
                summed += np.roll(signal_data[:, i], shift)
            summed /= signal_data.shape[1]  # normalizamos
            energy[az_idx, el_idx] = np.sum(summed**2)
    return energy

# Función de filtrado (bandpass)
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=400.0, highcut=18000.0, rate=RATE, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    return filtfilt(b, a, data, axis=0)

def process_audio_from_queue():
    # Define beamforming parameters
    azimuth_range = np.arange(-180, 181, 5)
    elevation_range = np.arange(0, 91, 5)
    # Example: a circular array of 8 microphones with a radius of 0.3 m
    radius = 0.3
    angles = np.linspace(0, 360, 8, endpoint=False)
    mic_positions = np.array([[radius * np.cos(np.radians(a)),
                               radius * np.sin(np.radians(a)),
                               0] for a in angles])

    while True:
        try:
            # Get an audio chunk from the queue (blocking up to 1 second)
            data = audio_processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            # Convert the raw bytes to a NumPy array. Adjust dtype and channel count as needed.
            samples = np.frombuffer(data, dtype=np.int16)
            sample_count = len(samples) // 8  # assuming 8 channels
            if sample_count == 0:
                continue
            samples = samples[:sample_count * 8].reshape((sample_count, 8))
        except Exception as e:
            print(f"Conversion error: {e}")
            audio_processing_queue.task_done()
            continue

        # Apply bandpass filtering (adjust parameters as needed)
        filtered = apply_bandpass_filter(samples, lowcut=400.0, highcut=18000.0, rate=RATE, order=5)

        # Apply beamforming (using delay-and-sum)
        energy = beamform_time(filtered, mic_positions, azimuth_range, elevation_range, RATE, c=343)

        # Estimate the direction by finding the maximum energy point
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]
        print(f"Estimated Direction: Azimuth {estimated_azimuth}°, Elevation {estimated_elevation}°")

        audio_processing_queue.task_done()


# ================================================================================================================#
def main():
    global selectedMainMenuFunction, selectedRecordingMethodValue, selectedDataToRecordValue, selectedSDcardFunctionToPerformValue
    selectedMainMenuFunction, selectedRecordingMethodValue, selectedDataToRecordValue, selectedSDcardFunctionToPerformValue = mainMenu()

    # Start the real-time processing thread (daemon so it exits with the main program)
    proc_thread = threading.Thread(target=process_audio_from_queue, daemon=True)
    proc_thread.start()

    # Based on the main menu selection, record data or access SD card.
    if selectedMainMenuFunction == "1":
        recordData(selectedRecordingMethodValue, selectedDataToRecordValue)
    elif selectedMainMenuFunction == "2":
        accessSDcard(masterDeviceEthernet, slaveDeviceEthernet, slaveDeviceEthernetDeviceNumber2,
                     DEVICE_0_ETHERNET_IP, DEVICE_1_ETHERNET_IP, DEVICE_2_ETHERNET_IP,
                     selectedSDcardFunctionToPerformValue)


if __name__ == "__main__":
    main()