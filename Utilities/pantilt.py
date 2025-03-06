import struct
import time
import serial
import numpy as np  # Needed for smoothing calculations

SERIAL_BAUD = 9600
SERIAL_SOF1 = 0xAA
SERIAL_SOF2 = 0xFF
CLOCK_SCALE_FACTOR = 16.0  # Rescales value from microseconds to ticks


class Pantilt:
    def __init__(self, port: str,
                 pan_n90: int = 900, pan_p90: int = 2100,
                 tilt_n90: int = 900, tilt_p90: int = 2100,
                 window_size: int = 20, slow_factor: float = 0.1,
                 threshold: float = 10.0, initial_pan: float = 0.0,
                 initial_tilt: float = 0.0):
        # Initialize the serial connection to the pantilt hardware
        self.serial = serial.Serial(port, baudrate=SERIAL_BAUD)
        time.sleep(3.0)  # Allow time for the serial connection to stabilize
        self.pan_n90 = pan_n90
        self.pan_p90 = pan_p90
        self.tilt_n90 = tilt_n90
        self.tilt_p90 = tilt_p90

        # Parameters for smoothing
        self.window_size = window_size  # Size of the sliding window (number of recent measurements)
        self.slow_factor = slow_factor  # Factor to gradually update when difference is high
        self.threshold = threshold  # Threshold in degrees to decide immediate vs. gradual update

        # Sliding window for recent angle estimates
        self.azimuth_window = []
        self.elevation_window = []

        # Current pan and tilt state (initialized with provided initial values)
        self.current_pan = initial_pan
        self.current_tilt = initial_tilt

    def set(self, pan_degrees: float, tilt_degrees: float):
        # Convert the input angles (in degrees) to timer ticks based on calibration parameters
        pan_val = (-pan_degrees + 90.0) / 180.0 * (self.pan_p90 - self.pan_n90) + self.pan_n90
        pan_val = round(pan_val * CLOCK_SCALE_FACTOR)

        tilt_val = (-tilt_degrees + 90.0) / 180.0 * (self.tilt_p90 - self.tilt_n90) + self.tilt_n90
        tilt_val = round(tilt_val * CLOCK_SCALE_FACTOR)

        # Build the message packet to send to the pantilt hardware
        msg = [SERIAL_SOF1, SERIAL_SOF2]
        msg += list(struct.pack("<H", pan_val))
        msg += list(struct.pack("<H", tilt_val))

        # Send the command via serial
        self.serial.write(bytearray(msg))
        self.serial.flush()

    def set_smoothed(self, pan_degrees: float, tilt_degrees: float):
        # ---------------------------------------------------------
        # Update the sliding window with the new estimated values
        # ---------------------------------------------------------
        self.azimuth_window.append(pan_degrees)
        self.elevation_window.append(tilt_degrees)
        if len(self.azimuth_window) > self.window_size:
            self.azimuth_window.pop(0)
            self.elevation_window.pop(0)

        # ---------------------------------------------------------
        # Determine the target angles using the smoothing strategy
        # ---------------------------------------------------------
        if len(self.azimuth_window) == self.window_size:
            # Calculate the mean of the window for both angles
            mean_pan = np.mean(self.azimuth_window)
            mean_tilt = np.mean(self.elevation_window)
            # Compute the Euclidean distance between the current estimate and the window mean
            distance = np.sqrt((pan_degrees - mean_pan) ** 2 + (tilt_degrees - mean_tilt) ** 2)
            if distance < self.threshold:
                # If within threshold, update immediately
                target_pan = pan_degrees
                target_tilt = tilt_degrees
            else:
                # If outside threshold, update gradually using the slow_factor
                target_pan = self.current_pan + self.slow_factor * (pan_degrees - self.current_pan)
                target_tilt = self.current_tilt + self.slow_factor * (tilt_degrees - self.current_tilt)
        else:
            # If the window is not full, update immediately
            target_pan = pan_degrees
            target_tilt = tilt_degrees

        # ---------------------------------------------------------
        # Update the current state and send the command
        # ---------------------------------------------------------
        self.current_pan = target_pan
        self.current_tilt = target_tilt
        self.set(target_pan, target_tilt)
