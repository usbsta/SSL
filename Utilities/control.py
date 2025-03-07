import struct
import time

import serial

SERIAL_BAUD = 9600
SERIAL_SOF1 = 0xAA
SERIAL_SOF2 = 0xFF

CLOCK_SCALE_FACTOR = 16.0         # Recales the value from uS to ticks for setting Timer 1 on the micro

class Pantilt:
    def __init__(
        self,
        port: str,
        pan_n90: int = 900,
        pan_p90: int = 2100,
        tilt_n90: int = 900,
        tilt_p90: int = 2100,
    ):
        self.serial = serial.Serial(port, baudrate=SERIAL_BAUD)
        time.sleep(3.0)
        self.pan_n90 = pan_n90
        self.pan_p90 = pan_p90
        self.tilt_n90 = tilt_n90
        self.tilt_p90 = tilt_p90

    def set(self, pan_degrees: float, tilt_degrees: float):
        pan_val  = (-pan_degrees  + 90.0) / 180.0 * (self.pan_p90  - self.pan_n90)  + self.pan_n90 
        pan_val  = round(pan_val * CLOCK_SCALE_FACTOR)

        tilt_val = (-tilt_degrees + 90.0) / 180.0 * (self.tilt_p90 - self.tilt_n90) + self.tilt_n90 
        tilt_val =  round(tilt_val * CLOCK_SCALE_FACTOR)

        msg  = [SERIAL_SOF1, SERIAL_SOF2]
        msg += list(struct.pack("<H", pan_val))
        msg += list(struct.pack("<H", tilt_val))
        self.serial.write(bytearray(msg))
        self.serial.flush()


