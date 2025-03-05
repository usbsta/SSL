import time

import control

pantilt = control.Pantilt("COM4")

for i in range(10):
    print("Loop {0}".format(i))
    pantilt.set(pan_degrees=0.0, tilt_degrees=0.0)
    time.sleep(1)
    pantilt.set(pan_degrees=-45.0, tilt_degrees=45.0)
    time.sleep(1)
    pantilt.set(pan_degrees=45.0, tilt_degrees=-45.0)
    time.sleep(1)