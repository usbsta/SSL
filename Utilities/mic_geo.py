import numpy as np

def mic_6_E_orange():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.16, 0.0]
    r = [0.13, 0.62]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

def mic_6_S_orange():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.16, 0.0]
    r = [0.13, 0.62]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

def mic_6_W_black():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.09, 0.0]
    r = [0.13, 0.5]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

def mic_6_N_black_thin():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.09, 0.0]
    r = [0.11, 0.54]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions