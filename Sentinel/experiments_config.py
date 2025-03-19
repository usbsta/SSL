#!/usr/bin/env python3
"""
experiment_configs.py

This module consolidates configuration parameters for multiple beamforming experiments.
Each experiment is defined by its own set of parameters, including:
    - Synchronization corrections (in samples)
    - List of unsynchronized WAV file paths
    - Skip seconds (time to skip at the beginning of each WAV file)
    - Initial beamforming offsets (azimuth and elevation in degrees)
    - Flight data parameters (start index, reference CSV file, flight CSV file)

You can retrieve the configuration for a specific experiment by calling the
get_experiment_config(exp_number) function.

Example usage:
    from experiments_config import get_experiment_config

    config = get_experiment_config(11)  # Retrieve parameters for experiment 11
    corrections = config["corrections"]
    wav_filenames = config["wav_filenames"]
    skip_seconds = config["skip_seconds"]
    initial_azimuth = config["initial_azimuth"]
    initial_elevation = config["initial_elevation"]
    start_index = config["start_index"]
    ref_file_path = config["ref_file_path"]
    file_path_flight = config["file_path_flight"]
"""

# Define a dictionary mapping experiment numbers to their configurations.
# Note: Replace the placeholder paths and values with the actual parameters for each experiment.
EXPERIMENT_CONFIGS = {

    0: {
        'name': 'DJI Air 3_5_Oct',
        "corrections": [0, 866, 626, 729],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_1_nosync.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_2_nosync.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_3_nosync.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_4_nosync.wav'
        ],
        "skip_seconds": 82,
        "initial_azimuth": -5.0,
        "initial_elevation": 0.0,
        "start_index": 39,
        "ref_file_path": "/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/11 Oct 24/DJIFlightRecord_2024-10-11_[14-32-34].csv",
        "file_path_flight": "/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/11 Oct 24/DJIFlightRecord_2024-10-11_[15-49-12].csv",
    },
    1: {
        'name': 'DJI Inspire 1_1', # Road
        "corrections": [0, 241, 385, 446], # from Nov 24 Dataset/Sync_Time_Sim_24Ch.py
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120459_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120501_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120502_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120503_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 72,
        "initial_azimuth": -6.0,
        "initial_elevation": 0.0,
        "start_index": 91,
        "ref_file_path": "/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/22 Nov/Ref/Nov-22nd-2024-11-45AM-Flight-Airdata.csv",
        "file_path_flight": "/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/22 Nov/1/Inspire_Nov-22nd-2024-11-55AM-Flight-Airdata.csv",
    },
    2: {
        'name': 'DJI Air 3_2',
        "corrections": [0, 485, 351, 538],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/2/20241122_121915_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/2/20241122_121917_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/2/20241122_121918_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/2/20241122_121920_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 139,
        "initial_azimuth": 6.0,
        "initial_elevation": 0.0,
        "start_index": 40,
        "ref_file_path": "/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/Ref/Nov-22nd-2024-11-48AM-Flight-Airdata.csv",
        "file_path_flight": "/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/2/Air_Nov-22nd-2024-12-09PM-Flight-Airdata.csv",
    },
    3: {
        'name': 'DJI Air 3_3',
        "corrections": [0, -151, -331, 111],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124721_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124723_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124725_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124727_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 225,
        "initial_azimuth": 6.0,
        "initial_elevation": 0.0,
        "start_index": 15,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/Ref/Nov-22nd-2024-11-48AM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/3/Air_Nov-22nd-2024-12-31PM-Flight-Airdata.csv',
    },
    31: {
        'name': 'DJI Inspire 1_3_1',
        "corrections": [0, -151, -331, 111],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124721_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124723_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124725_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124727_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 242,
        "initial_azimuth": -26.0,
        "initial_elevation": 0.0,
        "start_index": 173,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/22 Nov/Ref/Nov-22nd-2024-11-45AM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/22 Nov/3.1/Inspire_Nov-22nd-2024-12-31PM-Flight-Airdata.csv',
    },
    4: {
        'name': 'Holybro_X500_4',
        "corrections": [0, 952, 1100, 868],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/4/20241125_115024_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/4/20241125_115026_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/4/20241125_115028_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/4/20241125_115030_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 148,
        "initial_azimuth": -8.0,
        "initial_elevation": 0.0,
        "start_index": 76,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/X500_18.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/X500_19.csv',
    },
    5: {
        'name': 'DJI Inspire 1_5',
        "corrections": [0, -288, -91, 106],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/5/20241125_120834_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/5/20241125_120836_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/5/20241125_120838_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/5/20241125_120840_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 197,
        "initial_azimuth": -50.0,
        "initial_elevation": 0.0,
        "start_index": 35,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/Ref/Inspire_Nov-25th-2024-11-21AM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/5/Inspire_Nov-25th-2024-11-56AM-Flight-Airdata.csv',
    },
    6: {
        'name': 'DJI Air 3_6',
        "corrections": [0, 140, -148, 251],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/6/20241125_122233_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/6/20241125_122234_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/6/20241125_122236_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/6/20241125_122237_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 157,
        "initial_azimuth": -50.0,
        "initial_elevation": 0.0,
        "start_index": 32,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/Ref/Air_Nov-25th-2024-11-14AM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/6/Air_Nov-25th-2024-12-13PM-Flight-Airdata.csv',
    },
    7: {
        'name': 'DJI Inspire 1_7',
        "corrections": [0, -240, 196, 309],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/7/20241125_124246_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/7/20241125_124248_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/7/20241125_124250_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/7/20241125_124252_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 200,
        "initial_azimuth": -50.0,
        "initial_elevation": 0.0,
        "start_index": 40,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/Ref/Inspire_Nov-25th-2024-11-21AM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/7/Inspire_Nov-25th-2024-12-33PM-Flight-Airdata.csv',
    },
    8: {
        'name': 'Holybro X500_8',
        "corrections": [0, -104, -236, -135],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155601_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155609_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155616_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155624_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 62,
        "initial_azimuth": -7.0,
        "initial_elevation": 0.0,
        "start_index": 7410,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/RefLake.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/X500_21.csv',
    },
    81: {
        'name': 'Holybro X500_8_1',
        "corrections": [0, -104, -236, -135],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155601_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155609_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155616_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155624_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 1098,
        "initial_azimuth": -7.0,
        "initial_elevation": 0.0,
        "start_index": 2134,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/RefLake.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/X500_22.csv',
    },
    82: {
        'name': 'DJI Inspire 1_8_2',
        "corrections": [0, -104, -236, -135],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155601_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155609_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155616_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/8/20241125_155624_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 1116,
        "initial_azimuth": -7.0,
        "initial_elevation": 0.0,
        "start_index": 42,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/RefLake.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/8.2/Inspire_Nov-25th-2024-03-44PM-Flight-Airdata.csv',
    },
    10: {
        'name': 'DJI Air 3_10',
        "corrections": [0, 145, -197, -371],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161423_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161424_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161425_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161427_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 127,
        "initial_azimuth": -7.0,
        "initial_elevation": 0.0,
        "start_index": 30,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/Ref2/Air_Nov-25th-2024-03-19PM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/10/Air_Nov-25th-2024-04-04PM-Flight-Airdata.csv',
    },
    101: {
        'name': 'DJI Inspire 1_10_1',
        "corrections": [0, 145, -197, -371],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161423_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161424_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161425_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/10/20241125_161427_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 157,
        "initial_azimuth": -8.0,
        "initial_elevation": 0.0,
        "start_index": 43,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/RefLake.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/25 Nov/10.1/Inspire_Nov-25th-2024-04-04PM-Flight-Airdata.csv',
    },
    11: {
        'name': 'DJI Air 3_11',
        "corrections": [0, 240, 98, -538],
        "wav_filenames": [
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/11/20241125_165753_device_1_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/11/20241125_165757_device_2_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/11/20241125_165801_device_3_nosync_part1.wav',
            '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/25 Nov 24/11/20241125_165804_device_4_nosync_part1.wav'
        ],
        "skip_seconds": 149,
        "initial_azimuth": -17.0,
        "initial_elevation": 0.0,
        "start_index": 26,
        "ref_file_path": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/Ref2/Air_Nov-25th-2024-03-19PM-Flight-Airdata.csv',
        "file_path_flight": '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/11/Air_Nov-25th-2024-04-32PM-Flight-Airdata.csv',
    },

}

def get_experiment_config(exp_number):
    """
    Retrieve the configuration parameters for a given experiment number.

    Parameters:
        exp_number (int): The experiment number (e.g., 1, 2, ..., 11).

    Returns:
        dict: A dictionary containing the configuration parameters for the experiment.

    Raises:
        KeyError: If the experiment number is not defined in the configuration.
    """
    try:
        return EXPERIMENT_CONFIGS[exp_number]
    except KeyError:
        raise KeyError(
            f"Configuration for experiment {exp_number} is not defined. "
            "Please add its parameters to EXPERIMENT_CONFIGS."
        )
