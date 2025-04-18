#!/usr/bin/env python
import os
import re
import glob
import yaml
import math
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import scipy.ndimage as ndimage
import pandas as pd

# ---------------- Drone/Geo/Audio imports ----------------
from pyproj import Transformer
from numba import njit
from audio_beamforming import beamform_time
from geo_utils import (
    wrap_angle,
    calculate_azimuth_meters,
    calculate_elevation_meters,
    calculate_total_distance_meters,
    calculate_angle_difference
)
from io_utils import (
    read_wav_block,
    apply_bandpass_filter,
    calculate_time,
    initialize_beamforming_params,
    open_wav_files
)

# ---------------- UNet imports ----------------
# We assume unet_optuna.py exports the following:
#   device: the torch device
#   UNet: the U-Net model class
#   CombinedLoss: the combined BCE + "AtLeastOneMatch" loss
from unet_optuna import device, UNet, CombinedLoss

# ---------------- wavelet import (if needed) ----------------
import fcwt


# ----------------------- Helper Functions -----------------------
def format_time_s(total_seconds: float) -> str:
    """Convert seconds to 'MM:SS.ss' format."""
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def create_overlay(label_2d, pred_2d):
    """
    Create an RGB overlay image:
      - Yellow where pred & label overlap
      - Red where label=1 & pred=0
      - Green where pred=1 & label=0
    """
    label = label_2d.cpu().numpy() if isinstance(label_2d, torch.Tensor) else label_2d
    pred  = pred_2d.cpu().numpy()  if isinstance(pred_2d, torch.Tensor)  else pred_2d
    label = (label > 0.5).astype(np.uint8)
    pred  = (pred > 0.5).astype(np.uint8)

    H, W = label.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    overlap_mask    = (label == 1) & (pred == 1)
    label_only_mask = (label == 1) & (pred == 0)
    pred_only_mask  = (label == 0) & (pred == 1)

    rgb[overlap_mask]    = [1, 1, 0]  # yellow
    rgb[label_only_mask] = [1, 0, 0]  # red
    rgb[pred_only_mask]  = [0, 1, 0]  # green
    return rgb


# ----------------------------------------------------------
#                   UPDATED InferenceChunkDataset
# ----------------------------------------------------------
class InferenceChunkDataset:
    """
    A unified dataset that can handle two scenarios:
    1) The .npz contains 'X' & 'y':
       => shape(X) = (az, el, bins), shape(y) = (az, el).
       We optionally re-bin if num_fft_bins < 'bins', then (optionally) convert to polar.

    2) The .npz contains 'beamformed_data' & 'labels':
       => shape(beamformed_data) = (az, el, n_samples), shape(labels) = (az, el).
       We then perform on-the-fly feature extraction (FFT or wavelet), optional bin_mask,
       normalization, and optional polar conversion.
    """
    def __init__(
        self,
        folder_path,
        sr=48000,
        min_freq=200,
        max_freq=8000,
        num_fft_bins=32,
        bin_mask=None,
        feature_extraction_mode="fft",
        wavelet_param=4.0,
        convert_to_polar=False,
        debug=False
    ):
        self.folder_path = folder_path
        self.file_paths = sorted(
            glob.glob(os.path.join(folder_path, "*.npz")),
            key=self._numeric_key
        )
        if len(self.file_paths) == 0:
            raise ValueError(f"No NPZ files found in: {folder_path}")

        # Store parameters
        self.sr = sr
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_fft_bins = num_fft_bins
        self.bin_mask = bin_mask
        self.convert_to_polar = convert_to_polar
        self.feature_extraction_mode = feature_extraction_mode.lower()
        self.wavelet_param = wavelet_param
        self.debug = debug

        # If no bin_mask, default to all True
        if self.bin_mask is None:
            self.bin_mask = np.ones((self.num_fft_bins,), dtype=bool)
        if len(self.bin_mask) != self.num_fft_bins:
            raise ValueError("bin_mask length must match num_fft_bins.")

        # If wavelet mode, pre-initialize wavelet objects
        if self.feature_extraction_mode == "wavelet":
            self.morl = fcwt.Morlet(self.wavelet_param)
            self.scales = fcwt.Scales(
                self.morl,
                fcwt.FCWT_LINFREQS,
                self.sr,
                self.min_freq,
                self.max_freq,
                self.num_fft_bins
            )
            self.fcwt_obj = fcwt.FCWT(
                pwav=self.morl,
                pthreads=8,
                puse_optimalization_schemes=False,
                puse_normalization=False
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        data = np.load(path)

        # Check which scenario:
        if "beamformed_data" in data and "labels" in data:
            # raw time-domain => wavelet or fft
            return self._process_beamformed_data(data, path)
        elif "X" in data and "y" in data:
            # precomputed X,y => old style
            return self._process_precomputed(data, path)
        else:
            raise ValueError(
                f"NPZ file {os.path.basename(path)} missing keys: "
                "Expected either ('X','y') or ('beamformed_data','labels')."
            )

    def _process_precomputed(self, data, path):
        """
        If NPZ has 'X' & 'y':
          X: shape (az, el, bins)
          y: shape (az, el)
          -> optional re-binning if num_fft_bins < bins
          -> optional rect->polar
        """
        X_rect = data["X"]  # (az, el, bins)
        y_rect = data["y"]  # (az, el)

        original_bins = X_rect.shape[2]
        # If we want fewer bins, re-bin
        if self.num_fft_bins < original_bins:
            factor = original_bins // self.num_fft_bins
            if original_bins % self.num_fft_bins != 0:
                raise ValueError(
                    f"Cannot evenly re-bin {original_bins} -> {self.num_fft_bins}."
                )
            new_shape = (X_rect.shape[0], X_rect.shape[1], self.num_fft_bins, factor)
            X_rect = X_rect.reshape(new_shape).mean(axis=-1)

        # If convert_to_polar:
        if self.convert_to_polar:
            X_polar = self._rect2polar_input(X_rect)
            y_polar = self._rect2polar_mask(y_rect)
            X_tensor = torch.tensor(X_polar, dtype=torch.float32)
            y_tensor = torch.tensor(y_polar, dtype=torch.float32)
        else:
            # keep rectangular => (bins, az, el)
            X_rect = np.transpose(X_rect, (2, 0, 1))
            X_tensor = torch.tensor(X_rect, dtype=torch.float32)
            y_tensor = torch.tensor(y_rect, dtype=torch.float32)

        return X_tensor, y_tensor, os.path.basename(path)

    def _process_beamformed_data(self, data, path):
        """
        If NPZ has 'beamformed_data' & 'labels':
          shape(beamformed_data) = (az, el, n_samples)
          shape(labels)          = (az, el)
          -> wavelet/FFT
          -> bin_mask, normalization
          -> optional polar transform
        """
        beamformed_data = data["beamformed_data"]  # (az, el, n_samples)
        labels_rect     = data["labels"]           # (az, el)

        az_size, el_size, n_samples = beamformed_data.shape

        if self.feature_extraction_mode == "wavelet":
            # Wavelet approach
            all_features = np.zeros((az_size, el_size, self.num_fft_bins), dtype=np.float32)
            for i_az in range(az_size):
                for i_el in range(el_size):
                    signal_td = beamformed_data[i_az, i_el, :].astype(np.float32)
                    cwt_output = np.zeros((self.num_fft_bins, n_samples), dtype=np.complex64)

                    self.fcwt_obj.cwt(signal_td, self.scales, cwt_output)
                    # average magnitude across time
                    mag_avg = np.mean(np.abs(cwt_output), axis=1)
                    all_features[i_az, i_el, :] = mag_avg

            # global normalization
            mean_val = all_features.mean()
            std_val  = all_features.std() + 1e-8
            all_features = (all_features - mean_val) / std_val

        else:
            # FFT approach
            freq_resolution = self.sr / n_samples
            freqs = np.fft.rfftfreq(n_samples, d=1.0/self.sr)
            freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            freq_in_range = np.sum(freq_mask)
            if freq_in_range < self.num_fft_bins:
                raise ValueError(
                    f"freq_in_range({freq_in_range}) < num_fft_bins({self.num_fft_bins})."
                )

            all_ffts = np.zeros((az_size, el_size, freq_in_range), dtype=np.float32)
            for i_az in range(az_size):
                for i_el in range(el_size):
                    signal_td = beamformed_data[i_az, i_el, :]
                    fft_vals = np.fft.rfft(signal_td)
                    mag = np.abs(fft_vals).astype(np.float32)
                    all_ffts[i_az, i_el, :] = mag[freq_mask]

            # global normalization
            mean_val = all_ffts.mean()
            std_val  = all_ffts.std() + 1e-8
            all_ffts = (all_ffts - mean_val) / std_val

            # bin down from freq_in_range -> num_fft_bins
            bin_factor = freq_in_range // self.num_fft_bins
            freq_in_range_eff = bin_factor * self.num_fft_bins
            if freq_in_range_eff < freq_in_range:
                all_ffts = all_ffts[:, :, :freq_in_range_eff]

            all_ffts = all_ffts.reshape(
                az_size, el_size, self.num_fft_bins, bin_factor
            ).mean(axis=-1)

            all_features = all_ffts

        # apply bin_mask => shape (az, el, sum(bin_mask))
        all_features = all_features[:, :, self.bin_mask]

        # convert to polar if requested
        if self.convert_to_polar:
            X_polar = self._rect2polar_input(all_features)
            y_polar = self._rect2polar_mask(labels_rect)
            X_tensor = torch.tensor(X_polar, dtype=torch.float32)
            y_tensor = torch.tensor(y_polar, dtype=torch.float32)
        else:
            # shape => (az, el, channels) => (channels, az, el)
            X_rect_trans = np.transpose(all_features, (2, 0, 1))
            X_tensor = torch.tensor(X_rect_trans, dtype=torch.float32)
            y_tensor = torch.tensor(labels_rect, dtype=torch.float32)

        return X_tensor, y_tensor, os.path.basename(path)

    def _numeric_key(self, filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else 0

    def _rect2polar_input(self, X_rect):
        """
        Convert X_rect=(az, el, channels) -> (channels, 2*el, 2*el).
        """
        az_size, el_size, num_channels = X_rect.shape
        polar_size = el_size * 2
        center = el_size
        az_step = 360.0 / az_size
        el_step = 90.0 / el_size

        acc   = np.zeros((num_channels, polar_size, polar_size), dtype=np.float32)
        count = np.zeros((num_channels, polar_size, polar_size), dtype=np.int32)

        for c in range(num_channels):
            for i_az in range(az_size):
                for i_el in range(el_size):
                    val = X_rect[i_az, i_el, c]
                    el_deg = (el_size - 1 - i_el) * el_step
                    az_deg = i_az * az_step
                    az_rad = math.radians(az_deg)
                    r_pixels = (el_deg / 90.0) * el_size

                    x_c = center + r_pixels * math.cos(az_rad)
                    y_c = center + r_pixels * math.sin(az_rad)
                    x_pix = int(round(x_c))
                    y_pix = int(round(y_c))
                    if 0 <= x_pix < polar_size and 0 <= y_pix < polar_size:
                        acc[c, y_pix, x_pix] += val
                        count[c, y_pix, x_pix] += 1

        X_polar = np.zeros_like(acc, dtype=np.float32)
        mask_nonzero = (count > 0)
        X_polar[mask_nonzero] = acc[mask_nonzero] / count[mask_nonzero]
        return X_polar

    def _rect2polar_mask(self, y_rect):
        """
        Convert y_rect(az, el)->(2*el,2*el), plus morphological closing.
        """
        az_size, el_size = y_rect.shape
        polar_size = el_size * 2
        center = el_size
        az_step = 360.0 / az_size
        el_step = 90.0 / el_size

        polar_mask = np.zeros((polar_size, polar_size), dtype=np.float32)
        for i_az in range(az_size):
            for i_el in range(el_size):
                if y_rect[i_az, i_el] > 0.5:
                    az_deg = i_az * az_step
                    el_deg = (el_size - 1 - i_el) * el_step
                    az_rad = math.radians(az_deg)
                    r_pixels = (el_deg / 90.0) * el_size

                    x_c = center + r_pixels * math.cos(az_rad)
                    y_c = center + r_pixels * math.sin(az_rad)
                    x_pix = int(round(x_c))
                    y_pix = int(round(y_c))
                    if 0 <= x_pix < polar_size and 0 <= y_pix < polar_size:
                        polar_mask[y_pix, x_pix] = 1.0

        structure = np.ones((3, 3), dtype=bool)
        closed = ndimage.binary_closing(polar_mask, structure=structure, iterations=1)
        return closed.astype(np.float32)


# ----------------------- DRONE/CSV & WAV CONFIG -----------------------
CHANNELS        = 6
sample_rate     = 48000
chunk_duration_s= 0.1
chunk_duration_samples = int(chunk_duration_s * sample_rate)
RECORD_SECONDS  = 1200
lowcut, highcut = 200.0, 8000.0

azimuth_range   = np.arange(-180, 181, 4)
elevation_range = np.arange(0, 91, 4)

mic_positions, delay_samples, num_mics = initialize_beamforming_params(
    azimuth_range, elevation_range, 343, sample_rate
)

# Precompute mic positions & table of sample delays for the entire angle grid
# mic_positions, delay_samples, num_mics = initialize_beamforming_params(
#     azimuth_range,
#     elevation_range,
#     343,
#     sample_rate,
#     a_=[0, -120, -240],
#     a2_=[-40, -80, -160, -200, -280, -320],
#     h_=[1.12, 0.92, 0.77, 0.6, 0.42, 0.02],
#     r_=[0.1, 0.17, 0.25, 0.32, 0.42, 0.63]
# )

# Toggle this to False for "ambient mode" (no CSV):
# use_csv = True

# drone_config = {
#     "name": "DJI Air 3",
#     "ref_csv": "dataset/original/Nov_25_2025/Ref2/Air_Nov-25th-2024-03-19PM-Flight-Airdata.csv",
#     "flight_csv": "dataset/original/Nov_25_2025/11/Air_Nov-25th-2024-04-32PM-Flight-Airdata.csv",
#     "latitude_col": "latitude",
#     "longitude_col": "longitude",
#     "altitude_col": "altitude_above_seaLevel(feet)",
#     "time_col": "time(millisecond)",
#     "initial_azimuth": -17.0,
#     "initial_elevation": 0.0,
#     "start_index": 26,
#     "speed_cols_mph": [" xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"],
#     "corrections": [0, -240, -98, 540],
#     # "skip_seconds": 5.57,  # sync pulse
#     "skip_seconds": 149.0,   # seconds to skip from start of WAV
#     "wav_filenames": [
#         "dataset/original/Nov_25_2025/11_audio/20241125_165753_device_1_nosync_part1.wav",
#         "dataset/original/Nov_25_2025/11_audio/20241125_165757_device_2_nosync_part1.wav",
#         "dataset/original/Nov_25_2025/11_audio/20241125_165801_device_3_nosync_part1.wav",
#         "dataset/original/Nov_25_2025/11_audio/20241125_165804_device_4_nosync_part1.wav"
#     ]
# }
#
# TEST_DATA  = "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Nov_25_2025/11/"

use_csv = True
drone_config = {
    "name": "DJI Air 3",
    "ref_csv": "dataset/original/Mar_18_2025/Ref/Mar-18th-2025-10-31AM-Flight-Airdata2.csv",
    "flight_csv": "dataset/original/Mar_18_2025/2/Mar-18th-2025-11-55AM-Flight-Airdata.csv",
    "latitude_col": "latitude",
    "longitude_col": "longitude",
    "altitude_col": "altitude_above_seaLevel(feet)",
    "time_col": "time(millisecond)",
    "initial_azimuth": 15.0,
    "initial_elevation": 0.0,
    "start_index": 23,
    "speed_cols_mph": [" xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"],
    "corrections": [0, -1350, -1487, -1211],
    # "skip_seconds": 5.57,  # sync pulse
    "skip_seconds": 69.6,   # seconds to skip from start of WAV
    "wav_filenames": [
        "dataset/original/Mar_18_2025/2/Wavs/20250318_123443_device_1_nosync_part1.wav",
        "dataset/original/Mar_18_2025/2/Wavs/20250318_123526_device_2_nosync_part1.wav",
        "dataset/original/Mar_18_2025/2/Wavs/20250318_123624_device_3_nosync_part1.wav",
        "dataset/original/Mar_18_2025/2/Wavs/20250318_123730_device_4_nosync_part1.wav"
    ]
}

TEST_DATA  = "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2/"

# use_csv = True
# drone_config = {
#     "name": "DJI Air 3",
#     "ref_csv": "dataset/original/Oct_11_2024/Ref/DJIFlightRecord_2024-10-11_[14-32-34].csv",
#     "flight_csv": "dataset/original/Oct_11_2024/Oct-11th-2024-03-49PM-Flight-Airdata.csv",
#     "latitude_col": "latitude",
#     "longitude_col": "longitude",
#     "altitude_col": "altitude_above_seaLevel(feet)",
#     "time_col": "time(millisecond)",
#     "initial_azimuth": -5.0,
#     "initial_elevation": 0.0,
#     "start_index": 39,
#     "speed_cols_mph": [" xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"],
#     "corrections": [0, -866, -626, -729],
#     # "skip_seconds": 3.6,  # sync pulse
#     "skip_seconds": 82.0,   # seconds to skip from start of WAV
#     "wav_filenames": [
#         "dataset/original/Oct_11_2024/5/device_1_nosync.wav",
#         "dataset/original/Oct_11_2024/5/device_2_nosync.wav",
#         "dataset/original/Oct_11_2024/5/device_3_nosync.wav",
#         "dataset/original/Oct_11_2024/5/device_4_nosync.wav"
#     ]
# }
#
# TEST_DATA  = "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Oct_11_2024/"

# use_csv = False
# drone_config = {
#     "name": "DJI Air 3",
#     "ref_csv": "",
#     "flight_csv": "",
#     "corrections": [0, -1350, -1487, -1211],
#     # "skip_seconds": 3.37,  # sync pulse
#     "skip_seconds": 1290,  # seconds to skip from start of WAV
#     "wav_filenames": [
#         "dataset/original/Mar_18_2025/2/Wavs/20250318_123443_device_1_nosync_part1.wav",
#         "dataset/original/Mar_18_2025/2/Wavs/20250318_123526_device_2_nosync_part1.wav",
#         "dataset/original/Mar_18_2025/2/Wavs/20250318_123624_device_3_nosync_part1.wav",
#         "dataset/original/Mar_18_2025/2/Wavs/20250318_123730_device_4_nosync_part1.wav"
#     ]
# }
#
# TEST_DATA  = "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_2/"
# ----------------------- MODEL & HP CONFIG -----------------------
# model trained only on 1
# MODEL_PATH = "models/unet_20250410_020604.pth"
# HP_PATH    = "models/unet_20250410_020604_hyperparams.yaml"

# model trained on 1 and no_drone dataset 1
MODEL_PATH = "models/unet_TverskyLoss_20250412_154823.pth"
HP_PATH    = "models/unet_TverskyLoss_20250412_154823_hyperparams.yaml"

# model trained on 1 and no_drone dataset 1
# MODEL_PATH = "models/unet_CombinedLoss_20250412_181140.pth"
# HP_PATH    = "models/unet_CombinedLoss_20250412_181140_hyperparams.yaml"

# OUTPUT_VIDEO = "combined_inference_unet_20250410_020604.mp4"
# RESULTS_CSV  = "combined_inference_results_unet_20250410_020604.csv"
#
# OUTPUT_VIDEO = "combined_inference_unet_20250410_020604_train_1_test_out11.mp4"
# RESULTS_CSV  = "combined_inference_results_unet_20250410_020604_train_1_test_out11.csv"

# OUTPUT_VIDEO = "combined_inference_unet_20250410_020604_train_1_test_no_drone.mp4"
# RESULTS_CSV  = "combined_inference_results_unet_20250410_020604_train_1_test_no_drone.csv"

# testing on train data here
# OUTPUT_VIDEO = "combined_inference_unet_TverskyLoss_20250412_154823_train_1_no_drone_1_test_no_drone_2.mp4"
# RESULTS_CSV  = "combined_inference_results_unet_TverskyLoss_20250412_154823_train_1_no_drone_1_test_no_drone_2.csv"

# OUTPUT_VIDEO = "combined_inference_unet_TverskyLoss_20250412_154823_train_1_no_drone_1_test_out11.mp4"
# RESULTS_CSV  = "combined_inference_results_unet_TverskyLoss_20250412_154823_train_1_no_drone_1_test_out11.csv"

# OUTPUT_VIDEO = "combined_inference_unet_CombinedLoss_20250412_181140_train_1_no_drone_1_test_out11.mp4"
# RESULTS_CSV  = "combined_inference_unet_CombinedLoss_20250412_181140_train_1_no_drone_1_test_out11.csv"

# OUTPUT_VIDEO = "combined_inference_unet_CombinedLoss_20250412_181140_train_1_no_drone_1_test_2.mp4"
# RESULTS_CSV  = "combined_inference_unet_CombinedLoss_20250412_181140_train_1_no_drone_1_test_2.csv"

OUTPUT_VIDEO = "combined_inference_unet_TverskyLoss_20250412_154823_train_1_no_drone_1_test_2.mp4"
RESULTS_CSV  = "combined_inference_unet_TverskyLoss_20250412_154823_train_1_no_drone_1_test_2.csv"

# Load hyperparams
with open(HP_PATH, "r") as f:
    hp = yaml.safe_load(f)
print("Loaded hyperparameters:", hp)

# Extract the hyperparameters:
num_fft_bins           = hp.get("num_fft_bins", 32)
base_filters           = hp.get("base_filters", 16)
depth                  = hp.get("depth", 3)
kernel_size            = hp.get("kernel_size", 3)
attention_type         = hp.get("attention_type", "none")
convert_to_polar       = hp.get("convert_to_polar", False)
inference_thresh       = hp.get("inference_threshold", 0.5)

feature_extraction_mode= hp.get("feature_extraction_mode", "fft")
wavelet_param          = hp.get("wavelet_param", 4.0)
min_freq               = hp.get("min_freq", 200)
max_freq               = hp.get("max_freq", 8000)
bin_mask               = hp.get("bin_mask", None)

# Build the model
model = UNet(
    in_channels=num_fft_bins,  # or sum(bin_mask) if bin_mask is partially zero
    out_channels=1,
    base_filters=base_filters,
    depth=depth,
    kernel_size=kernel_size,
    attention_type=attention_type
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded UNet model from {MODEL_PATH}")

# Convert bin_mask if needed
if bin_mask is not None:
    bin_mask = np.array(bin_mask, dtype=bool)

# ---------------- Inference Dataset ----------------
dataset = InferenceChunkDataset(
    folder_path=TEST_DATA,
    sr=48000,
    min_freq=min_freq,
    max_freq=max_freq,
    num_fft_bins=num_fft_bins,
    bin_mask=bin_mask,
    feature_extraction_mode=feature_extraction_mode,
    wavelet_param=wavelet_param,
    convert_to_polar=convert_to_polar,
    debug=False
)
print(f"Loaded inference dataset: {len(dataset)} samples from {TEST_DATA}")


wav_files = open_wav_files(drone_config["wav_filenames"])
for i, wf in enumerate(wav_files):
    corr = drone_config['corrections'][i] if i < len(drone_config['corrections']) else 0
    wf.setpos(int(drone_config["skip_seconds"] * sample_rate) + corr)


# ---------------- If CSV is used, load and prep flight_data ----------------
if use_csv:
    ref_csv = drone_config['ref_csv']
    flight_csv = drone_config['flight_csv']

    latitude_col  = drone_config['latitude_col']
    longitude_col = drone_config['longitude_col']
    altitude_col  = drone_config['altitude_col']
    time_col      = drone_config['time_col']

    ref_data = pd.read_csv(ref_csv)
    flight_data = pd.read_csv(flight_csv)

    flight_data = flight_data.iloc[drone_config['start_index']:].reset_index(drop=True)
    all_cols = [latitude_col, longitude_col, altitude_col, time_col] + drone_config['speed_cols_mph']
    flight_data = flight_data[all_cols].dropna()
    ref_data = ref_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()

    # Convert altitude (feet -> meters)
    flight_data[altitude_col] *= 0.3048
    ref_data[altitude_col]    *= 0.3048
    initial_altitude = ref_data[altitude_col].iloc[0]

    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)
    reference_longitude = ref_data[longitude_col].astype(float).mean()
    reference_latitude  = ref_data[latitude_col].astype(float).mean()
    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)

    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data[longitude_col].values,
        flight_data[latitude_col].values
    )

    drone_init_az = calculate_azimuth_meters(
        ref_x, ref_y,
        flight_data.iloc[0]['X_meters'],
        flight_data.iloc[0]['Y_meters']
    )
    drone_init_el = calculate_elevation_meters(
        flight_data.iloc[0][altitude_col],
        ref_x, ref_y,
        flight_data.iloc[0]['X_meters'],
        flight_data.iloc[0]['Y_meters'],
        initial_altitude
    )
    az_offset = drone_config['initial_azimuth'] - drone_init_az
    el_offset = drone_config['initial_elevation']
else:
    # No CSV, just define placeholders
    flight_data = None
    ref_x = ref_y = 0.0  # not used
    initial_altitude = 0.0
    az_offset = 0.0
    el_offset = 0.0


# ---------------- Figure Setup (conditional subplots) ----------------
if use_csv:
    # We’ll make a 2x2 figure with XY + altitude + beam + UNet
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    ax_xy   = axs[0, 0]
    ax_alt  = axs[0, 1]
    ax_beam = axs[1, 0]
    ax_unet = axs[1, 1]

    ax_xy.set_title('Drone XY Position')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.grid(True)
    line_xy,   = ax_xy.plot([], [], 'b-', label='Path')
    marker_xy, = ax_xy.plot([], [], 'ro', label='Current Pos')
    ax_xy.legend()

    ax_alt.set_title('Drone Altitude vs Time')
    ax_alt.set_xlabel('Time (s)')
    ax_alt.set_ylabel('Altitude (m) (relative)')
    ax_alt.grid(True)
    line_alt,   = ax_alt.plot([], [], 'g-', label='Altitude Trace')
    marker_alt, = ax_alt.plot([], [], 'mo', label='Current Alt')
    ax_alt.legend()

else:
    # Ambient mode: just 1x2 figure – beam & UNet, for instance
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    ax_beam = axs[0]
    ax_unet = axs[1]

# Common beam plot
ax_beam.set_title('Beamforming Energy')
im_beam = ax_beam.imshow(
    np.zeros((len(elevation_range), len(azimuth_range))),
    extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
    origin='lower', aspect='auto', cmap='jet'
)
ax_beam.set_xlabel('Azimuth (deg)')
ax_beam.set_ylabel('Elevation (deg)')
cb_beam = fig.colorbar(im_beam, ax=ax_beam, label='Energy')
marker_max, = ax_beam.plot([], [], 'ro', label='Estimated Dir')
ref_marker, = ax_beam.plot([], [], 'kx', markersize=10, label='CSV Reference')
ax_beam.legend()

# Common UNet plot
ax_unet.set_title('UNet Inference vs Ground Truth')
im_unet = ax_unet.imshow(np.zeros((128, 128, 3)), origin='lower')
ax_unet.axis('off')
text_unet = ax_unet.text(
    0.01, 0.99, '', transform=ax_unet.transAxes,
    va="top", ha="left", color="white",
    bbox=dict(facecolor="black", alpha=0.7)
)

# Footer & info text
footer_text = fig.text(0.5, 0.005, '', va='bottom', ha='center', fontsize=10)
info_text   = fig.text(0.84, 0.7, '', va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

plt.tight_layout()


# ---------------- Prepare for writing frames ----------------
writer = FFMpegWriter(fps=10)
criterion = CombinedLoss(tau=10, epsilon=1e-6, alpha=0.5)

# Keep track for CSV output
results_list = []

if use_csv:
    xy_x = []
    xy_y = []
    alt_times = []
    alt_values = []
    max_iterations = min(len(flight_data), len(dataset))
else:
    # If no CSV, just iterate over dataset length
    max_iterations = len(dataset)

# ---------------- Main loop ----------------
with writer.saving(fig, OUTPUT_VIDEO, dpi=100):
    for i in tqdm(range(max_iterations), desc="Combined Inference", unit="frame"):
        # 1) Read next chunk of audio & beamform
        buffers = []
        for wf in wav_files:
            block = read_wav_block(wf, chunk_duration_samples, CHANNELS)
            if block is None:
                break
            buffers.append(block)

        # If no more audio blocks, stop
        if len(buffers) == 0:
            break

        combined_signal = np.hstack(buffers)
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, sample_rate)
        energy = beamform_time(filtered_signal, delay_samples)

        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth   = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]

        # 2) If CSV is in use, get the corresponding row
        if use_csv:
            row = flight_data.iloc[i]
            x = row['X_meters']
            y = row['Y_meters']
            alt_absolute = row[altitude_col]
            csv_time = row[time_col] / 1000.0

            csv_azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) + az_offset
            csv_azimuth = wrap_angle(csv_azimuth)
            # Example sign-flip if needed
            csv_azimuth = -csv_azimuth
            csv_elevation = calculate_elevation_meters(
                alt_absolute, ref_x, ref_y, x, y, initial_altitude
            ) + el_offset

            # Update XY
            xy_x.append(x - ref_x)
            xy_y.append(y - ref_y)
            line_xy.set_data(xy_x, xy_y)
            marker_xy.set_data([x - ref_x], [y - ref_y])
            ax_xy.relim()
            ax_xy.autoscale_view()

            # Update altitude
            alt_times.append(csv_time)
            alt_values.append(alt_absolute - initial_altitude)
            line_alt.set_data(alt_times, alt_values)
            marker_alt.set_data([csv_time], [alt_absolute - initial_altitude])
            ax_alt.relim()
            ax_alt.autoscale_view()

            # Mark reference in beam plot
            ref_marker.set_data([csv_azimuth], [csv_elevation])
        else:
            # No CSV => no reference
            csv_azimuth = np.nan
            csv_elevation = np.nan
            csv_time = i * chunk_duration_s  # or whatever you like

        # Common beam plot updates
        im_beam.set_data(energy.T)
        im_beam.set_clim(vmin=np.min(energy), vmax=np.max(energy))
        marker_max.set_data([estimated_azimuth], [estimated_elevation])

        # 3) UNet inference on chunk i
        X_tensor, y_tensor, filename = dataset[i]
        X_tensor = X_tensor.unsqueeze(0).to(device)  # (1, channels, H, W)
        y_tensor = y_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            outputs = model(X_tensor)
            if outputs.shape[-2:] != y_tensor.shape[-2:]:
                outputs = F.interpolate(outputs, size=y_tensor.shape[-2:],
                                        mode='bilinear', align_corners=False)

        loss_val = criterion(outputs, y_tensor).item()
        preds = (torch.sigmoid(outputs) > inference_thresh).float()

        correct = (preds == y_tensor).sum().item()
        total   = y_tensor.numel()
        acc_val = correct / total

        # create overlay
        pred_2d  = preds[0,0,:,:]
        label_2d = y_tensor[0,0,:,:]
        overlay  = create_overlay(label_2d, pred_2d)
        im_unet.set_data(overlay)
        text_unet.set_text(f"{filename}\nLoss: {loss_val:.4f}  Acc: {acc_val:.4f}")

        # 4) Update textual info
        audio_time = i * chunk_duration_s + drone_config["skip_seconds"]
        footer_str = (
            f"Audio={format_time_s(audio_time)} | "
            f"SSL=(Az={estimated_azimuth:.1f}, El={estimated_elevation:.1f})"
        )
        if use_csv:
            az_diff, el_diff = calculate_angle_difference(
                estimated_azimuth, csv_azimuth,
                estimated_elevation, csv_elevation
            )
            footer_str += (
                f" | CSV={format_time_s(csv_time)} "
                f"| CSV=(Az={csv_azimuth:.1f}, El={csv_elevation:.1f}) "
                f"| Diff=(Az={az_diff:.1f}, El={el_diff:.1f})"
            )

            # Speed info
            speed_cols = drone_config['speed_cols_mph']
            vx_mps = row[speed_cols[0]] * 0.44704
            vy_mps = row[speed_cols[1]] * 0.44704
            vz_mps = row[speed_cols[2]] * 0.44704
            total_distance = calculate_total_distance_meters(
                ref_x, ref_y, x, y, initial_altitude, alt_absolute
            )
            info_text.set_text(
                f"Alt: {(alt_absolute - initial_altitude):.2f} m\n"
                f"Dist: {total_distance:.2f} m\n"
                f"Vx: {vx_mps:.2f} m/s\n"
                f"Vy: {vy_mps:.2f} m/s\n"
                f"Vz: {vz_mps:.2f} m/s"
            )

            results_list.append({
                "audio_time_s":  audio_time,
                "csv_time_s":    csv_time,
                "az_est":        estimated_azimuth,
                "el_est":        estimated_elevation,
                "az_csv":        csv_azimuth,
                "el_csv":        csv_elevation,
                "az_diff":       az_diff,
                "el_diff":       el_diff,
                "unet_loss":     loss_val,
                "unet_acc":      acc_val
            })
        else:
            # Ambient/no CSV mode => store minimal results
            results_list.append({
                "audio_time_s":  audio_time,
                "az_est":        estimated_azimuth,
                "el_est":        estimated_elevation,
                "unet_loss":     loss_val,
                "unet_acc":      acc_val
            })

        footer_text.set_text(footer_str)

        # Render frame
        plt.pause(0.001)
        writer.grab_frame()

# Close WAV files
for wf in wav_files:
    wf.close()

plt.ioff()
plt.show()
print(f"Saved combined inference video to {OUTPUT_VIDEO}")

# Save CSV with either drone columns (if use_csv) or minimal columns
results_df = pd.DataFrame(results_list)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"Saved CSV results to {RESULTS_CSV}")
