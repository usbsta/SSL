import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchviz import make_dot
import torch.nn.functional as F
from matplotlib.pyplot import tight_layout, show, subplots
import scipy.ndimage as ndimage

import optuna
from optuna.exceptions import TrialPruned
from tqdm import tqdm
import fcwt

# -------------------------
# 1) DEVICE SELECTION
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# 2) CUSTOM DATASET WITH OPTIONAL WAVELET EXTRACTION
# -------------------------
class ChunkNPZDataset(Dataset):
    """
    Loads .npz files in the new format:
        beamformed_data: shape (az, el, n_samples) [time-domain signals]
        labels: shape (az, el)                     [0 or 1 mask]

    For each .npz (treated as "one chunk"):
      If feature_extraction_mode="fft":
        1) Perform FFT on each (az, el) signal => magnitude spectrum
        2) Restrict to [min_freq, max_freq]
        3) Global normalization
        4) Binning => apply bin_mask
        5) (Optionally) convert to polar

      If feature_extraction_mode="wavelet":
        1) Perform wavelet transform (Morlet) on each (az, el) signal
           for frequencies in [min_freq, max_freq] with num_fft_bins steps
        2) Average across time => a single amplitude per frequency
        3) Global normalization
        4) (Optionally) apply bin_mask (same shape => (az, el, freq))
        5) (Optionally) convert to polar

    final output shape if rectangular:
        X => (sum(bin_mask), az, el)
        y => (az, el)
    or if polar:
        X => (sum(bin_mask), 2*el, 2*el)
        y => (2*el, 2*el)

    Also supports multiple folders:
        folder_path: str or list of str
    """

    def __init__(
        self,
        folder_path,
        sr=48000,                # sampling rate (Hz)
        min_freq=200,           # minimum frequency (Hz)
        max_freq=8000,          # maximum frequency (Hz)
        num_fft_bins=32,        # number of frequency bins (for both FFT and wavelet)
        bin_mask=None,          # boolean mask for final frequency dimension
        convert_to_polar: bool = False,
        feature_extraction_mode: str = "fft",  # "fft" or "wavelet"
        wavelet_param: float = 4.0,           # e.g. Morlet parameter
        debug: bool = False
    ):
        # Gather .npz file paths from one or multiple directories
        self.file_paths = []
        if isinstance(folder_path, list):
            for single_path in folder_path:
                self.file_paths.extend(
                    sorted(glob.glob(os.path.join(single_path, '*.npz')))
                )
        else:
            self.file_paths.extend(
                sorted(glob.glob(os.path.join(folder_path, '*.npz')))
            )
        if len(self.file_paths) == 0:
            raise ValueError(f"No NPZ files found in: {folder_path}")

        self.sr = sr
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_fft_bins = num_fft_bins
        self.bin_mask = bin_mask
        self.convert_to_polar = convert_to_polar
        self.feature_extraction_mode = feature_extraction_mode.lower()
        self.wavelet_param = wavelet_param
        self.debug = debug

        # If no bin_mask is provided, default = all True with length = num_fft_bins
        if self.bin_mask is None:
            self.bin_mask = np.ones((self.num_fft_bins,), dtype=bool)
        if len(self.bin_mask) != self.num_fft_bins:
            raise ValueError(
                f"bin_mask length {len(self.bin_mask)} does not match num_fft_bins {self.num_fft_bins}."
            )

        # Pre-initialize wavelet objects if in wavelet mode
        # (so we don't re-create them for each sample).
        if self.feature_extraction_mode == "wavelet":
            self.morl = fcwt.Morlet(self.wavelet_param)  # e.g. Morlet with param=4.0
            # We'll create a set of scales that correspond to min_freq..max_freq
            # at linear intervals => "fn" = num_fft_bins
            # fcwt.FCWT_LINFREQS => linearly spaced in frequency domain
            self.scales = fcwt.Scales(
                self.morl,
                fcwt.FCWT_LINFREQS,
                self.sr,
                self.min_freq,  # f0
                self.max_freq,  # f1
                self.num_fft_bins
            )
            # Create the fcwt object
            self.fcwt_obj = fcwt.FCWT(
                pwav=self.morl,
                pthreads=8,
                puse_optimalization_schemes=False,
                puse_normalization=False
            )
            # Precompute an array of frequencies so we know which freq each row corresponds to
            self.fcwt_freqs = np.zeros(self.num_fft_bins, dtype='single')
            self.scales.getFrequencies(self.fcwt_freqs)
            # This is mostly for reference if you need freq axis

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        beamformed_data = data['beamformed_data']  # shape (az, el, n_samples)
        labels_rect = data['labels']               # shape (az, el)

        az_size, el_size, n_samples = beamformed_data.shape

        if self.feature_extraction_mode == "wavelet":
            # --------------- Wavelet approach ---------------
            # We'll build a 3D array: (az, el, num_fft_bins)
            # Each (az, el) => cwt => shape (num_fft_bins, n_samples).
            # Then we average over time => shape (num_fft_bins,).
            all_features = np.zeros((az_size, el_size, self.num_fft_bins), dtype=np.float32)

            for i_az in range(az_size):
                for i_el in range(el_size):
                    signal_td = beamformed_data[i_az, i_el, :].astype(np.float32)

                    # output array for wavelet => shape (num_fft_bins, n_samples)
                    cwt_output = np.zeros((self.num_fft_bins, n_samples), dtype=np.complex64)

                    self.fcwt_obj.cwt(signal_td, self.scales, cwt_output)

                    # Convert to magnitude & average across time
                    # shape => (num_fft_bins,)
                    mag_avg = np.mean(np.abs(cwt_output), axis=1)
                    all_features[i_az, i_el, :] = mag_avg

        else:
            # --------------- FFT approach ---------------
            freq_resolution = self.sr / n_samples
            freqs = np.fft.rfftfreq(n_samples, d=1.0/self.sr)
            freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            freq_in_range = np.sum(freq_mask)

            if freq_in_range < self.num_fft_bins:
                raise ValueError(
                    f"freq_in_range ({freq_in_range}) < num_fft_bins ({self.num_fft_bins}); cannot bin upward."
                )

            # We'll accumulate the magnitude spectra in (az, el, freq_in_range)
            all_ffts = np.zeros((az_size, el_size, freq_in_range), dtype=np.float32)
            for i_az in range(az_size):
                for i_el in range(el_size):
                    signal_td = beamformed_data[i_az, i_el, :]
                    fft_vals = np.fft.rfft(signal_td)
                    mag = np.abs(fft_vals).astype(np.float32)
                    all_ffts[i_az, i_el, :] = mag[freq_mask]

            # 1) Global normalization for all_ffts
            mean_val = all_ffts.mean()
            std_val = all_ffts.std() + 1e-8
            all_ffts = (all_ffts - mean_val) / std_val

            # 2) Binning from freq_in_range => num_fft_bins
            bin_factor = freq_in_range // self.num_fft_bins
            if bin_factor == 0:
                bin_factor = 1
                self.num_fft_bins = freq_in_range
            freq_in_range_effective = bin_factor * self.num_fft_bins
            if freq_in_range_effective < freq_in_range:
                all_ffts = all_ffts[:, :, :freq_in_range_effective]
                freq_in_range = freq_in_range_effective

            # reshape & average => shape (az, el, num_fft_bins)
            all_ffts = all_ffts.reshape(
                az_size,
                el_size,
                self.num_fft_bins,
                bin_factor
            ).mean(axis=-1)

            # We'll store this result in 'all_features'
            all_features = all_ffts

        # --------------- Now we do the remainder (global normalization + bin_mask) ---------------
        if self.feature_extraction_mode == "wavelet":
            # Wavelet produced shape => (az, el, num_fft_bins). We'll do global normalization now.
            mean_val = all_features.mean()
            std_val = all_features.std() + 1e-8
            all_features = (all_features - mean_val) / std_val

        # Apply bin_mask => shape (az, el, sum(bin_mask))
        all_features = all_features[:, :, self.bin_mask]

        # Keep label as is => shape (az, el)
        y_rect = labels_rect

        # --------------- Convert to polar if needed ---------------
        if self.convert_to_polar:
            X_polar = self._rect2polar_input(all_features)
            y_polar = self._rect2polar_mask(y_rect)
            X_tensor = torch.tensor(X_polar, dtype=torch.float32)
            y_tensor = torch.tensor(y_polar, dtype=torch.float32)
            if self.debug and idx == 0:
                self._debug_plot_label(y_rect, y_polar)
        else:
            # shape => (az, el, sum_bin_mask) => transpose => (sum_bin_mask, az, el)
            X_rect_trans = np.transpose(all_features, (2, 0, 1))
            X_tensor = torch.tensor(X_rect_trans, dtype=torch.float32)
            y_tensor = torch.tensor(y_rect, dtype=torch.float32)

        return X_tensor, y_tensor

    def _rect2polar_input(self, X):
        """
        Convert rectangular X (az_size, el_size, freq_channels)
        => shape (freq_channels, 2*el_size, 2*el_size) in "polar".
        We'll do a naive approach that accumulates data in a polar grid.
        """
        az_size, el_size, num_channels = X.shape
        polar_size = el_size * 2
        center = el_size

        az_step = 360.0 / az_size
        el_step = 90.0 / el_size

        acc   = np.zeros((num_channels, polar_size, polar_size), dtype=np.float32)
        count = np.zeros((num_channels, polar_size, polar_size), dtype=np.int32)

        for c in range(num_channels):
            for i_az in range(az_size):
                for i_el in range(el_size):
                    val = X[i_az, i_el, c]
                    # invert el => top => small radius
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

    def _rect2polar_mask(self, y):
        """
        Convert rectangular y (az_size x el_size) => polar mask (2*el x 2*el),
        then apply a binary closing to fill small holes.
        """
        az_size, el_size = y.shape
        if az_size == 0 or el_size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        polar_size = el_size * 2
        center = el_size

        az_step = 360.0 / az_size
        el_step = 90.0 / el_size

        polar_mask = np.zeros((polar_size, polar_size), dtype=np.float32)
        for i_az in range(az_size):
            for i_el in range(el_size):
                if y[i_az, i_el] > 0.5:
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
        polar_mask = closed.astype(np.float32)
        return polar_mask

    def _debug_plot_label(self, y_rect, y_polar):
        fig, axes = subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(y_rect.T, origin='lower', aspect='auto', cmap='gray')
        axes[0].set_title("Rect Label (Az x El)")
        axes[0].set_xlabel("Az index")
        axes[0].set_ylabel("El index")

        axes[1].imshow(y_polar, origin='lower', cmap='gray')
        axes[1].set_title("Polar Label (2*El x 2*El)")
        axes[1].set_xlabel("X pixel")
        axes[1].set_ylabel("Y pixel")
        tight_layout()
        show()


def create_dataloaders(
    train_folder,
    test_folder,
    sr=48000,
    min_freq=200,
    max_freq=8000,
    num_fft_bins=32,
    bin_mask=None,
    batch_size=16,
    num_workers=0,
    convert_to_polar=False,
    feature_extraction_mode="fft",
    wavelet_param=4.0,
    debug=False
):
    """
    Creates train/test DataLoaders from ChunkNPZDataset, either using FFT or wavelet features.
    """
    train_dataset = ChunkNPZDataset(
        folder_path=train_folder,
        sr=sr,
        min_freq=min_freq,
        max_freq=max_freq,
        num_fft_bins=num_fft_bins,
        bin_mask=bin_mask,
        convert_to_polar=convert_to_polar,
        feature_extraction_mode=feature_extraction_mode,
        wavelet_param=wavelet_param,
        debug=debug
    )
    test_dataset = ChunkNPZDataset(
        folder_path=test_folder,
        sr=sr,
        min_freq=min_freq,
        max_freq=max_freq,
        num_fft_bins=num_fft_bins,
        bin_mask=bin_mask,
        convert_to_polar=convert_to_polar,
        feature_extraction_mode=feature_extraction_mode,
        wavelet_param=wavelet_param,
        debug=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # The # of input channels = sum(bin_mask)
    in_channels = sum(bin_mask) if bin_mask is not None else num_fft_bins
    return train_loader, test_loader, in_channels


# -------------------------
# 3) ATTENTION MODULE
# -------------------------
class AttentionGate(nn.Module):
    """
    A simple "Attention Gate" block, loosely inspired by Attention U-Net:
    We use gating signal 'g' to focus on skip connection 'x'.
    """

    def __init__(self, in_channels_x, in_channels_g, inter_channels=None):
        super(AttentionGate, self).__init__()
        if inter_channels is None:
            # Choose smaller intermediate channels for gating
            inter_channels = in_channels_x // 2

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g => gating signal (decoder feature)
        x => skip connection
        """
        # Project
        theta_x = self.W_x(x)
        phi_g   = self.W_g(g)

        # sum + ReLU
        f = self.relu(theta_x + phi_g)

        # 1x1 conv => gating coefficient
        psi_f = self.psi(f)

        # multiply gate coefficient into skip connection
        out = x * psi_f
        return out


# -------------------------
# 4) U-NET MODEL WITH OPTIONAL ATTENTION
# -------------------------
class UNet(nn.Module):
    """
    U-Net that operates on (in_channels, H, W).
    We add optional attention:
      - skip => apply attention gates on skip connections
      - bottleneck => apply attention in the bottleneck
      - both => skip + bottleneck
      - none => no attention
    """

    def __init__(self, in_channels, out_channels=1,
                 base_filters=64,
                 depth=3,
                 kernel_size=3,
                 attention_type="none"):
        super(UNet, self).__init__()
        assert depth >= 2, "depth must be >= 2"
        self.depth = depth
        self.kernel_size = kernel_size
        self.attention_type = attention_type.lower()  # "none", "skip", "bottleneck", or "both"

        # Flags
        self.use_attn_skip = ("skip" in self.attention_type)
        self.use_attn_bottleneck = ("bottleneck" in self.attention_type)

        # 1) Down blocks
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2) Up blocks, up-convs
        self.up_convs = nn.ModuleList()
        self.conv_after_concat = nn.ModuleList()

        # 3) Attention gates for skip
        if self.use_attn_skip:
            # We'll create one attention gate per skip connection
            # i.e. for each level in the decoder
            self.attn_gates = nn.ModuleList()

        current_filters = base_filters
        # Encoder
        self.down_blocks.append(self._conv_block(in_channels, current_filters))
        for _ in range(1, depth):
            self.down_blocks.append(self._conv_block(current_filters, current_filters * 2))
            current_filters *= 2

        # Bottleneck
        self.bottleneck = self._conv_block(current_filters, current_filters * 2)
        bottleneck_filters = current_filters * 2

        # Optional attention in bottleneck
        if self.use_attn_bottleneck:
            self.bottleneck_attention = AttentionGate(
                in_channels_x=bottleneck_filters,
                in_channels_g=bottleneck_filters
            )

        # Decoder
        for i in range(depth):
            # Up-conv
            self.up_convs.append(nn.ConvTranspose2d(
                bottleneck_filters,
                bottleneck_filters // 2,
                kernel_size=2, stride=2
            ))
            # Concat conv block
            self.conv_after_concat.append(
                self._conv_block(bottleneck_filters, bottleneck_filters // 2)
            )

            # If skip-attention, add an AttentionGate for each skip
            if self.use_attn_skip:
                in_ch_x = current_filters  # skip has 'current_filters'
                in_ch_g = bottleneck_filters // 2  # gating from upsample
                self.attn_gates.append(
                    AttentionGate(in_ch_x, in_ch_g)
                )

            bottleneck_filters //= 2
            current_filters //= 2

        self.final_conv = nn.Conv2d(bottleneck_filters, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        pad = self.kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # ----------- Encoder -----------
        encoder_outputs = []
        cur = x
        for i, block in enumerate(self.down_blocks):
            c = block(cur)
            encoder_outputs.append(c)
            if i < self.depth:
                cur = self.pool(c)

        # ----------- Bottleneck -----------
        bn = self.bottleneck(cur)
        if self.use_attn_bottleneck:
            # apply bottleneck attention
            bn = self.bottleneck_attention(bn, bn)

        # ----------- Decoder -----------
        d = bn
        for i in range(self.depth):
            up = self.up_convs[i](d)
            # skip from encoder
            skip = encoder_outputs[self.depth - 1 - i]

            # If shapes differ (due to integer division), interpolate
            if up.shape[2:] != skip.shape[2:]:
                up = F.interpolate(up, size=skip.shape[2:])

            # If skip attention is used, gate the skip connection
            if self.use_attn_skip:
                skip = self.attn_gates[i](g=up, x=skip)

            concat = torch.cat([up, skip], dim=1)
            d = self.conv_after_concat[i](concat)

        out = self.final_conv(d)
        return out


# -------------------------
# 5) LOSSES
# -------------------------
class AtLeastOneMatchLoss(nn.Module):
    def __init__(self, tau=10, epsilon=1e-6):
        super(AtLeastOneMatchLoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        p = torch.sigmoid(outputs).squeeze(1)
        targets = targets.float()
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Dilate ground truth
        dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1).squeeze(1)
        mask = (dilated > 0.5).float()
        mask_sum = mask.sum(dim=(1, 2))

        sm = (1 / self.tau) * torch.log(
            torch.sum(torch.exp(self.tau * p) * mask, dim=(1, 2)) + self.epsilon
        )
        sm = torch.where(mask_sum > 0, sm, torch.zeros_like(sm))

        loss = -torch.log(sm + self.epsilon)
        valid_loss = loss[mask_sum > 0]
        if valid_loss.numel() > 0:
            return valid_loss.mean()
        else:
            return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, tau=10, epsilon=1e-6, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.at_least_loss = AtLeastOneMatchLoss(tau=tau, epsilon=epsilon)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        loss1 = self.at_least_loss(outputs, targets)
        loss2 = self.bce_loss(outputs, targets)
        return self.alpha * loss1 + (1 - self.alpha) * loss2

class TverskyLoss(nn.Module):
    """
    Tversky loss for binary segmentation.
    This variant applies:
      - Standard Tversky loss for samples that have at least one foreground pixel.
      - A false-positive-based penalty for samples that contain no foreground.

    Args:
        alpha (float): Weight factor for false positives in the denominator.
        beta (float): Weight factor for false negatives in the denominator.
        gamma (float): Multiplier to emphasize true positives in the Tversky computation.
        eps (float): Small epsilon to prevent division by zero.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=10, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Computes the Tversky-based loss.

        Args:
            logits: Tensor of shape [N, 1, H, W], raw unnormalized model outputs.
            targets: Tensor of shape [N, 1, H, W] or [N, H, W], binary ground-truth labels.
                     Values should be 0 or 1, indicating background or foreground pixels.

        Returns:
            A scalar tensor representing the average loss across all samples in the batch.
        """
        # 1) Convert the model's unnormalized outputs (logits) to probabilities [0, 1].
        probs = torch.sigmoid(logits)  # shape: [N, 1, H, W]

        # Prepare to accumulate individual sample losses.
        batch_size = probs.size(0)
        losses = []

        # 2) Compute the loss on a per-sample basis.
        for i in range(batch_size):
            # Flatten the predicted probabilities and ground-truth masks
            # so we can easily compute TP, FP, FN for the entire image.
            probs_flat = probs[i].view(-1)       # shape: (H*W,)
            targets_flat = targets[i].view(-1)   # shape: (H*W,)

            # Check whether this sample has any foreground pixels.
            if targets_flat.sum() > 0:
                # -- SAMPLE HAS FOREGROUND --

                # Calculate True Positives, False Positives, and False Negatives.
                # Note: Multiplying predicted probs_flat with targets_flat
                #       isolates the predicted probability only in the foreground regions.
                tp = (probs_flat * targets_flat).sum()
                # False positives occur where model predicts positive but the ground truth is 0.
                fp = ((1 - targets_flat) * probs_flat).sum()
                # False negatives occur where model predicts 0 but the ground truth is 1.
                fn = (targets_flat * (1 - probs_flat)).sum()

                # Tversky index with an extra 'gamma' multiplier on true positives
                # to "reward" them more heavily:
                #     Tversky = gamma * TP / (gamma*TP + alpha*FP + beta*FN + eps)
                tversky_index = (self.gamma * tp) / (
                    self.gamma * tp + self.alpha * fp + self.beta * fn + self.eps
                )

                # Convert similarity measure (Tversky index) to a loss.
                loss_sample = 1.0 - tversky_index

            else:
                # -- SAMPLE HAS NO FOREGROUND --
                # Instead of Tversky, we penalize the model for predicting positives
                # in an all-background sample. Here, we simply take the sum of predicted
                # probabilities (i.e. the total predicted "foreground mass") and divide by 100
                # to scale it down. If the model predicts more positives, this portion
                # of the loss increases.
                #
                # You can adjust this strategy by changing the denominator or using a mean.
                # The idea is to encourage the model not to generate false positives where
                # the ground truth has none.
                loss_sample = probs_flat.sum() / 100.0

            losses.append(loss_sample)

        # 3) Average the losses across all samples in the batch and return.
        return torch.stack(losses).mean()

# -------------------------
# 6) TRAIN & EVALUATION
# -------------------------
def train_for_epochs(model, train_loader, optimizer, criterion, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in epoch_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_bar.set_postfix(loss=loss.item())
    return model


def evaluate_model(model, dataloader, criterion, device, threshold):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    eval_bar = tqdm(dataloader, desc="Evaluation")
    with torch.no_grad():
        for inputs, labels in eval_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


# -------------------------
# 7) OPTUNA OBJECTIVE
# -------------------------
def objective(trial):
    # Example hyperparams for wavelet vs. fft
    # feature_mode = trial.suggest_categorical("feature_extraction_mode", ["fft", "wavelet"])
    feature_mode = trial.suggest_categorical("feature_extraction_mode", ["fft"])

    # wavelet_param (Morlet parameter)
    if feature_mode == "wavelet":
        wavelet_param = trial.suggest_categorical("wavelet_param", [2.0, 4.0, 6.0])
    else:
        wavelet_param = 0

    # number of final FFT bins
    num_fft_bins = trial.suggest_categorical("num_fft_bins", [8, 16, 32])
    #num_fft_bins = trial.suggest_categorical("num_fft_bins", [16])

    # min/max freq in [100..8000]
    # min_freq = trial.suggest_int("min_freq", 100, 1900, step=200)
    # max_freq = trial.suggest_int("max_freq", min_freq + 200, 8000, step=200)
    min_freq = trial.suggest_categorical("min_freq", [200])
    max_freq = trial.suggest_categorical("max_freq", [2300])

    # base_filters = trial.suggest_categorical("base_filters", [8, 16, 32])
    base_filters = trial.suggest_categorical("base_filters", [32, 64, 128])
    depth = trial.suggest_categorical("depth", [3, 4, 5])
    lr = trial.suggest_categorical("lr", [0.001, 0.005, 0.01])
    # polar_setting = trial.suggest_categorical("polar", [False, True])
    polar_setting = trial.suggest_categorical("polar", [True])
    k_size = trial.suggest_categorical("kernel_size", [3])
    #threshold = trial.suggest_categorical("inference_threshold", [0.1, 0.3, 0.5, 0.7, 0.9]) # This is not used for the loss function


    # NEW: attention hyperparameter
    attention_type = trial.suggest_categorical("attention_type", ["none", "skip", "bottleneck", "both"])

    # Build a bin_mask for the final freq dimension = num_fft_bins
    bin_mask_list = []
    for i in range(num_fft_bins):
        # val = trial.suggest_int(f"mask_bin_{i}", 0, 1)
        val = trial.suggest_categorical(f"mask_bin_{i}", [1])
        bin_mask_list.append(val)
    if sum(bin_mask_list) == 0:
        # Must keep at least one bin
        raise optuna.exceptions.TrialPruned("All bins in the mask are 0 => no valid bins.")
    bin_mask = np.array(bin_mask_list, dtype=bool)

    print("\n=== Trial Parameters ===")
    print(f"feature_extraction_mode = {feature_mode}")
    print(f"wavelet_param = {wavelet_param}")
    print(f"num_fft_bins  = {num_fft_bins}")
    print(f"min_freq      = {min_freq}")
    print(f"max_freq      = {max_freq}")
    print(f"base_filters  = {base_filters}")
    print(f"depth         = {depth}")
    print(f"lr            = {lr}")
    print(f"polar         = {polar_setting}")
    print(f"kernel_size   = {k_size}")
    print(f"attention_type= {attention_type}")
    print(f"bin_mask      = {bin_mask_list} (sum={sum(bin_mask_list)})")

    try:
        # train_folder = [
        #     "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/1/",
        #     #"/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2/"
        #     "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_1/",
        # ]
        # test_folder = [
        #     # "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Nov_25_2025/11/"
        #     #"/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2/"
        #     "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_2/",
        #     ]

        train_folder = [
            "/mnt/data/Datasets/processed_dataset_drones/Mar_18_2025/1/",
            "/mnt/data/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_1/"
        ]
        test_folder = [
            "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Nov_25_2025/11/",
            "/mnt/data/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_2/"
        ]

        train_loader, test_loader, in_channels = create_dataloaders(
            train_folder,
            test_folder,
            sr=48000,
            min_freq=min_freq,
            max_freq=max_freq,
            num_fft_bins=num_fft_bins,
            bin_mask=bin_mask,
            batch_size=16,
            num_workers=0,
            convert_to_polar=polar_setting,
            feature_extraction_mode=feature_mode,
            wavelet_param=wavelet_param,
            debug=False
        )

        model = UNet(
            in_channels=in_channels,
            out_channels=1,
            base_filters=base_filters,
            depth=depth,
            kernel_size=k_size,
            attention_type=attention_type
        ).to(device)

        #criterion = CombinedLoss(tau=10, epsilon=1e-6, alpha=0.5)
        criterion = TverskyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model = train_for_epochs(model, train_loader, optimizer, criterion, device, num_epochs=2)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, threshold=0.5)

        return test_loss

    except (ValueError, RuntimeError) as e:
        print(f"Trial failed with error: {e}")
        raise TrialPruned(str(e))


# -------------------------
# 8) MAIN
# -------------------------
if __name__ == "__main__":
    db_path = "sqlite:///optuna_study_polar_attn.db"
    study_name = "combined_loss_study_polar_attn"

    study = optuna.create_study(
        direction="minimize",
        storage=db_path,
        study_name=study_name,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=100, n_jobs=1)

    print("\n=== BEST TRIAL ===")
    best_trial = study.best_trial
    print(f"  Value (Test Loss): {best_trial.value}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
