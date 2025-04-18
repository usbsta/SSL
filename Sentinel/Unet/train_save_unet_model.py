#!/usr/bin/env python

"""
train_unet_model.py

A standalone script that imports functions/classes from unet_optuna.py,
trains a U-Net with a specified set of hyperparameters, and then:
  1) saves the model weights to e.g. "unet_YYYYMMDD_HHMMSS.pth"
  2) saves the hyperparameters to a separate YAML file, e.g. "unet_YYYYMMDD_HHMMSS_hyperparams.yaml"

Both files contain a timestamp for easy versioning and reference.
"""

import os
import datetime
import yaml   # pip install pyyaml
import torch
from torchsummary import summary

# 1) Import relevant components from unet_optuna.py
from unet_optuna import (
    device,
    create_dataloaders,
    UNet,
    CombinedLoss,
    TverskyLoss,
    AtLeastOneMatchLoss,
    train_for_epochs,
    evaluate_model
)

def train_and_save_unet(
    train_folder,
    test_folder,
    hyperparams=None,
    out_dir="models"
):
    """
    Train a UNet model with given hyperparams, then save model weights
    and hyperparams in separate files, both with timestamped filenames.

    Args:
        train_folder (str or list): Path(s) to training dataset folder(s)
        test_folder  (str or list): Path(s) to testing dataset folder(s)
        hyperparams  (dict): Dictionary of hyperparameters
        out_dir      (str): Output directory for saved files
    """

    # 1) Default hyperparams if none provided
    if hyperparams is None:
        # Example defaults
        hyperparams = {
            "train_folder": train_folder,
            "test_folder": test_folder,
            "feature_extraction_mode": "fft",    # or "wavelet"
            "wavelet_param": 4.0,               # Morlet wavelet parameter (used if mode="wavelet")
            "min_freq": 200,
            "max_freq": 8000,
            "num_fft_bins": 32,
            "bin_mask": None,                   # or a 1D boolean array of length num_fft_bins
            "base_filters": 16,
            "depth": 3,
            "kernel_size": 3,
            "lr": 1e-4,
            "batch_size": 16,
            "num_workers": 0,
            "convert_to_polar": False,
            "epochs": 5,
            "attention_type": "none",           # can be "none", "skip", "bottleneck", or "both"
            "inference_threshold": 0.5
        }

    # 2) Print hyperparams
    print("=== Using Hyperparameters ===")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")

    # 3) Create dataloaders
    train_loader, test_loader, in_channels = create_dataloaders(
        train_folder=train_folder,
        test_folder=test_folder,
        sr=48000,
        min_freq=hyperparams["min_freq"],
        max_freq=hyperparams["max_freq"],
        num_fft_bins=hyperparams["num_fft_bins"],
        bin_mask=hyperparams.get("bin_mask", None),
        batch_size=hyperparams["batch_size"],
        num_workers=hyperparams["num_workers"],
        convert_to_polar=hyperparams["convert_to_polar"],
        feature_extraction_mode=hyperparams["feature_extraction_mode"],
        wavelet_param=hyperparams["wavelet_param"],
        debug=False
    )
    print(f"Input channels (in_channels) = {in_channels}")

    # 4) Build model
    model = UNet(
        in_channels=in_channels,
        out_channels=1,
        base_filters=hyperparams["base_filters"],
        depth=hyperparams["depth"],
        kernel_size=hyperparams["kernel_size"],
        attention_type=hyperparams["attention_type"]
    ).to(device)

    # 5) Define Loss & Optimizer
    # criterion = CombinedLoss(tau=10, epsilon=1e-6, alpha=0.5)
    criterion = TverskyLoss()
    # criterion = AtLeastOneMatchLoss(tau=10, epsilon=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    # 6) Train
    model = train_for_epochs(
        model, train_loader, optimizer, criterion, device,
        num_epochs=hyperparams["epochs"]
    )

    # 7) Evaluate
    # The evaluate_model in unet_optuna has a 'threshold' parameter:
    test_loss, test_acc = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        threshold=hyperparams["inference_threshold"]
    )
    print(f"Final Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # 8) Save with timestamped filenames
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model file
    model_filename = f"unet_{timestamp}.pth"
    model_path = os.path.join(out_dir, model_filename)

    # Hyperparams file
    hp_filename = f"unet_{timestamp}_hyperparams.yaml"
    hp_path = os.path.join(out_dir, hp_filename)

    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")

    # Save hyperparams in a YAML
    with open(hp_path, "w") as f:
        yaml.safe_dump(hyperparams, f)
    print(f"Hyperparams saved to: {hp_path}")

    print("Done.")


if __name__ == "__main__":
    # Example usage
    # train_folder = [
    #     "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/1/",
    #
    #     # "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone/",
    #     "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_1/",
    # ]
    # test_folder = [
    #     # "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Nov_25_2025/11/"
    #     "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Mar_18_2025/2/",
    # ]

    train_folder = [
        "/mnt/data/Datasets/processed_dataset_drones/Mar_18_2025/1/",
        "/mnt/data/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_1/"
    ]
    test_folder = [
        "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Nov_25_2025/11/",
        "/mnt/data/Datasets/processed_dataset_drones/Mar_18_2025/2_no_drone_2/"
    ]
    OUTPUT_DIR = "models"

    # Example hyperparameters
    hp = {
        "train_folder": train_folder,
        "test_folder": test_folder,
        "feature_extraction_mode": "fft",    # or "wavelet"
        "wavelet_param": 0,                 # used if feature_extraction_mode="wavelet"
        "min_freq": 200,
        "max_freq": 2300,
        "num_fft_bins": 16,
        "bin_mask": None,
        "base_filters": 32,
        "depth": 5,
        "kernel_size": 3,
        "lr": 0.001,
        "batch_size": 16,
        "num_workers": 0,
        "convert_to_polar": True,
        "epochs": 10,
        "attention_type": "skip",           # "none", "skip", "bottleneck", or "both"
        "inference_threshold": 0.1
    }

    train_and_save_unet(
        train_folder=train_folder,
        test_folder=test_folder,
        hyperparams=hp,
        out_dir=OUTPUT_DIR
    )
