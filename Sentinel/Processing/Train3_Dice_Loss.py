import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




#############################################
# 1. Custom Dataset for NPZ Loading (GridDataset)
#############################################
class GridDataset(Dataset):
    """
    Custom dataset for grid-based features and binary labels.
    Expects an NPZ file with keys 'X' and 'y':
      - X: numpy array of shape (num_samples, num_azimuth_points, num_elevation_points, num_bins)
      - y: numpy array of shape (num_samples, num_azimuth_points, num_elevation_points)

    The dataset rearranges X to shape (num_samples, num_bins, num_azimuth_points, num_elevation_points)
    to be compatible with PyTorch.
    """

    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = data['X'].astype(np.float32)
        self.y = data['y'].astype(np.float32)
        # Rearrange dimensions: (N, H, W, C) -> (N, C, H, W)
        self.X = np.transpose(self.X, (0, 3, 1, 2))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.tensor(x), torch.tensor(y)


####################################################
# 2. Combined Dataset that Concatenates All Instances
####################################################
class CombinedGridDataset(Dataset):
    """
    This dataset concatenates the X and y arrays from multiple GridDataset instances.
    In this way, each sample from the new datasets is added as a new instance.
    """

    def __init__(self, datasets):
        # Concatenate the X arrays along the sample axis (axis=0)
        self.X = np.concatenate([ds.X for ds in datasets], axis=0)
        # Concatenate the y arrays along the sample axis (axis=0)
        self.y = np.concatenate([ds.y for ds in datasets], axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.tensor(x), torch.tensor(y)


####################################################
# 3. Load Primary Dataset and Additional Datasets
####################################################
# Primary NPZ file path (adjust the path if needed)
npz_file_path = 'Datasets/data_fft_11.npz'
dataset_primary = GridDataset(npz_file_path)

# List of additional NPZ file paths (ensure they share the same internal structure)
#additional_npz_paths = [
#    'Datasets/data_fft_6.npz',
    #'Datasets/5_Oct_10deg.npz'
#]

# Create additional GridDataset instances
#additional_datasets = [GridDataset(path) for path in additional_npz_paths]

# Combine all datasets by concatenating their data arrays
#combined_dataset = CombinedGridDataset([dataset_primary] + additional_datasets)
#dataset = combined_dataset # using multiple datasets
dataset = dataset_primary # using just one dataset

# Retrieve the number of bins (input channels) from the concatenated X array
num_bins = dataset.X.shape[1]
print("Number of bins:", num_bins)

####################################################
# 4. Split the Dataset and Create DataLoaders
####################################################
# Split the dataset: 70% for training and 30% for testing
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


####################################################
# 5. Modified U-Net Architecture (UNetSmall)
####################################################
class UNetSmall(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A modified U-Net architecture with three downsampling steps,
        suitable for small input sizes.
        """
        super(UNetSmall, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.down1 = self.conv_block(in_channels, 64)  # Output: (N, 64, H, W)
        self.down2 = self.conv_block(64, 128)  # Output: (N, 128, H, W)
        self.down3 = self.conv_block(128, 256)  # Output: (N, 256, H, W)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)  # Output: (N, 512, H, W)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Two consecutive convolutional layers with BatchNorm and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        p3 = self.pool(c3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder path
        u3 = self.up3(bn)
        if u3.shape[2:] != c3.shape[2:]:
            u3 = nn.functional.interpolate(u3, size=c3.shape[2:])
        u3 = torch.cat([u3, c3], dim=1)
        c4 = self.conv3(u3)

        u2 = self.up2(c4)
        if u2.shape[2:] != c2.shape[2:]:
            u2 = nn.functional.interpolate(u2, size=c2.shape[2:])
        u2 = torch.cat([u2, c2], dim=1)
        c5 = self.conv2(u2)

        u1 = self.up1(c5)
        if u1.shape[2:] != c1.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=c1.shape[2:])
        u1 = torch.cat([u1, c1], dim=1)
        c6 = self.conv1(u1)

        output = self.final_conv(c6)
        return output


# Instantiate the model with the number of bins as input channels
model = UNetSmall(in_channels=num_bins, out_channels=1).to(device)

####################################################
# 6. Define Loss, Optimizer, and Metrics
####################################################



class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convertir a probabilidades
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=0.5):
        """
        pos_weight: tensor que da mayor peso a la clase positiva.
        alpha: factor para combinar BCE y Dice Loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

pos_weight = torch.tensor([90.0]).to(device)  # Ajusta este valor según el desbalance observado
criterion = BCEDiceLoss(pos_weight=pos_weight, alpha=0.5)
#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Combines a sigmoid layer with binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)


####################################################
# 7. Evaluation and Training Functions
####################################################
def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset, calculating the average loss and pixel-wise accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)  # Add channel dimension
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            # Apply sigmoid and threshold to obtain binary predictions
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            correct += (preds == labels).sum().item()
            total += torch.numel(labels)
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    """
    Training function that calculates and stores loss and accuracy metrics
    for both training and testing datasets at each epoch.
    """
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)  # (batch, 1, H, W)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

        # Evaluate on both training and testing datasets
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return train_losses, train_accuracies, test_losses, test_accuracies


####################################################
# 8. Run Training
####################################################
num_epochs = 20  # Adjust the number of epochs as needed
train_losses, train_accuracies, test_losses, test_accuracies = train_model(
    model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs
)

# Plot training and testing loss and accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model's state_dict to a file.
#model_save_path = 'model_fft_Air_11_6_loss.pth'
#model_save_path = 'model_fft_Air_11_BCEWeight.pth'
torch.save(model.state_dict(), model_save_path)
print("Model saved to", model_save_path)
