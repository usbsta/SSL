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
        # Rearranger dimensiones: (N, H, W, C) -> (N, C, H, W)
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
    Each sample from the new dataset es una nueva instancia.
    """

    def __init__(self, datasets):
        # Concatenar las X a lo largo del eje de muestras (axis=0)
        self.X = np.concatenate([ds.X for ds in datasets], axis=0)
        # Concatenar las y a lo largo del eje de muestras (axis=0)
        self.y = np.concatenate([ds.y for ds in datasets], axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.tensor(x), torch.tensor(y)


####################################################
# 3. Load Primary Dataset (y opcionalmente, datasets adicionales)
####################################################
npz_file_path = 'Datasets/data_fft_11.npz'
dataset_primary = GridDataset(npz_file_path)

# Si tienes datasets adicionales, puedes combinarlos:
# additional_npz_paths = [
#    'Datasets/data_fft_6.npz',
#    'Datasets/5_Oct_10deg.npz'
# ]
# additional_datasets = [GridDataset(path) for path in additional_npz_paths]
# combined_dataset = CombinedGridDataset([dataset_primary] + additional_datasets)
# dataset = combined_dataset
dataset = dataset_primary  # Usando solo un dataset

# Obtener el número de bins (canales de entrada) a partir de X
num_bins = dataset.X.shape[1]
print("Number of bins:", num_bins)

####################################################
# 4. Split the Dataset and Create DataLoaders
####################################################
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


####################################################
# 5. Modified U-Net Architecture (UNetSmall)
####################################################
class UNetSmall(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        U-Net modificado con tres pasos de downsampling.
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
        Bloque convolucional compuesto por dos capas convolucionales seguidas de BatchNorm y ReLU.
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
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        p3 = self.pool(c3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder
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


# Instanciar el modelo
model = UNetSmall(in_channels=num_bins, out_channels=1).to(device)


####################################################
# 6. Focal Loss, Optimizer y other parameters
####################################################
class AtLeastOneMatchLoss(nn.Module):
    def __init__(self, tau=10, epsilon=1e-6):
        """
        Custom loss function that encourages at least one prediction within the ground truth
        (or near its borders via dilation) to have a high probability.

        Parameters:
            tau (float): Temperature parameter for smooth maximum approximation.
            epsilon (float): Small value to prevent numerical issues.
        """
        super(AtLeastOneMatchLoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        """
        Compute the loss.

        Args:
            outputs (torch.Tensor): The raw model outputs (logits) with shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks. Expected shape is either (B, H, W)
                                    or (B, 1, H, W).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Apply sigmoid to convert logits to probabilities in [0, 1]
        p = torch.sigmoid(outputs).squeeze(1)  # shape: (B, H, W)

        # Ensure targets are float (0.0 or 1.0)
        targets = targets.float()

        # Adjust target dimensions: if targets are (B, H, W) unsqueeze to (B, 1, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif targets.dim() == 4 and targets.size(1) != 1:
            # If the channel dimension is not 1, select the first channel
            targets = targets[:, 0:1, ...]

        # Apply max pooling to dilate the ground truth mask (kernel size 3, stride 1, padding 1)
        dilated_targets = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1).squeeze(1)

        # Create a binary mask from the dilated ground truth (1 where region is considered)
        mask = (dilated_targets > 0.5).float()  # shape: (B, H, W)

        # Compute the sum of the mask per sample to detect if any positive exists
        mask_sum = torch.sum(mask, dim=(1, 2))  # shape: (B,)

        # Compute the smooth maximum of probabilities in the masked region for each sample:
        # smooth_max = (1/tau) * log(sum(exp(tau * p) * mask) + epsilon)
        smooth_max = (1.0 / self.tau) * torch.log(torch.sum(torch.exp(self.tau * p) * mask, dim=(1, 2)) + self.epsilon)

        # For samples with no valid ground truth region, set smooth_max to zero
        smooth_max = torch.where(mask_sum > 0, smooth_max, torch.zeros_like(smooth_max))

        # Loss per sample: encourage smooth_max to be close to 1 (i.e., -log(smooth_max))
        loss = -torch.log(smooth_max + self.epsilon)

        # Average the loss only over samples that contain valid ground truth region
        valid_loss = loss[mask_sum > 0]
        if valid_loss.numel() > 0:
            return valid_loss.mean()
        else:
            return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, tau=10, epsilon=1e-6, alpha=0.5):
        """
        Combined loss that mixes AtLeastOneMatchLoss and standard BCE loss.

        Parameters:
            tau (float): Temperature parameter for smooth maximum approximation.
            epsilon (float): Small value to prevent numerical issues.
            alpha (float): Weighting factor for the AtLeastOneMatchLoss.
                           (1 - alpha) is used for the BCE loss.
        """
        super(CombinedLoss, self).__init__()
        self.at_least_loss = AtLeastOneMatchLoss(tau=tau, epsilon=epsilon)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        """
        Compute the combined loss.

        Args:
            outputs (torch.Tensor): Raw logits from the model (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks (B, H, W) or (B, 1, H, W).

        Returns:
            torch.Tensor: The combined loss value.
        """
        loss1 = self.at_least_loss(outputs, targets)
        loss2 = self.bce_loss(outputs, targets)
        return self.alpha * loss1 + (1 - self.alpha) * loss2


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Implementación de Focal Loss para clasificación binaria.

        Parámetros:
          - alpha: Factor de balance entre clases.
          - gamma: Factor que reduce la contribución de ejemplos fáciles.
          - reduction: Especifica la reducción a aplicar ('mean', 'sum' o 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calcular la BCE sin reducción
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Calcular pt, la probabilidad de la clase correcta
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


#criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
criterion = CombinedLoss(tau=10, epsilon=1e-6, alpha=0.5)
#criterion = AtLeastOneMatchLoss(tau=10, epsilon=1e-6) Not converging
optimizer = optim.Adam(model.parameters(), lr=1e-4)


####################################################
# 7. Train and test functions
####################################################
def evaluate_model(model, dataloader, criterion, device):
    """
    Evalúa el modelo en un dataset, calculando la pérdida y la precisión pixel a pixel.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)  # Añadir dimensión de canal
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            # Aplicar sigmoid y threshold para obtener predicciones binarias
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            correct += (preds == labels).sum().item()
            total += torch.numel(labels)
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    """
    Función de entrenamiento que almacena la pérdida y precisión para cada época.
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

        # Evaluar en entrenamiento y prueba
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
# 8. Train
####################################################
num_epochs = 20
train_losses, train_accuracies, test_losses, test_accuracies = train_model(
    model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs
)


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


model_save_path = 'model_fft_Air_11_own_loss.pth'
torch.save(model.state_dict(), model_save_path)
print("Model saved to", model_save_path)
