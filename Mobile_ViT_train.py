import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import numpy as np
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import warp_polar
from mobilevit import MobileViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ThyroidDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return image, label

class SharedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        shared_output = self.fc(x)
        return shared_output * torch.sigmoid(self.threshold)

class SimpleMobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes):
        super(SimpleMobileViT, self).__init__()
        self.mobilevit = MobileViT(
            image_size=image_size,
            dims=dims,
            channels=channels,
            num_classes=num_classes
        )
        self.shared_layer = SharedLayer(input_dim=1000, output_dim=128)  # Shared layer

    def forward(self, x):
        features = self.mobilevit(x)
        shared_features = self.shared_layer(features)
        return shared_features

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.shared_layer = SharedLayer(input_dim=128, output_dim=128)  # Shared layer

    def forward(self, x1, x2):
        output1 = self.cnn(x1)
        output1 = output1.view(output1.size(0), -1)
        output1 = self.fc(output1)
        output1 = self.shared_layer(output1)

        output2 = self.cnn(x2)
        output2 = output2.view(output2.size(0), -1)
        output2 = self.fc(output2)
        output2 = self.shared_layer(output2)

        return output1, output2

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.shared_layer = SharedLayer(input_dim=output_dim, output_dim=128)  # Shared layer

    def forward(self, x):
        features = self.fc(x)
        shared_features = self.shared_layer(features)
        return shared_features

class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes):
        super(TransformerHead, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
            x = self.transformer_encoder(x)
            x = self.fc(x.mean(dim=1))
        return x

class MultiFeatureNetwork(nn.Module):
    def __init__(self):
        super(MultiFeatureNetwork, self).__init__()
        self.composition_net = SimpleMobileViT(image_size=(64, 64), dims=[96, 120, 144], channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], num_classes=1000)
        self.echogenic_foci_net = SimpleMobileViT(image_size=(64, 64), dims=[96, 120, 144], channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], num_classes=1000)
        self.echogenicity_net = SiameseNetwork()
        self.margin_net = SimpleMobileViT(image_size=(64, 64), dims=[96, 120, 144], channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], num_classes=1000)
        self.shape_net = SimpleMLP(input_dim=1, hidden_dim=64, output_dim=128)

        self.weights = nn.Parameter(torch.ones(5))  # One weight for each feature vector

        self.transformer_head = TransformerHead(input_dim=128 * 5, num_heads=8, num_layers=3, num_classes=2)  # Binary classification

    def forward(self, roi_composition, roi_echogenic_foci, roi_echogenicity, roi_margin, roi_shape):
        composition_vector = self.composition_net(roi_composition)
        echogenic_foci_vector = self.echogenic_foci_net(roi_echogenic_foci)
        echogenicity_edge_vector, echogenicity_inner_vector = self.echogenicity_net(*roi_echogenicity)
        margin_vector = self.margin_net(roi_margin)
        shape_vector = self.shape_net(roi_shape)

        echogenicity_vector = (echogenicity_edge_vector + echogenicity_inner_vector) / 2

        weighted_vectors = [
            composition_vector * self.weights[0],
            echogenic_foci_vector * self.weights[1],
            echogenicity_vector * self.weights[2],
            margin_vector * self.weights[3],
            shape_vector * self.weights[4],
        ]

        combined_features = torch.cat(weighted_vectors, dim=1)

        output = self.transformer_head(combined_features.unsqueeze(0))  # Add batch dimension
        return output

transform = Compose([
    ToTensor(),
    Resize((64, 64)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = os.listdir('./Thyroid_dataset')
labels = []
with open('./Thyroid_dataset/training.txt', 'r') as file:
    for line in file:
        columns = line.strip().split()
        if columns:
            labels.append(columns[-1])

dataset = ThyroidDataset(image_paths, labels, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = MultiFeatureNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, images, (images, images), images, torch.tensor([1.0]).to(device))  # Dummy shape input
        loss = criterion(outputs.squeeze(0), labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images, images, (images, images), images, torch.tensor([1.0]).to(device))
            loss = criterion(outputs.squeeze(0), labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, "
          f"Val Accuracy: {100 * correct / total:.2f}%")