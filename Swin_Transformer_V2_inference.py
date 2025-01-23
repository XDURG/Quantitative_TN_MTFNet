import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import numpy as np
import os
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import warp_polar
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ThyroidDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image)
        return image

class SharedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        shared_output = self.fc(x)
        return shared_output * torch.sigmoid(self.threshold)

class SimpleSwinTransformer(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleSwinTransformer, self).__init__()
        self.swin = timm.create_model("swin_v2_tiny_window16_256", pretrained=True, num_classes=num_classes)
        self.shared_layer = SharedLayer(input_dim=1000, output_dim=128)

    def forward(self, x):
        features = self.swin(x)
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
        self.shared_layer = SharedLayer(input_dim=128, output_dim=128)

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
        self.shared_layer = SharedLayer(input_dim=output_dim, output_dim=128)

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
        self.composition_net = SimpleSwinTransformer()
        self.echogenic_foci_net = SimpleSwinTransformer()
        self.echogenicity_net = SiameseNetwork()
        self.margin_net = SimpleSwinTransformer()
        self.shape_net = SimpleMLP(input_dim=1, hidden_dim=64, output_dim=128)
        self.weights = nn.Parameter(torch.ones(5))
        self.transformer_head = TransformerHead(input_dim=128 * 5, num_heads=8, num_layers=3, num_classes=2)

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
        output = self.transformer_head(combined_features.unsqueeze(0))
        return output

transform = Compose([
    ToTensor(),
    Resize((256, 256)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = os.listdir('./test_imgs')
dataset = ThyroidDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = MultiFeatureNetwork().to(device)
model.load_state_dict(torch.load("./model_weights/st_70.pth"))
model.eval()

with torch.no_grad():
    for images in dataloader:
        images = images.to(device)
        outputs = model(images, images, (images, images), images, torch.tensor([1.0]).to(device))
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted: {predicted.item()}")