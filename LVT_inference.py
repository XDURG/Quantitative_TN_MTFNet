import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import numpy as np
import os
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import warp_polar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class lite_vision_transformer(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000, patch_size=4,
                 embed_dims=None, num_heads=None,
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=None,
                 mlp_ratios=None, mlp_depconv=None, sr_ratios=[1, 1, 1, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head

        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        self.backbone = nn.ModuleList(network)

        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.with_cls_head:
            for idx, stage in enumerate(self.backbone):
                x = stage(x)
            x = self.norm(x)
            x = self.head(x.mean(dim=(1, 2)))
            return x
        else:
            outs = []
            for idx, stage in enumerate(self.backbone):
                x = stage(x)
                x = self.downstream_norms[idx](x)
                outs.append(x.permute(0, 3, 1, 2).contiguous())
            return outs

class lvt(lite_vision_transformer):
    def __init__(self, rasa_cfg=None, with_cls_head=True, **kwargs):
        super().__init__(
            layers=[2, 2, 2, 2],
            patch_size=4,
            embed_dims=[64, 64, 160, 256],
            num_heads=[2, 2, 5, 8],
            mlp_ratios=[4, 8, 4, 4],
            mlp_depconv=[False, True, True, True],
            sr_ratios=[8, 4, 2, 1],
            sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
            rasa_cfg=rasa_cfg,
            with_cls_head=with_cls_head,
        )

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

class SimpleLVT(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleLVT, self).__init__()
        self.lvt = lvt(rasa_cfg=None, with_cls_head=True)
        self.shared_layer = SharedLayer(input_dim=1000, output_dim=128)

    def forward(self, x):
        features = self.lvt(x)
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
        self.composition_net = SimpleLVT()
        self.echogenic_foci_net = SimpleLVT()
        self.echogenicity_net = SiameseNetwork()
        self.margin_net = SimpleLVT()
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
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = os.listdir('./test_imgs')
dataset = ThyroidDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = MultiFeatureNetwork().to(device)
model.load_state_dict(torch.load("./model_weights/lvt_86.pth"))
model.eval()

with torch.no_grad():
    for images in dataloader:
        images = images.to(device)
        outputs = model(images, images, (images, images), images, torch.tensor([1.0]).to(device))
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted: {predicted.item()}")