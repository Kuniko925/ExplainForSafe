import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
    def forward(self, x):
        x = self.base_model(x)
        return x


class ViT16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.base_model.heads[0].in_features
        self.base_model.heads = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
    def forward(self, x):
        x = self.base_model(x)
        return x