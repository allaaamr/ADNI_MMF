import torch
import torch.nn as nn

import torch
import torch.nn as nn
from models.ResNet3D import *
from models.ClinicalMLP import *

class LateFusion(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.resnet3d     = ResNet3D(num_classes=num_classes, in_channels=1, layers=(2,2,2,2), widths=(32,64,128,256))
        self.clinical_mlp = ClinicalMLP2(num_features=8, num_classes=num_classes)

        # Branches already output (B, num_classes) → no extra projection
        self.resnet_proj   = nn.Identity()
        self.clinical_proj = nn.Identity()

    def forward(self, data_mri, data_clinical):
        # Each branch returns logits (B, num_classes)
        x_resnet   = self.resnet3d(data_mri, data_clinical)
        x_clinical = self.clinical_mlp(data_mri, data_clinical)

        resnet_logits   = self.resnet_proj(x_resnet)      # passthrough
        clinical_logits = self.clinical_proj(x_clinical)  # passthrough

        return (resnet_logits + clinical_logits) / 2.0
