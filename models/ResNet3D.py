import os, math, random, glob
import numpy as np
import nibabel as nib
from typing import Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)


# --- Add these imports near the top of your previous file ---
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
from models.utils import *




class ResNet3D(nn.Module):
    """
    A light 3D-ResNet: layers = [2,2,2,2] with widths [32,64,128,256]
    Input: (B,1,D,H,W) -> output logits (B,num_classes)
    """
    def __init__(self, num_classes=3, in_channels=1, layers=(2,2,2,2), widths=(16,32,64,128)):
        super().__init__()
        self.in_planes = widths[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(widths[3], layers[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(widths[3]*BasicBlock3D.expansion, num_classes)
        )

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock3D(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * BasicBlock3D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, data_mri, data_clinical):
        # x: (B,1,D,H,W)
        x = self.stem(data_mri)
        x = self.layer1(x)  # (B,32, D/4, H/4, W/4) roughly
        x = self.layer2(x)  # (B,64, ...)
        x = self.layer3(x)  # (B,128, ...)
        x = self.layer4(x)  # (B,256, ...)
        x = self.head(x)    # (B,num_classes)
        return x
