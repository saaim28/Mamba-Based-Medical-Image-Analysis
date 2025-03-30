import sys
sys.path.append("./mamba")

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class VisionMambaClassifier(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, num_classes=2):
        super(VisionMambaClassifier, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
        )

        # Flatten and reshape for Mamba (B, C, H, W) -> (B, L, D)
        self.mamba_input_proj = nn.Linear(64, hidden_dim)
        self.mamba = Mamba(hidden_dim)
        self.mamba_output_proj = nn.Linear(hidden_dim, 128)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),  # added dropout
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)              # (B, 64, 32, 32)
        x = x.flatten(2)                # (B, 64, 256)
        x = x.transpose(1, 2)           # (B, 256, 64)
        x = self.mamba_input_proj(x)    # (B, 256, hidden_dim)
        x = self.mamba(x)               # (B, 256, hidden_dim)
        x = x.mean(dim=1)               # (B, hidden_dim)
        x = self.mamba_output_proj(x)   # (B, 128)
        out = self.classifier(x)        # (B, num_classes)
        return out