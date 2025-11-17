import torch
import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.initializers import init_chainer_default


class SkillQNetwork(nn.Module):
    """Q-network: CNN → Q-values for each skill."""

    def __init__(self, num_skills=5, input_channels=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_skills),
            DiscreteActionValueHead(),
        )
        self.apply(init_chainer_default)

    def forward(self, x):
        features = self.backbone(x)
        return self.q_head(features)
