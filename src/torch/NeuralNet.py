# Author: Trevor Settembre
# Project Title: SpaceShip Titanic
# Description: Deep neural net with extended residual depth, wider layers, and optimized architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

# Label smoothing loss
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, target):
        n_classes = x.size(1)

        # Safer device-aware one-hot initialization
        one_hot = torch.zeros_like(x, device=x.device).scatter(1, target.unsqueeze(1), 1)

        # Apply label smoothing
        one_hot = one_hot * (1 - self.epsilon) + (self.epsilon / n_classes)

        # Log probabilities
        log_p = F.log_softmax(x, dim=1)

        # Smoothed cross-entropy
        loss = -(one_hot * log_p).sum(dim=1).mean()
        return loss

# Mish activation
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            Mish(),
            nn.BatchNorm1d(dim, momentum=0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            Mish(),
            nn.BatchNorm1d(dim, momentum=0.1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

# Output block with shortcut connection
class OutputBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            Mish(),
            nn.BatchNorm1d(in_dim, momentum=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_dim, out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x) + self.shortcut(x)

# Full neural net
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, device, task_type='binary'):
        super(NeuralNet, self).__init__()
        self.device = device
        self.task_type = task_type

        self.fc_in = nn.Sequential(
            nn.Linear(input_size, 2048),
            Mish(),
            nn.BatchNorm1d(2048, momentum=0.1),
            nn.Dropout(0.5),
        )

        self.res_block1 = ResidualBlock(2048, dropout=0.4)
        self.res_block2 = ResidualBlock(2048, dropout=0.4)
        self.res_block3 = ResidualBlock(2048, dropout=0.4)
        self.res_block4 = ResidualBlock(2048, dropout=0.4)
        self.res_block5 = ResidualBlock(2048, dropout=0.4)

        self.fc_out = nn.Sequential(
            nn.Linear(2048, 1024),
            Mish(),
            nn.BatchNorm1d(1024, momentum=0.1),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            Mish(),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            Mish(),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.Dropout(0.5),

            nn.Linear(256, output_size)
        )

        self.criterion = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.fc_out(x)
        return x

    def compute_loss(self, x, y):
        assert self.criterion is not None, "Loss function (criterion) must be set before calling compute_loss."
        output = self.forward(x)
        loss = self.criterion(output, y)
        return loss, output
