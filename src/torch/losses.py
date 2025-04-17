# Author: Trevor Settembre
# Project Title: SpaceShip Titanic
# Description: this file creates a pytorh compatible loss fuction 

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # Ensure targets are of type long
        targets = targets.long()
        
        # Cross entropy loss calculation
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Calculate pt and the focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()
