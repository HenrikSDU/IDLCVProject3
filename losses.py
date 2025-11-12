import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + F.softplus(-y_pred))
        return loss

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)  # Ensure predictions are in [0,1]
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        return torch.mean(focal_weight * bce)


class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Numerically stable BCE with logits
        loss = torch.mean(y_pred - y_true * y_pred + F.softplus(-y_pred))

        # Total Variation regularization (spatial smoothness)
        tv_h = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])  # vertical diffs
        tv_w = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])  # horizontal diffs
        regularization = torch.mean(tv_h) + torch.mean(tv_w)

        return loss + 0.1 * regularization
