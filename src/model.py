#%%writefile src/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrameAverageResNet(nn.Module):
    """
    ResNet-18 backbone with temporal average pooling.

    Input:
        x: [B, T, C, H, W]

    Output:
        logits: [B, num_classes]
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        b, t, c, h, w = x.shape

        # 把 batch 维和时间维合并，让每一帧单独进入 ResNet
        x = x.reshape(b * t, c, h, w)

        logits = self.backbone(x)      # [B*T, num_classes]

        # 恢复成视频维度
        logits = logits.reshape(b, t, -1)

        # 时间平均，得到视频级预测
        logits = logits.mean(dim=1)    # [B, num_classes]

        return logits