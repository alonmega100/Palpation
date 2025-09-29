import torch
import torch.nn as nn
from torchvision import models

class CropPoseRegressor(nn.Module):
    """
    x:  (B,3,H,W) crop image (RGB, normalized)
    aux:(B,4)  = [cx/W, cy/H, cw/W, ch/H]  (float32)
    returns: (B, D_OUT)  e.g. D_OUT=6 or 9 (angles as sin/cos)
    """
    def __init__(self, d_out=9, aux_dim=4, pretrained=True):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()          # keep global avg pooled 512-d features
        self.backbone = base

        self.head = nn.Sequential(
            nn.Linear(feat_dim + aux_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, d_out),
        )
        self.aux_dim = aux_dim

    def forward(self, x, aux=None):
        feats = self.backbone(x)         # (B,512)
        if aux is None:
            aux = torch.zeros(x.size(0), self.aux_dim, device=x.device, dtype=feats.dtype)
        out = self.head(torch.cat([feats, aux], dim=1))
        return out
