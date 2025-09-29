import torch
import torch.nn as nn
from torchvision import models

class CropPoseRegressor(nn.Module):
    """
    x:  (B,3,H,W)   crop image (normalized)
    aux:(B,4)       [cx/W, cy/H, cw/W, ch/H]
    out: (B,9)      [x,y,z, sy,cy, sp,cp, sr,cr]
    """
    def __init__(self, d_out=9, aux_dim=4, pretrained=True, normalize_angle_pairs=True):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.aux_dim = aux_dim
        self.normalize_angle_pairs = normalize_angle_pairs

        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + aux_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.pos_head = nn.Linear(256, 3)   # x,y,z
        self.ang_head = nn.Linear(256, 6)   # sy,cy, sp,cp, sr,cr

    def forward(self, x, aux=None):
        feats = self.backbone(x)  # (B,512)
        if aux is None:
            aux = torch.zeros(x.size(0), self.aux_dim, device=x.device, dtype=feats.dtype)
        h = self.fuse(torch.cat([feats, aux], dim=1))
        pos = self.pos_head(h)
        ang = self.ang_head(h)
        if self.normalize_angle_pairs:
            ang = ang.view(-1, 3, 2)
            ang = ang / (ang.norm(dim=2, keepdim=True) + 1e-8)  # unit circle per pair
            ang = ang.view(-1, 6)
        return torch.cat([pos, ang], dim=1)


