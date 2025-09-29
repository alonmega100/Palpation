# datasets.py
import json, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

def _load_label6(json_path: Path) -> torch.Tensor:
    """[x,y,z,yaw_deg,pitch_deg,roll_deg]"""
    with open(json_path, "r") as f:
        obj = json.load(f)
    x = float(obj["pos_m"]["x"]);  y = float(obj["pos_m"]["y"]);  z = float(obj["pos_m"]["z"])
    yaw = float(obj["euler_zyx_deg"]["yaw"])
    pit = float(obj["euler_zyx_deg"]["pitch"])
    rol = float(obj["euler_zyx_deg"]["roll"])
    return torch.tensor([x, y, z, yaw, pit, rol], dtype=torch.float32)

def _angles_deg_to_sincos(angles_deg: torch.Tensor) -> torch.Tensor:
    """angles_deg: tensor([yaw,pitch,roll]) in degrees -> [sy,cy, sp,cp, sr,cr]."""
    rad = angles_deg * (math.pi / 180.0)
    s = torch.sin(rad); c = torch.cos(rad)
    return torch.stack([s[0], c[0], s[1], c[1], s[2], c[2]])

class AprilTagCropDataset(Dataset):
    """
    Expects per sample:
      <stem>_crop.png, <stem>_crop_meta.json, <stem>.json

    Returns:
      img : transformed crop (3xHxW)
      aux : [cx/W, cy/H, cw/W, ch/H]  (float32)
      y9  : [x,y,z, sin(yaw),cos(yaw), sin(pitch),cos(pitch), sin(roll),cos(roll)]
    """
    root: Path
    items: List[Tuple[Path, Optional[Path], Path]]

    def __init__(self, root_dir: str, split: str = "train", transform=None, require_crop: bool = True):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.require_crop = require_crop

        pattern = "*_crop.png" if require_crop else "*_masked.png"
        imgs = sorted(self.root.glob(pattern))
        if not imgs:
            raise FileNotFoundError(f"No {pattern} in {self.root}")

        items: List[Tuple[Path, Optional[Path], Path]] = []
        for img in imgs:
            if img.name.endswith("_crop.png"):
                base = img.name[:-9]  # strip "_crop.png"
                meta = img.with_name(f"{base}_crop_meta.json")
            else:
                base = img.name.replace("_masked.png", "")
                meta = None if not require_crop else img.with_name(f"{base}_crop_meta.json")

            label = img.with_name(f"{base}.json")
            if not label.exists():
                print(f"Warning: missing label JSON for {img.name}")
                continue
            if require_crop and (meta is None or not meta.exists()):
                print(f"Warning: missing meta for {img.name}")
                continue
            items.append((img, meta, label))

        if not items:
            raise RuntimeError("No valid (image, meta, label) triplets found.")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _load_aux(self, meta_path: Optional[Path]) -> torch.Tensor:
        if meta_path is None:
            return torch.zeros(4, dtype=torch.float32)
        with open(meta_path, "r") as f:
            m = json.load(f)
        W, H = m["orig_size"]["w"], m["orig_size"]["h"]
        bb = m["crop_bbox_xyxy"]; x0, y0, x1, y1 = bb["x0"], bb["y0"], bb["x1"], bb["y1"]
        cw, ch = (x1 - x0), (y1 - y0)
        cx, cy = (x0 + cw/2.0), (y0 + ch/2.0)
        return torch.tensor([cx/W, cy/H, cw/W, ch/H], dtype=torch.float32)

    def __getitem__(self, idx: int):
        img_path, meta_path, label_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y6 = _load_label6(label_path)
        y9 = torch.cat([y6[:3], _angles_deg_to_sincos(y6[3:])], dim=0)  # 9D target

        aux = self._load_aux(meta_path)
        return img, aux, y9
