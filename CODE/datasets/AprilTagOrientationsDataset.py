import os, json, glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset


def _extract_target_vec(obj: Dict) -> torch.Tensor:
    """
    Return 6D tensor [x,y,z,yaw_deg,pitch_deg,roll_deg] as float32.
    Raises KeyError with a clear message if a field is missing.
    """
    try:
        x = float(obj["pos_m"]["x"])
        y = float(obj["pos_m"]["y"])
        z = float(obj["pos_m"]["z"])
        yaw   = float(obj["euler_zyx_deg"]["yaw"])
        pitch = float(obj["euler_zyx_deg"]["pitch"])
        roll  = float(obj["euler_zyx_deg"]["roll"])
    except KeyError as e:
        raise KeyError(f"JSON missing key: {e}. Full object keys: {list(obj.keys())}") from e
    return torch.tensor([x, y, z, yaw, pitch, roll], dtype=torch.float32)


class AprilTagOrientationsDataset(Dataset):
    """
    Minimal dataset:
      root/
        ... *_masked.png + matching .json (same stem)
    Returns: (image, target6) where image is RGB (apply transforms outside or add transform=).
    """
    def __init__(self, data_dir: str, transform=None):
        self.root = Path(data_dir)
        self.transform = transform

        imgs = sorted(self.root.glob("*_masked.png"))
        if not imgs:
            raise FileNotFoundError(f"No *_masked.png in {self.root}")

        items: List[Tuple[Path, Path]] = []
        for img in imgs:
            jpath = img.with_suffix(".json").as_posix().replace("_masked.json", ".json")
            jpath = Path(jpath)
            if not jpath.exists():
                jpath = img.parent / (img.name.replace("_masked.png", ".json"))
            if jpath.exists():
                items.append((img, jpath))
            else:
                print(f"Warning: missing JSON for {img.name}")

        if not items:
            raise RuntimeError("No image/json pairs found.")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load_target(json_path: Path) -> torch.Tensor:
        with open(json_path, "r") as f:
            obj = json.load(f)
        y = _extract_target_vec(obj)

        return y

    def __getitem__(self, idx: int):
        img_path, json_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = self._load_target(json_path)
        return img, y
