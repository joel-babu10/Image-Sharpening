import os
import cv2
import torch
from torch.utils.data import Dataset

class ImagePairDataset(Dataset):
    def __init__(self, blurry_dir, sharp_dir, transform=None, image_size=256):
        self.blurry_paths = sorted([
            os.path.join(blurry_dir, f)
            for f in os.listdir(blurry_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.sharp_paths = sorted([
            os.path.join(sharp_dir, f)
            for f in os.listdir(sharp_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        assert len(self.blurry_paths) == len(self.sharp_paths), "Mismatch in number of images!"
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.blurry_paths)

    def __getitem__(self, idx):
        blurry = cv2.imread(self.blurry_paths[idx])
        sharp = cv2.imread(self.sharp_paths[idx])

        blurry = cv2.cvtColor(blurry, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        blurry = cv2.resize(blurry, (self.image_size, self.image_size))
        sharp = cv2.resize(sharp, (self.image_size, self.image_size))

        if self.transform:
            augmented = self.transform(image=blurry, mask=sharp)
            blurry = augmented['image']  # Already float tensor [3,H,W] in [0,1]
            sharp = augmented['mask']    # Already float tensor [3,H,W] in [0,1]
        else:
            blurry = torch.from_numpy(blurry).permute(2, 0, 1).float() / 255.0
            sharp = torch.from_numpy(sharp).permute(2, 0, 1).float() / 255.0

        return blurry, sharp
