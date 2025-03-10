import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, folder_path, resize_value=256):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.resize_value = resize_value

    @staticmethod
    def load_and_transform(image_path, resize_value):
        image = Image.open(image_path).convert("RGB")
        transformations = v2.Compose(
            [
                v2.Resize((resize_value, resize_value)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )
        return transformations(image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensor = self.load_and_transform(image_path, self.resize_value)
        return image_tensor
