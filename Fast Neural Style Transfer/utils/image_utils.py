import torch
from PIL import Image
import torchvision.transforms.v2 as transforms
from dataset.image_dataset import ImageDataset

def load_image(image_path, size=1080):
    image_tensor = ImageDataset.load_and_transform(image_path, resize_value=size)
    return image_tensor.unsqueeze(0)


def save_image(tensor, filename):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(filename)
    print(f"Saved output image to {filename}")
