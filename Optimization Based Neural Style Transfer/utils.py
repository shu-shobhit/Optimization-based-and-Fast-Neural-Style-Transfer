import torch
from torchvision.transforms import v2
from PIL import Image

def load_and_transform(image_path, resize_value, device):
    image = Image.open(image_path)
    transformations = v2.Compose([
        v2.Resize(resize_value),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformations(image).unsqueeze(0).to(device)

def generate_white_noise(image, device):
    batch, channels, h, w = image.shape
    return torch.randn(size=[batch, channels, h, w], requires_grad=True, device=device)

def tensor_to_image(tensor):
    tensor = tensor.clone().detach().squeeze(0).cpu()
    tensor = tensor.mul(torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    tensor = tensor.add(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    tensor = torch.clamp(tensor, 0, 1)
    return v2.ToPILImage()(tensor)

def get_gram_matrix(input):
    batch, channel, height, width = input.shape
    reshaped_input = input.view(batch, channel, height * width)
    gram_matrix = torch.bmm(reshaped_input, reshaped_input.transpose(1, 2))
    return gram_matrix / (height * width)
