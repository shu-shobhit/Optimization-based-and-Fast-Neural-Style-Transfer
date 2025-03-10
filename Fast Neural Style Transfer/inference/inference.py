import torch
from models.image_transform_net import ImageTransformationNetwork

class Stylizer:
    def __init__(self, model_path, device="cuda"):
        """Loads a trained model for inference."""
        self.device = device
        self.model = ImageTransformationNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        self.model.eval()

    def stylize(self, input_tensor):
        """Applies the style transfer to an input image tensor."""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return output_tensor
