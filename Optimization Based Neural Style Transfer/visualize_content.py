import torch
from utils import generate_white_noise


def visualize_content_rep(nst):

    random_image = generate_white_noise(nst.content_image, nst.device)
    content_features = nst.model(nst.content_image, [nst.content_layer])[0].detach()

    def content_loss_function():
        generated_features = nst.model(random_image, [nst.content_layer])[0]
        return torch.mean((content_features - generated_features) ** 2)

    return nst.optimize(random_image, content_loss_function)
