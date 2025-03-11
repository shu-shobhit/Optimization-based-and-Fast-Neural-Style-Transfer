import torch
from utils import generate_white_noise, get_gram_matrix

def visualize_style_rep(nst):
    random_image = generate_white_noise(nst.style_image, nst.device)
    style_features = nst.model(nst.style_image, nst.style_layers_list)
    style_gram_matrices = [get_gram_matrix(f) for f in style_features]

    def style_loss_function():
        generated_features = nst.model(random_image, nst.style_layers_list)
        loss = sum(
            nst.style_weights[i] * torch.mean((get_gram_matrix(g) - style_gram_matrices[i]) ** 2)
            for i, g in enumerate(generated_features)
        )
        return loss

    return nst.optimize(random_image, style_loss_function)
