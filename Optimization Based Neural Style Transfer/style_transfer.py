import torch
import torch.optim as optim
from tqdm import tqdm
from model import VGG19
from utils import load_and_transform, generate_white_noise, tensor_to_image, get_gram_matrix
from config import(
    DEVICE, 
    RESIZE_VALUE, 
    NUM_STEPS, 
    LEARNING_RATE, 
    ALPHA, 
    BETA, 
    OPTIMIZER,
    CONTENT_LAYER_NAME,
    STYLE_LAYER_WEIGHTS,
    STYLE_LAYERS_LIST
)

class NeuralStyleTransfer:
    def __init__(self, content_image_path, style_image_path):
        self.device = DEVICE
        self.lr = LEARNING_RATE
        self.alpha = ALPHA
        self.beta = BETA
        self.optimizer_choice = OPTIMIZER
        self.num_steps = NUM_STEPS
        self.content_layer = CONTENT_LAYER_NAME
        self.style_layers_list = STYLE_LAYERS_LIST
        self.style_weights = STYLE_LAYER_WEIGHTS

        self.content_image = load_and_transform(content_image_path, RESIZE_VALUE, DEVICE)
        self.style_image = load_and_transform(style_image_path, RESIZE_VALUE, DEVICE)

        self.model = VGG19(pool_type="avg").to(DEVICE).eval()

    def optimize(self, target_image, loss_function):
        progress_bar = tqdm(range(self.num_steps), desc="Optimizing", unit="step")
        if self.optimizer_choice == "LBFGS":
            optimizer = torch.optim.LBFGS([target_image], lr=self.lr)

            def closure():
                optimizer.zero_grad()
                loss = loss_function()
                loss.backward()
                return loss

            for _ in progress_bar:
                optimizer.step(closure)

        elif self.optimizer_choice == "Adam":
            optimizer = optim.Adam([target_image], lr=self.lr)
            for _ in progress_bar:
                optimizer.zero_grad()
                loss = loss_function()
                loss.backward()
                optimizer.step()

        return tensor_to_image(target_image)

    def style_transfer(self):
        random_image = generate_white_noise(self.content_image, self.device)
        
        content_features = self.model(self.content_image, [self.content_layer])[0]
        
        style_features = self.model(self.style_image, self.style_layers_list)
        style_gram_matrices = [get_gram_matrix(f) for f in style_features]

        def total_loss_function():
            generated_features = self.model(random_image, [self.content_layer] + self.style_layers_list)
            
            content_loss = torch.mean((content_features - generated_features[0]) ** 2)
            
            style_loss = sum(
                self.style_weights[i] * torch.mean((get_gram_matrix(g) - style_gram_matrices[i]) ** 2)
                for i, g in enumerate(generated_features[1:])
            )
            
            return self.alpha * content_loss + self.beta * style_loss

        return self.optimize(random_image, total_loss_function)
