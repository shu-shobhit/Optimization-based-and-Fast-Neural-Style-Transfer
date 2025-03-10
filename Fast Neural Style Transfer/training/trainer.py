import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from models.image_transform_net import ImageTransformationNetwork
from models.vgg16 import VGG16
from dataset.image_dataset import ImageDataset


class Trainer:
    def __init__(
        self,
        content_folder,
        style_path,
        lr=0.001,
        batch_size=4,
        num_epochs=1,
        alpha=1e5,
        beta=1e10,
        seed=42,
        resize_value=256,
        style_resize_value=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        content_layer=["8"],
        style_layers=[str(i) for i in [3, 8, 15, 22]],
    ):
        print("Initializing Trainer...")
        self.seed = seed
        self.lr = lr
        self.num_epochs = num_epochs
        self.beta = beta
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.content_layer = content_layer
        self.seed = seed
        self.style_layers = style_layers
        self.device = device
        print(f"Using device: {self.device}")

        self.loss_net = VGG16(pool_type="max").to(self.device).eval()
        print("loss_net (VGG16) initialized.")

        self.style_image = (
            ImageDataset.load_and_transform(style_path, style_resize_value)
            .unsqueeze(0)
            .to(self.device)
        )

        self.img_transform_net = ImageTransformationNetwork().to(device)
        print("img_transform_net initialized.")

        self.dataset, self.dataloader = self._get_dataloader(
            folder_path=content_folder,
            resize_value=resize_value,
            batch_size=self.batch_size,
        )
        print(
            f"Dataset and DataLoader created. Dataset length: {len(self.dataset)}"
        )
        self.optimizer = optim.Adam(self.img_transform_net.parameters(), lr=self.lr)
        print("Optimizer initialized.")

    @staticmethod
    def _get_dataloader(folder_path, resize_value, batch_size):
        print("Creating ImageDataset and DataLoader...")
        dataset = ImageDataset(folder_path=folder_path, resize_value=resize_value)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        print(f"DataLoader created with batch_size = {batch_size}")
        return dataset, dataloader

    def vgg_transform(self, input):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(-1, 1, 1)
        input = torch.div(input, 255.0)
        normalized = (input - mean) / std
        return normalized

    def get_gram_matrix(self, input):
        batch, channel, height, width = input.shape
        reshaped_input = input.view(batch, channel, height * width)
        gram_matrix = torch.bmm(reshaped_input, reshaped_input.transpose(1, 2)) / (
            channel * height * width
        )
        return gram_matrix

    def get_content_loss(self, output, target):
        loss = self.mse_loss(output, target)
        return loss

    def get_style_loss(self, outputs, target_gram_matrices):
        style_loss = 0
        for output, G_target in zip(outputs, target_gram_matrices):
            G = self.get_gram_matrix(output)
            target_gram_matrix = G_target.repeat(G.shape[0], 1, 1)
            style_loss = style_loss + self.mse_loss(G, target_gram_matrix)
        return style_loss

    def train(self, checkpoint_dir="checkpoints_"):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.img_transform_net.train()
        self.loss_net.eval()

        style_image = self.vgg_transform(self.style_image)
        style_targets = self.loss_net(style_image, self.style_layers)

        target_gram_matrices = [
            self.get_gram_matrix(style_targets[key]) for key in self.style_layers
        ]

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.8
        )
        print("Scheduler initialized.")

        global_iter = 0

        history = {"content_loss": [], "style_loss": [], "total_loss": []}

        for epoch in range(self.num_epochs):
            print("*" * 10 + f" Epoch - {epoch + 1} " + "*" * 10)

            progress_bar = tqdm(
                self.dataloader,
                desc="Optimizing",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            agg_content_loss = agg_style_loss = 0
            for idx, input_images in enumerate(progress_bar):
                X = input_images.to(self.device)
                self.optimizer.zero_grad()
                y = self.img_transform_net(X)

                y = self.vgg_transform(y)
                X = self.vgg_transform(X)

                y_rep = self.loss_net(y, self.content_layer + self.style_layers)

                y_content_rep = y_rep[self.content_layer[0]]
                content_target_rep = self.loss_net(X, self.content_layer).get(
                    self.content_layer[0]
                )

                content_loss = self.alpha * self.get_content_loss(
                    y_content_rep, content_target_rep
                )

                y_style_rep = [y_rep[key] for key in self.style_layers]
                style_loss = self.beta * self.get_style_loss(
                    y_style_rep, target_gram_matrices
                )

                total_loss = content_loss + style_loss
                total_loss.backward()

                self.optimizer.step()
                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                progress_bar.set_postfix(
                    {
                        "content_loss": agg_content_loss / (idx + 1),
                        "style_loss": agg_style_loss / (idx + 1),
                        "total_loss": (agg_content_loss + agg_style_loss) / (idx + 1),
                    },
                    refresh=True,
                )
                history["content_loss"].append(agg_content_loss / (idx + 1))
                history["style_loss"].append(agg_style_loss / (idx + 1))
                history["total_loss"].append(
                    (agg_content_loss + agg_style_loss) / (idx + 1)
                )

                global_iter += 1
                scheduler.step()
                if global_iter % 2000 == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"model_epoch_{epoch + 1}.pth"
                    )

                    self.img_transform_net.eval().cpu()

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.img_transform_net.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": history["total_loss"][-1],
                        },
                        checkpoint_path,
                    )
                    print(f"Checkpoint saved at {checkpoint_path}")
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Iteration {global_iter}: Adjusted learning rate to {current_lr:.6f}"
                    )

                    self.img_transform_net.to(self.device).train()

                if (idx + 1) % 500 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch {idx + 1}: Content Loss: {agg_content_loss / (idx + 1)}, Style Loss: {agg_style_loss / (idx + 1)}, Total Loss: {(agg_content_loss + agg_style_loss) / (idx + 1)}"
                    )

        plt.figure(figsize=(10, 5))
        plt.plot(history["content_loss"], label="Content Loss", alpha=0.7)
        plt.plot(history["style_loss"], label="Style Loss", alpha=0.7)
        plt.plot(history["total_loss"], label="Total Loss", alpha=0.7)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training Loss Curves (Log Scale)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--")
        plt.show()

        return self.img_transform_net, history

