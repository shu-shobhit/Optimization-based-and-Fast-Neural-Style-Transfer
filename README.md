# Neural-Style-Transfer-and-Fast-Neural-Style-Transfer

## Summarizing the traditional NST

Neural Style Transfer (NST) is an image transformation task where a regular image (called the content image) is blended with the style of an artistic image (called the style image).

In NST, **"style"** refers to the artistic and textural characteristics of an image that define its visual appearance, distinct from its content. **Texture** includes repetitive visual patterns, such as brushstrokes, geometric shapes, or surface details.

Deep Convolutional Neural Networks have been trained on large datasets for image classification tasks. During this training, different layers of the network learn varying levels and depths of feature representations of images.

The lower layers of the network capture fundamental features like edges, boundaries, and lines, representing basic visual elements at the pixel level. In contrast, the higher layers capture representations of the "actual content" while losing detailed pixel-level information. These representations are known as **"Content Representation."**

For constructing style representations, we do the following:
for a particular layer conv layer, we get feature correlations from the feature maps in that layer. By including these feature correlation matrices from different layers of the network, we can construct overall **"Style representation"** of the image which captures the overall texture information instead of any global arrangement in the image.

"*The key finding of this paper is that the representations of content and style in the Convolutional Neural Network are separable. That is, we can manipulate both representations independently to produce new, perceptually meaningful images.*"
~ Gatys et. al.
## Visualizing Content and Style Representations from Different Layers


To visualize the information captured by a particular layer, say $L$, we can start with a random noise image, $X$, and iteratively update its pixel values so that its feature representation at layer $L$ closely matches (ideally, becomes equal to) the feature representation of a reference image. This reference image could be used to capture either content or style, depending on the objective.

### Visualizing Content Representation:
To do this, we forward pass the image $\overrightarrow{X}$ till layer $L$ and get its content representation at layer $L$, $F_x^L$ . Similarly, we get the content representation of the content image $C$, $F_c^L$. We define the mean square error loss function:
$$
Loss_{content} = \frac12 \sum_{i,j} ((F_x^L)_{ij}^2 - (F_c^L)_{ij}^2)
$$
We get the gradient with respect to $\overrightarrow{X}$  by backpropagation and we iteratively change $\overrightarrow{X}$ through standard gradient descent.

### Visualizing Style Representation  
To visualize the style of an image, we extract its style representation by computing the **Gram matrix** at a given layer $L$. The Gram matrix $G_x^L$ for an image $\overrightarrow{X}$ is defined as:  

$$
G_x^L = (F_x^L)(F_x^L)^T
$$

Similarly, we compute the Gram matrix for the style image $S$:  

$$
G_s^L = (F_s^L)(F_s^L)^T
$$

We define the **style loss function** as the mean square error between the Gram matrices:  

$$
Loss_{style} = \frac{1}{4N^2M^2} \sum_{i,j} \left( (G_x^L)_{ij} - (G_s^L)_{ij} \right)^2
$$

where $N$ is the number of feature maps at layer $L$, and $M$ is the spatial dimension of the feature maps.

We compute the gradient of this loss with respect to $\overrightarrow{X}$ using **backpropagation** and iteratively update $\overrightarrow{X}$ through **standard gradient descent** to match the style of the reference style image.


### CODE
```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt
```

##### Modules in vgg19
- **0**: conv1_1 - Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **1**: ReLU(inplace=True)
- **2**: conv1_2 - Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **3**: ReLU(inplace=True)
- **4**:pool1-MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- **5**: conv2_1 - Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **6**: ReLU(inplace=True)
- **7**: conv2_2 - Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **8**: ReLU(inplace=True)
- **9**: pool2 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- **10**: conv3_1 - Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **11**: ReLU(inplace=True)
- **12**: conv3_2 - Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **13**: ReLU(inplace=True)
- **14**: conv3_3 - Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **15**: ReLU(inplace=True)
- **16**: conv3_4 - Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **17**: ReLU(inplace=True)
- **18**: pool3 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- **19**: conv4_1 - Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **20**: ReLU(inplace=True)
- **21**: conv4_2 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **22**: ReLU(inplace=True)
- **23**: conv4_3 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **24**: ReLU(inplace=True)
- **25**: conv4_4 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **26**: ReLU(inplace=True)
- **27**: pool4 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- **28**: conv5_1 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **29**: ReLU(inplace=True)
- **30**: conv5_2 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **31**: ReLU(inplace=True)
- **32**: conv5_3 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **33**: ReLU(inplace=True)
- **34**: conv5_4 - Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- **35**: ReLU(inplace=True)
- **36**: pool5 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

**"*We used the feature space provided by the 16 convolutional and 5 pooling layers of the 19 layer VGG Network. For image synthesis we found that replacing the max-pooling operation by average pooling improves the gradient flow and one obtains slightly more appealing results, which is why the images shown were generated with average pooling.*" ~ Gatys et. al.**

```python
class VGG19(nn.Module):
    def __init__(self, pool_type: str):
        super().__init__()
        if pool_type == "avg":
            self.layer_list = []
            self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
            for layer in self.vgg19:
                if isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                    self.layer_list.append(
                        nn.AvgPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    )
                else:
                    self.layer_list.append(layer)
            self.final_model = nn.Sequential(*self.layer_list)
        else:
            self.final_model = models.vgg19().features

        for param in self.final_model.parameters():
            param.requires_grad = False

    def forward(self, input, layer_keys):
        out = {}
        last_layer_key = str(max([int(key) for key in layer_keys]))
        for name, layer in self.final_model.named_children():
            out[name] = layer(input)
            input = out[name]
            if name == last_layer_key:
                return [out[key] for key in layer_keys]

        return [out[key] for key in layer_keys]

```
**Defining the Neural Style transfer Class:**
```python
class NeuralStyleTransfer:

    def __init__(
        self,
        content_image_path,
        style_image_path,
        content_layer_name,
        style_layers_list,
        style_layer_weights,
        optimizer="LBFGS",
        alpha=1,
        beta=1000,
        lr=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_steps=500,
        resize_value=256,
    ):
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self._optim = optimizer
        self.content_layer = content_layer_name
        self.style_layers_list = style_layers_list
        self.style_weights = style_layer_weights
        self.num_steps = num_steps

        self.content_image = self.load_and_transform(
            image_path=content_image_path, resize_value=resize_value
        )

        self.style_image = self.load_and_transform(
            image_path=style_image_path, resize_value=resize_value
        )

        self.model = VGG19(pool_type="avg").to(device).eval()
```

**Preprocessing Function for VGG19**

```python
def load_and_transform(self, image_path, resize_value):
    image = Image.open(image_path)
    transformations = v2.Compose(
        [
            v2.Resize(resize_value),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transformations(image).unsqueeze(0).to(self.device)
```

**Static Method for generating white noise image:**

```python
@staticmethod
def generate_white_noise(image, device):
    batch, channels, h, w = image.shape
    random_image = torch.randn(
        size=[batch, channels, h, w], requires_grad=True, device=device
    )
    return random_image
```

**Static method for converting a output tensor back to PIL image:**

```python
@staticmethod
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().squeeze(0).cpu()

    tensor = tensor.mul(torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    tensor = tensor.add(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    tensor = torch.clamp(tensor, 0, 1)

    return v2.ToPILImage()(tensor)
```

**Static method for getting gram matrix** 
```python
@staticmethod
def get_gram_matrix(input):
    batch, channel, height, width = input.shape
    reshaped_input = input.view(batch, channel, height * width)
    gram_matrix = torch.bmm(reshaped_input, reshaped_input.transpose(1, 2))
    gram_matrix = torch.div(gram_matrix, height * width)
    return gram_matrix
```
  
**Function to visualize the content representation of a image at a particular layer.**
The function optimizes a random noise image using either **L-BFGS** or **Adam** optimization.

```python
def visualize_content_rep(self):
    print(f"Using Device: {self.device}")
    random_image = self.generate_white_noise(self.content_image, self.device)

    original_img_features = self.model(self.content_image, self.content_layer)[0].detach()

    progress_bar = tqdm(range(self.num_steps), desc="Optimizing", unit="step")
    if self._optim == "LBFGS":
        optimizer = optim.LBFGS([random_image], lr=self.lr)
        for step in progress_bar:
            current_loss = [0.0]

            def closure():
                optimizer.zero_grad()
                random_img_features = self.model(random_image, self.content_layer)[0]
                content_loss = torch.mean(
                    (original_img_features - random_img_features) ** 2
                )

                content_loss.backward()
                current_loss[0] = content_loss.item()
                return content_loss

            optimizer.step(closure)
            progress_bar.set_postfix(loss=current_loss[0])

    elif self._optim == "Adam":
        optimizer = optim.Adam([random_image], lr=self.lr)
        for step in progress_bar:
            optimizer.zero_grad()
            random_img_features = self.model(random_image, self.content_layer)[0]
            content_loss = torch.mean(
                (original_img_features - random_img_features) ** 2
            )
            content_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=content_loss.item())
    generated_image = self.tensor_to_image(random_image)
    plt.imshow(generated_image)
    plt.axis("off")
    plt.show()
    return generated_image
```

```python
def visualize_style_rep(self):
    print(f"Using Device: {self.device}")
    random_image = self.generate_white_noise(self.style_image, self.device)

    original_style_features = self.model(self.style_image, self.style_layers_list)
    original_gram_matrices = []
    for f in original_style_features:
        gram_matrix = self.get_gram_matrix(f)
        original_gram_matrices.append(gram_matrix)

    progress_bar = tqdm(range(self.num_steps), desc="Optimizing", unit="step")
    if self._optim == "LBFGS":
        optimizer = optim.LBFGS([random_image], lr=self.lr)
        for step in progress_bar:
            current_loss = [0.0]

            def closure():
                optimizer.zero_grad()
                style_loss = torch.zeros([], device=self.device, requires_grad=True)

                style_outputs = self.model(random_image, self.style_layers_list)
                for idx, o in enumerate(style_outputs):
                    G = self.get_gram_matrix(o)
                    style_loss = torch.add(
                        style_loss,
                        torch.mean((G - original_gram_matrices[idx]) ** 2)
                        * self.style_weights[idx],
                    )

                style_loss.backward()

                current_loss[0] = style_loss.item()
                return style_loss

            optimizer.step(closure)
            progress_bar.set_postfix(loss=current_loss[0])

    elif self._optim == "Adam":
        optimizer = optim.Adam([random_image], lr=self.lr)
        for step in progress_bar:
            optimizer.zero_grad()
            style_loss = torch.zeros([], device=self.device, requires_grad=True)
            style_outputs = self.model(random_image, self.style_layers_list)

            for idx, o in enumerate(style_outputs):
                G = self.get_gram_matrix(o)
                style_loss = torch.add(
                    self.style_weights[idx]
                    * torch.mean((G - original_gram_matrices[idx]) ** 2),
                    style_loss,
                )

            style_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=style_loss.item())
    generated_image = self.tensor_to_image(random_image)
    plt.imshow(generated_image)
    plt.axis("off")
    plt.show()
    return generated_image
```


