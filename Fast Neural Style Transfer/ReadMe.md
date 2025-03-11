# Summarizing Fast NST

---

Neural Style Transfer (NST) is an image transformation task where a regular image (called the content image) is blended with the style of an artistic image (called the style image).

One method for solving image transformation tasks is to train a feed forward network in a supervised manner and using per-pixel losses for difference between the output and ground-truth images. The advantage of this method is the speed during test time. During inference, the input image only requires a forward pass through the trained network, leading to lower inference time.

The disadvantage of this method is that:

*"the per-pixel losses used by these methods do not capture perceptual differences between output and ground-truth images. For example, consider two identical images offset from each other by one pixel; despite their perceptual similarity they would be very different as measured by per-pixel losses."*
*~Justin Johnson et. al.*

Another method to solve this problem is to use "perceptual" losses as have been done in the previous optimization based NST (Gatys et. al.). In this method, our loss function is not defined using per-pixel values but instead a higher level feature representations extracted from pre-trained convolutional neural networks.

*"We train feedforward transformation networks for image transformation tasks, but rather than using per-pixel loss functions depending only on low-level pixel information, we train our networks using perceptual loss functions that depend on high-level features from a pretrained loss network. During training, perceptual losses measure image similarities more robustly than per-pixel losses, and at test-time the transformation networks run in real-time."*
*~Justin Johnson et. al.*

# Methodology

---

The system for neural style transfer transformation task consists of two main components:

1. **Image Transformation Network** $f_W$: A deep residual convolutional neural network that transforms an input image $x$ into an output image $\hat{y}$.
2. **Loss Network** $\phi$: A pre-trained convolutional neural network used to define multiple loss functions $ℓ_1,…,ℓ_k$ for training.

### Image Transformation Network

- The transformation network is parameterized by weights $W$ and maps input images to stylized output images as: $\hat{y} = f_W (x)$
- It is trained using **stochastic gradient descent (SGD)** to minimize a weighted sum of loss functions:
```math
$$
W^* = \arg\min_W \mathbb{E}_{x, \{y_i\}} \left[ \sum_{i=1}^{k} \lambda_i \ell_i(f_W(x), y_i) \right]
$$
```
- The goal is to ensure the transformed image $\hat{y}$ captures the content of the input image $x$ while incorporating the style of a target image $y_s$

### Loss Functions and Perceptual Loss

---

**Loss Network:**

- A pre-trained 16-layer VGG network ($\phi$) on the ImageNet dataset is utilized as the loss network.
- $\phi_j(x)$ represents the activations of the $j$-th layer of $\phi$ when processing input image $x$.
- For convolutional layers, $\phi_j(x)$ is a feature map of shape $C_j \times H_j \times W_j$, where $C_j$ is the number of channels, $H_j$ is the height, and $W_j$ is the width.

**1. Feature Reconstruction Loss:**

- This loss encourages the output image $\hat{y} = f_W(x)$ to have similar feature representations as the target image $y$.
- It is defined as the squared, normalized Euclidean distance between feature representations:
```math
$$
\phi_{,j}^{feat}(\hat{y}, y) = \frac{1}{C_j H_j W_j} ||\phi_j(\hat{y}) - \phi_j(y)||_2^2
$$
```
- Minimizing this loss for early layers of $\phi$ results in images visually indistinguishable from $y$.
- Higher layers preserve content and spatial structure, but not color, texture, or exact shape.

**2. Style Reconstruction Loss:**

- This loss penalizes differences in style (colors, textures, patterns) between the output $\hat{y}$ and target $y$.
- It utilizes the Gram matrix $G^{\phi}_j(x)$, which captures information about feature co-occurrence.
- The Gram matrix is defined as:
```math
$$
G^{\phi}_j(x)_{c, c'} = \frac{1}{C_j H_j W_j} \sum_{h=1}^{H_j} \sum_{w=1}^{W_j} \phi_j(x)_{h, w, c} \phi_j(x)_{h, w, c'}
$$
```
- Where $G^{\phi}_j(x)$ is a $C_j \times C_j$ matrix.
- Equivalently, if $\psi$ is the reshaped version of $\phi_j(x)$ of shape $C_j \times H_j W_j$, then:
```math
$$
G^{\phi}_j(x) = \frac{\psi \psi^T}{C_j H_j W_j}
$$
```
- The style reconstruction loss is the squared Frobenius norm of the difference between Gram matrices:
```math
$$
\phi_{,j}^{style}(\hat{y}, y) = ||G^{\phi}_j(\hat{y}) - G^{\phi}_j(y)||_F^2
$$
```
- This loss is well-defined for different image sizes.
- Minimizing this loss preserves stylistic features but not spatial structure.
- Reconstructing from higher layers transfers larger-scale structures.

**Style Reconstruction from Multiple Layers:**

- For style reconstruction from a set of layers $J$, the total style loss is the sum of individual layer losses:
```math
$$
\phi_{,J}^{style}(\hat{y}, y) = \sum_{j \in J} \phi_{,j}^{style}(\hat{y}, y)
$$
```
### Training Process

- The **content target** $y_c$ is the input image **$x$**.
- The network is trained to produce an output image $\hat{y}$ that preserves content from $x$ while adopting the style of a given style image $y_s$.
- One network is trained per style target.
  Training using this method produces high-quality stylized images in real time.
