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

1. **Image Transformation Network** $f_W$: A deep residual convolutional neural network that transforms an input image xx into an output image $\hat{y}$.
2. **Loss Network** $\phi$: A pre-trained convolutional neural network used to define multiple loss functions $ℓ_1,…,ℓ_k$ for training.

### Image Transformation Network

- The transformation network is parameterized by weights $W$ and maps input images to stylized output images as: $\hat{y} = f_W (x)$
- It is trained using **stochastic gradient descent (SGD)** to minimize a weighted sum of loss functions:

$$
W^* = \arg\min_W \mathbb{E}_{x, \{y_i\}} \left[ \sum_{i=1}^{k} \lambda_i \ell_i(f_W(x), y_i) \right]
$$

- The goal is to ensure the transformed image $\hat{y}$ captures the content of the input image $x$ while incorporating the style of a target image $y_s$

### Loss Functions and Perceptual Loss

---

We use perceptual losses inspired by methods that optimize images using deep networks (traditional NST):

- **Loss Network ($\phi$)**:

  - A pre-trained CNN for image classification (VGG16 here) is used as a fixed loss network.
  - It helps compute feature-based losses that measure differences in content and style.
- **Feature Reconstruction Loss (Content Loss) $\ell_{\phi}^{\text{feat}}$**:

  - Measures the difference in high-level feature representations between $\hat{y}$ and content target $y_c$.
- **Style Reconstruction Loss $\ell_{\phi}^{\text{style}}$**:

  - Captures the difference in texture and style by comparing Gram matrices of feature maps.

### Training Process

- The **content target** $y_c$ is the input image **$x$**.
- The network is trained to produce an output image $\hat{y}$ that preserves content from $x$ while adopting the style of a given style image $y_s$.
- One network is trained per style target.
  Training using this method produces high-quality stylized images in real time.
