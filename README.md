# Face Generation with a GAN (CelebA, PyTorch)

A simple Generative Adversarial Network (GAN) that learns to generate 64×64 human face images from random noise, trained on the CelebA dataset.

Two networks play a game: the **Generator** tries to create fake faces that look real, and the **Discriminator** tries to tell real faces from fake ones. Over time, both improve.

## Dataset

- **CelebA** (CelebFaces Attributes) from Kaggle: [`jessicali9530/celeba-dataset`](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- 202,599 aligned celebrity face images
- Downloaded in the notebook using `kagglehub`

### Preprocessing

| Step | Purpose |
|------|---------|
| `CenterCrop(178)` | 178×218 → 178×178 (square crop around the face) |
| `Resize(64)` | 178×178 → 64×64 |
| `ToTensor()` | Image → tensor with values in [0, 1] |
| `Normalize(0.5, 0.5)` | Rescale to [-1, 1] to match the Generator's `Tanh` output |

A custom `ImageProcessor` dataset class loads the images, and a `DataLoader` serves them in shuffled batches of 128.

## Model Architecture

Both networks are fully connected (MLP) networks.

**Generator**: noise vector (100) → 256 → 512 → 1024 → 12,288 (64×64×3), reshaped to an image.
- ReLU activations, `Tanh` at the output

**Discriminator**: image flattened (12,288) → 1024 → 512 → 256 → 1.
- LeakyReLU(0.2) activations, `Sigmoid` at the output (probability that the image is real)

## Training Setup

| Setting | Value |
|---------|-------|
| Loss | Binary Cross-Entropy (`BCELoss`) |
| Optimizer | Adam (lr = 0.0002, betas = (0.5, 0.999)) for both networks |
| Batch size | 128 |
| Latent dimension | 100 |
| Epochs | 5 (configurable) |
| Hardware | GPU (Colab T4) |

### Training loop (per batch)

1. **Train the Discriminator**: score real images as 1 and generated images as 0. The generated images are `detach()`ed so only the Discriminator is updated.
2. **Train the Generator**: generate fakes and try to make the Discriminator label them as real (1).
3. Every 50 batches, print the Generator and Discriminator losses.
4. At the end of each epoch, save a grid of 8 generated faces as `epoch_N.png`.

## How to Run

1. Open `GAN_Project.ipynb` in Google Colab (set runtime to **GPU**) or Kaggle.
2. Make sure you have Kaggle access for the dataset download (`kagglehub`).
3. Run all cells in order.
4. Generated sample images are saved as `epoch_1.png`, `epoch_2.png`, and so on.

### Requirements

```
torch
torchvision
numpy
matplotlib
pillow
kagglehub
```

## Limitations

- Because the networks are fully connected, they ignore the 2D structure of images. Expect blurry, noisy faces compared to convolutional GANs.
- Training is short (5 epochs), and GAN training can be unstable (loss values alone do not show image quality, so check the saved samples).
- No checkpoints are saved for the models.

## Possible Improvements

- Switch to a **DCGAN** (convolutional Generator with `ConvTranspose2d`, convolutional Discriminator with `BatchNorm`)
- Train for more epochs and save model checkpoints
- Use a fixed noise vector for sample images, so progress is easy to compare across epochs
- Track loss curves and plot them
- Try improved objectives such as WGAN-GP or LSGAN
- Evaluate with FID (Fréchet Inception Distance)
