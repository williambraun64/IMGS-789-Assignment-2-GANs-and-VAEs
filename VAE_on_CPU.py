# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:33:12 2024

@author: parsalab_kiosk
"""

"""
Created on Thu Nov  7 12:46:39 2024

@author: parsalab_kiosk
"""

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        log_sigma = self.linear3(x)
        sigma = torch.exp(log_sigma)
        z = mu + sigma * self.N.sample(mu.shape).to(device)
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


def train(autoencoder, data, epochs):
    opt = torch.optim.Adam(autoencoder.parameters())
    elbo_loss_history, kl_history = [], []

    for epoch in range(epochs):
        total_loss, total_kl = 0, 0
        for x, _ in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            # Separate reconstruction loss and KL divergence
            reconstruction_loss = ((x - x_hat)**2).sum()
            kl_divergence = autoencoder.encoder.kl
            loss = reconstruction_loss + kl_divergence  # ELBO Loss
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_kl += kl_divergence.item()
        
        elbo_loss_history.append(total_loss / len(data.dataset))
        kl_history.append(total_kl / len(data.dataset))
        print(f"Epoch {epoch+1}/{epochs}, ELBO Loss: {total_loss / len(data.dataset):.4f}, KL-Divergence: {total_kl / len(data.dataset):.4f}")
    
    return elbo_loss_history, kl_history

# %%
# Set parameters and load data 
latent_dims = 2
epochs = 40

# Load data
data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=True),
    batch_size=128, shuffle=True
)

# %%

# Initialize and train the VAE
vae = VariationalAutoencoder(latent_dims).to(device)
elbo_loss_history, kl_history = train(vae, data, epochs)

# Plot ELBO loss and KL-divergence
plt.figure(figsize=(10, 4))
plt.plot(elbo_loss_history, label='ELBO Loss')
plt.plot(kl_history, label='KL-Divergence')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("ELBO Loss and KL-Divergence During Training")
plt.show()

# %%


# Function to visualize reconstruction of a batch of images
def visualize_reconstruction(autoencoder, data, n=10):
    with torch.no_grad():
        x, _ = next(iter(data))
        x = x[:n].to(device)
        x_hat = autoencoder(x).cpu()
        img = np.zeros((28 * 2, n * 28))  # Original + reconstructed images
        for i in range(n):
            img[:28, i*28:(i+1)*28] = x[i].cpu().squeeze()
            img[28:, i*28:(i+1)*28] = x_hat[i].squeeze()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title("Original and Reconstructed Images")
        plt.show()

# Visualize reconstructions
visualize_reconstruction(vae, data, n=10)


# %% 

# Function to add noise to the dataset for anomaly detection
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clip(noisy_images, 0., 1.)

# Function to calculate reconstruction error
def calculate_reconstruction_error(autoencoder, images):
    with torch.no_grad():
        images = images.to(device)
        reconstructed = autoencoder(images).cpu()
        error = F.mse_loss(reconstructed, images, reduction='none').mean(dim=[1, 2, 3])
    return error

# %%

# Generate anomalous data by adding noise to MNIST images
anomalous_data = [add_noise(images) for images, _ in data]

# %%

# Collect reconstruction errors for normal and anomalous data
normal_errors, anomalous_errors = [], []

for images, _ in data:
    normal_errors.extend(calculate_reconstruction_error(vae, images).numpy())

for images in anomalous_data:
    anomalous_errors.extend(calculate_reconstruction_error(vae, images).numpy())

# Plot distribution of reconstruction errors
plt.figure(figsize=(10, 5))
plt.hist(normal_errors, bins=50, alpha=0.6, label="Normal Data")
plt.hist(anomalous_errors, bins=50, alpha=0.6, label="Anomalous Data")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()

# %%

# Set threshold for anomaly detection based on normal data
threshold = 0.1  # 95th percentile of normal errors

# Function to classify images based on reconstruction error
def classify_anomalies(errors, threshold):
    return ["Anomalous" if error > threshold else "Normal" for error in errors]

# Evaluate the model on some test images
test_iter = iter(data)
test_images, _ = next(test_iter)
noisy_test_images = add_noise(test_images)

# Calculate errors for test images
test_errors = calculate_reconstruction_error(vae, test_images)
noisy_test_errors = calculate_reconstruction_error(vae, noisy_test_images)

# Classify images as normal or anomalous
test_classification = classify_anomalies(test_errors.numpy(), threshold)
noisy_test_classification = classify_anomalies(noisy_test_errors.numpy(), threshold)

# %%

# Visualize some examples of normal and anomalous classifications
def visualize_classifications(images, errors, classifications, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"{classifications[i]}\nError: {errors[i]:.4f}")
        plt.axis("off")
    plt.show()

print("Normal Test Images:")
visualize_classifications(test_images, test_errors, test_classification)

print("Noisy (Anomalous) Test Images:")
visualize_classifications(noisy_test_images, noisy_test_errors, noisy_test_classification)


# %%

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            plt.show()
            break

plot_latent(vae, data)

