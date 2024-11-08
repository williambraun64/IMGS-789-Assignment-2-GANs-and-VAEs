# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# %% Load MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# %% Define generator and discriminator networks

latent_dim = 100  # Size of the noise vector
image_dim = 28 * 28  # 784 (for 28x28 images)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def generate_images(generator, num_images=16):
    # Generate random noise and pass it through the generator
    z = torch.randn(num_images, latent_dim)
    with torch.no_grad():  # No need to calculate gradients for inference
        fake_images = generator(z)
    
    # Rescale images from [-1, 1] to [0, 1]
    fake_images = fake_images.view(num_images, 28, 28).cpu().numpy()
    fake_images = (fake_images + 1) / 2  # Normalize to [0, 1]

    # Plot the images in a grid
    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fake_images[i], cmap='gray')
        plt.axis('off')
    plt.show()



# %%Initialize generator and discriminator

generator = Generator()
discriminator = Discriminator()

# Loss and optimizers
criterion = nn.BCELoss()
lr = 0.0002
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# %%Train the GAN

num_epochs = 10
loss_g_list, loss_d_list = [], []

for epoch in range(num_epochs):
    loss_g_epoch, loss_d_epoch = 0, 0
    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1)

        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizer_d.zero_grad()
        outputs = discriminator(real_images)
        loss_d_real = criterion(outputs, real_labels)
        loss_d_real.backward()

        # Generate fake images and train discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        loss_d_fake = criterion(outputs, fake_labels)
        loss_d_fake.backward()
        optimizer_d.step()

        loss_d = loss_d_real + loss_d_fake
        loss_d_epoch += loss_d.item()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        loss_g = criterion(outputs, real_labels)  # flip real labels
        loss_g.backward()
        optimizer_g.step()

        loss_g_epoch += loss_g.item()

    loss_g_list.append(loss_g_epoch / len(train_loader))
    loss_d_list.append(loss_d_epoch / len(train_loader))

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')
    
# %% Generate and display images
generate_images(generator, num_images=16)

# %% Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(loss_g_list, label="Generator Loss")
plt.plot(loss_d_list, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

