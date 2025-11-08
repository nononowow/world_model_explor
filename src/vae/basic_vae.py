"""
Basic Variational Autoencoder (VAE) implementation from scratch.

This module implements a VAE to learn latent representations of observations.
The VAE consists of:
- Encoder: Maps observations to latent distribution parameters (mean, log_var)
- Decoder: Reconstructs observations from latent samples
- Loss: Reconstruction loss + KL divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Encoder(nn.Module):
    """Encoder network that maps observations to latent distribution parameters."""
    
    def __init__(self, input_channels=3, latent_dim=32, hidden_dims=[64, 128, 256, 512]):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
        """
        super(Encoder, self).__init__()
        
        modules = []
        in_channels = input_channels
        
        # Build encoder layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size after convolutions
        # Assuming input is 64x64, after 4 stride-2 convolutions: 4x4
        self.flattened_size = hidden_dims[-1] * 4 * 4
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, channels, height, width]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network that reconstructs observations from latent samples."""
    
    def __init__(self, latent_dim=32, output_channels=3, hidden_dims=[512, 256, 128, 64]):
        """
        Args:
            latent_dim: Dimension of latent space
            output_channels: Number of output channels
            hidden_dims: List of hidden layer dimensions (reversed from encoder)
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.flattened_size = hidden_dims[0] * 4 * 4
        
        # Project latent to feature map size
        self.fc = nn.Linear(latent_dim, self.flattened_size)
        
        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent sample [batch_size, latent_dim]
        
        Returns:
            Reconstructed observation [batch_size, channels, height, width]
        """
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)  # Reshape to feature map
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.
    
    The VAE learns to encode observations into a latent space and decode them back.
    Uses reparameterization trick for differentiable sampling.
    """
    
    def __init__(self, input_channels=3, latent_dim=32, hidden_dims=[64, 128, 256, 512]):
        """
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
        """
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_channels, list(reversed(hidden_dims)))
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1).
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input observation [batch_size, channels, height, width]
        
        Returns:
            recon_x: Reconstructed observation
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent vector
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z
    
    def encode(self, x):
        """Encode observation to latent distribution parameters."""
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent sample to observation."""
        return self.decoder(z)
    
    def sample(self, num_samples, device):
        """Sample random observations from the latent space."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss: reconstruction loss + KL divergence.
    
    Args:
        recon_x: Reconstructed observation
        x: Original observation
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (β-VAE)
    
    Returns:
        loss: Total VAE loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (binary cross-entropy for normalized images)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

