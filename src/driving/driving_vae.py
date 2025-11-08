"""
Driving-specific VAE for multi-modal inputs.

Handles camera images and vehicle state (speed, steering angle, etc.).
"""

import torch
import torch.nn as nn
from src.vae.basic_vae import VAE, Encoder, Decoder


class MultiModalEncoder(nn.Module):
    """
    Encoder that processes both camera images and vehicle state.
    """
    
    def __init__(self, image_channels=3, state_dim=4, latent_dim=32, hidden_dims=[64, 128, 256, 512]):
        """
        Args:
            image_channels: Number of image channels (3 for RGB)
            state_dim: Dimension of vehicle state (speed, steering, etc.)
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for image encoder
        """
        super(MultiModalEncoder, self).__init__()
        
        # Image encoder (same as basic VAE encoder)
        from src.vae.basic_vae import Encoder
        self.image_encoder = Encoder(image_channels, latent_dim // 2, hidden_dims)
        
        # State encoder (simple MLP)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim // 2)
        )
        
        # Combine image and state latents
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, image, state):
        """
        Args:
            image: [batch_size, channels, height, width]
            state: [batch_size, state_dim]
        
        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        # Encode image
        image_mu, image_logvar = self.image_encoder(image)
        
        # Encode state
        state_latent = self.state_encoder(state)
        
        # Combine
        combined = torch.cat([image_mu, state_latent], dim=-1)
        
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar


class MultiModalDecoder(nn.Module):
    """
    Decoder that reconstructs both image and state.
    """
    
    def __init__(self, latent_dim=32, image_channels=3, state_dim=4, hidden_dims=[512, 256, 128, 64]):
        """
        Args:
            latent_dim: Dimension of latent space
            image_channels: Number of image channels
            state_dim: Dimension of vehicle state
            hidden_dims: Hidden dimensions for image decoder
        """
        super(MultiModalDecoder, self).__init__()
        
        # Image decoder
        from src.vae.basic_vae import Decoder
        self.image_decoder = Decoder(latent_dim, image_channels, hidden_dims)
        
        # State decoder
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    
    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_dim]
        
        Returns:
            image: [batch_size, channels, height, width]
            state: [batch_size, state_dim]
        """
        image = self.image_decoder(z)
        state = self.state_decoder(z)
        return image, state


class DrivingVAE(nn.Module):
    """
    VAE for driving that handles multi-modal inputs.
    """
    
    def __init__(self, image_channels=3, state_dim=4, latent_dim=32, hidden_dims=[64, 128, 256, 512]):
        """
        Args:
            image_channels: Number of image channels
            state_dim: Dimension of vehicle state
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions
        """
        super(DrivingVAE, self).__init__()
        
        self.encoder = MultiModalEncoder(image_channels, state_dim, latent_dim, hidden_dims)
        self.decoder = MultiModalDecoder(latent_dim, image_channels, state_dim, list(reversed(hidden_dims)))
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, image, state):
        """
        Forward pass.
        
        Args:
            image: [batch_size, channels, height, width]
            state: [batch_size, state_dim]
        
        Returns:
            recon_image, recon_state, mu, logvar, z
        """
        mu, logvar = self.encoder(image, state)
        z = self.reparameterize(mu, logvar)
        recon_image, recon_state = self.decoder(z)
        return recon_image, recon_state, mu, logvar, z
    
    def encode(self, image, state):
        """Encode to latent distribution."""
        return self.encoder(image, state)
    
    def decode(self, z):
        """Decode from latent."""
        return self.decoder(z)

