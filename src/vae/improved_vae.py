"""
Improved VAE architectures.

Includes β-VAE for disentangled representations and ResNet-based architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vae.basic_vae import VAE, Encoder, Decoder


class ResBlock(nn.Module):
    """Residual block for ResNet-based encoder/decoder."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet-based encoder."""
    
    def __init__(self, input_channels=3, latent_dim=32):
        super(ResNetEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ResNetDecoder(nn.Module):
    """ResNet-based decoder."""
    
    def __init__(self, latent_dim=32, output_channels=3):
        super(ResNetDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project latent to feature map
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Residual blocks (reverse of encoder)
        self.layer1 = self._make_layer(512, 256, 2, stride=2, upsample=True)
        self.layer2 = self._make_layer(256, 128, 2, stride=2, upsample=True)
        self.layer3 = self._make_layer(128, 64, 2, stride=2, upsample=True)
        self.layer4 = self._make_layer(64, 64, 2, stride=1, upsample=False)
        
        # Final upsampling and output
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, upsample):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False))
        layers.append(ResBlock(in_channels, out_channels, stride=1))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x


class BetaVAE(VAE):
    """
    β-VAE for learning disentangled representations.
    
    Uses a higher weight (β > 1) on KL divergence to encourage
    more independent latent factors.
    """
    
    def __init__(self, input_channels=3, latent_dim=32, hidden_dims=[64, 128, 256, 512], beta=4.0):
        """
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            beta: Weight for KL divergence (higher = more disentanglement)
        """
        super(BetaVAE, self).__init__(input_channels, latent_dim, hidden_dims)
        self.beta = beta
    
    def forward(self, x, beta=None):
        """
        Forward pass with optional beta override.
        
        Args:
            x: Input observation
            beta: Optional beta value (overrides self.beta)
        """
        recon_x, mu, logvar, z = super().forward(x)
        return recon_x, mu, logvar, z


class ResNetVAE(VAE):
    """
    ResNet-based VAE for better feature extraction.
    """
    
    def __init__(self, input_channels=3, latent_dim=32):
        """
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
        """
        from src.vae.basic_vae import VAE
        
        # Create custom encoder and decoder
        encoder = ResNetEncoder(input_channels, latent_dim)
        decoder = ResNetDecoder(latent_dim, input_channels)
        
        # Initialize VAE with custom components
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z
    
    def encode(self, x):
        """Encode observation to latent distribution parameters."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent sample to observation."""
        return self.decoder(z)
    
    def sample(self, num_samples, device):
        """Sample random observations from the latent space."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

