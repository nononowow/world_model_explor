"""
VAE model for World Models.

This is a specialized VAE for processing game frames (e.g., CarRacing-v0).
"""

from src.vae.basic_vae import VAE, Encoder, Decoder


class WorldModelVAE(VAE):
    """
    VAE specifically designed for World Models on game environments.
    
    Optimized for 64x64 RGB frames from environments like CarRacing-v0.
    """
    
    def __init__(self, latent_dim=32, input_channels=3):
        """
        Args:
            latent_dim: Dimension of latent space
            input_channels: Number of input channels (3 for RGB)
        """
        # Use architecture similar to World Models paper
        hidden_dims = [32, 64, 128, 256]
        super(WorldModelVAE, self).__init__(
            input_channels=input_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )

