"""
Complete driving world model integrating all components.
"""

import torch
import numpy as np
from src.driving.driving_vae import DrivingVAE
from src.driving.driving_dynamics import DrivingDynamics
from src.driving.driving_controller import DrivingController


class DrivingWorldModel:
    """
    Complete world model for autonomous driving.
    
    Integrates VAE, dynamics model, and controller.
    """
    
    def __init__(self, latent_dim=32, action_dim=3, state_dim=4, device='cuda'):
        """
        Args:
            latent_dim: Dimension of latent space
            action_dim: Dimension of action space
            state_dim: Dimension of vehicle state
            device: Device to run on
        """
        self.device = device
        self.latent_dim = latent_dim
        
        # Initialize components
        self.vae = DrivingVAE(
            image_channels=3,
            state_dim=state_dim,
            latent_dim=latent_dim
        ).to(device)
        
        self.dynamics = DrivingDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_mixtures=5
        ).to(device)
        
        self.controller = DrivingController(
            latent_dim=latent_dim,
            action_dim=action_dim
        ).to(device)
    
    def encode(self, image, state):
        """Encode observation to latent state."""
        self.vae.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = torch.FloatTensor(image).unsqueeze(0)
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            
            # Normalize image
            if image.max() > 1.0:
                image = image / 255.0
            
            # Reshape if needed
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
            
            image = image.to(self.device)
            state = state.to(self.device)
            
            mu, logvar = self.vae.encode(image, state)
            z = self.vae.reparameterize(mu, logvar)
            return z
    
    def predict(self, latent, action, hidden=None):
        """Predict next latent state."""
        self.dynamics.eval()
        with torch.no_grad():
            if isinstance(latent, np.ndarray):
                latent = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
            if isinstance(action, np.ndarray):
                action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            next_latent, hidden = self.dynamics.predict(latent, action, hidden)
            return next_latent, hidden
    
    def act(self, latent, deterministic=True):
        """Get action from latent state."""
        self.controller.eval()
        with torch.no_grad():
            if isinstance(latent, np.ndarray):
                latent = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
            
            action = self.controller.act(latent, deterministic=deterministic)
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            return action
    
    def to_dict(self):
        """Convert to dict format for compatibility with planners."""
        return {
            'vae': self.vae,
            'rnn': self.dynamics,  # Use 'rnn' key for compatibility
            'controller': self.controller
        }

