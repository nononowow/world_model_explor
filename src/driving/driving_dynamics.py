"""
Driving-specific dynamics model.

Models vehicle dynamics and road interactions in latent space.
"""

from src.rnn.mdn_rnn import MDNRNN


class DrivingDynamics(MDNRNN):
    """
    Dynamics model for driving that predicts future vehicle states.
    
    Extends MDN-RNN to handle driving-specific dynamics.
    """
    
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_mixtures=5):
        """
        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action space (steering, throttle, brake)
            hidden_dim: Hidden dimension of LSTM
            num_mixtures: Number of Gaussian mixtures
        """
        super(DrivingDynamics, self).__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_mixtures=num_mixtures
        )

