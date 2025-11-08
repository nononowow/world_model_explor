"""
RNN model for World Models.

This wraps the MDN-RNN for use in the World Models architecture.
"""

from src.rnn.mdn_rnn import MDNRNN


class WorldModelRNN(MDNRNN):
    """
    RNN model for World Models.
    
    Predicts future latent states given current latent state and action.
    """
    
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_mixtures=5):
        """
        Args:
            latent_dim: Dimension of latent state from VAE
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension of LSTM
            num_mixtures: Number of Gaussian mixtures
        """
        super(WorldModelRNN, self).__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_mixtures=num_mixtures
        )

