"""
Controller for World Models.

This wraps the simple controller for use in World Models.
"""

from src.controller.simple_controller import SimpleController, RNNController


class WorldModelController(SimpleController):
    """
    Controller for World Models.
    
    Maps latent states to actions. Can be trained using evolutionary strategies
    or gradient-based methods.
    """
    
    def __init__(self, latent_dim=32, action_dim=3):
        """
        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action space
        """
        hidden_dims = [256, 256]
        super(WorldModelController, self).__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )

