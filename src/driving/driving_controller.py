"""
Driving-specific controller.

Maps latent states to driving actions (steering, throttle, brake).
"""

from src.controller.simple_controller import SimpleController


class DrivingController(SimpleController):
    """
    Controller for driving that outputs steering, throttle, and brake.
    """
    
    def __init__(self, latent_dim=32, action_dim=3):
        """
        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action space (3 for steering, throttle, brake)
        """
        hidden_dims = [256, 256, 128]
        super(DrivingController, self).__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
    
    def act(self, latent_state, deterministic=True):
        """
        Get driving action from latent state.
        
        Args:
            latent_state: Latent state
            deterministic: Whether to use deterministic action
        
        Returns:
            action: [steering, throttle, brake]
                - steering: [-1, 1] (left to right)
                - throttle: [0, 1] (no throttle to full throttle)
                - brake: [0, 1] (no brake to full brake)
        """
        action = super().act(latent_state, deterministic)
        
        # Ensure throttle and brake are in [0, 1]
        # Map from [-1, 1] to [0, 1] for throttle and brake
        action[1] = (action[1] + 1) / 2  # throttle
        action[2] = (action[2] + 1) / 2  # brake
        
        # Clamp to valid ranges
        action[0] = torch.clamp(action[0], -1, 1)  # steering
        action[1] = torch.clamp(action[1], 0, 1)    # throttle
        action[2] = torch.clamp(action[2], 0, 1)    # brake
        
        return action

