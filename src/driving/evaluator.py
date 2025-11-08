"""
Evaluation utilities for driving world models.
"""

import numpy as np
import torch
try:
    import gymnasium as gym
except ImportError:
    import gym as gymnasium
    gym = gymnasium
from tqdm import tqdm


class DrivingEvaluator:
    """
    Evaluator for driving world models.
    """
    
    def __init__(self, env_name, world_model, device='cuda'):
        """
        Args:
            env_name: Name of environment or environment instance
            world_model: DrivingWorldModel instance
            device: Device to run on
        """
        if isinstance(env_name, str):
            self.env = gym.make(env_name, render_mode='rgb_array')
        else:
            self.env = env_name
        
        self.world_model = world_model
        self.device = device
    
    def evaluate(self, num_episodes=10, max_steps=1000, render=False):
        """
        Evaluate the world model on the environment.
        
        Args:
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            render: Whether to render episodes
        
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            success = False
            
            # Encode initial observation
            # Extract state if available (for multi-modal)
            if isinstance(obs, dict):
                image = obs['image']
                state = obs['state']
            else:
                image = obs
                state = np.array([0.0, 0.0, 0.0, 0.0])  # Dummy state
            
            latent = self.world_model.encode(image, state)
            hidden = None
            
            for step in range(max_steps):
                # Get action
                action = self.world_model.act(latent, deterministic=True)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
                
                if done:
                    success = not truncated  # Success if not truncated
                    break
                
                # Predict next latent state
                if isinstance(next_obs, dict):
                    next_image = next_obs['image']
                    next_state = next_obs['state']
                else:
                    next_image = next_obs
                    next_state = np.array([0.0, 0.0, 0.0, 0.0])
                
                # Use world model prediction instead of encoding
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                latent, hidden = self.world_model.predict(latent, action_tensor, hidden)
                latent = latent.squeeze(0).cpu().numpy()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_success.append(success)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(episode_success),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def close(self):
        """Close the environment."""
        self.env.close()

