"""
Data collection utilities for World Models.

Collects random rollouts from environments for training the VAE and RNN.
"""

import numpy as np
import torch
from collections import deque
import gymnasium as gym


class DataCollector:
    """
    Collects experience data from an environment.
    
    Used to collect random rollouts for training the VAE and RNN.
    """
    
    def __init__(self, env_name, max_episodes=1000, max_steps_per_episode=1000):
        """
        Args:
            env_name: Name of the gymnasium environment
            max_episodes: Maximum number of episodes to collect
            max_steps_per_episode: Maximum steps per episode
        """
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # Storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def collect_random_rollouts(self, num_episodes=None):
        """
        Collect random rollouts from the environment.
        
        Args:
            num_episodes: Number of episodes to collect (default: max_episodes)
        
        Returns:
            Dictionary containing collected data
        """
        if num_episodes is None:
            num_episodes = self.max_episodes
        
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_obs = [obs]
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            
            for step in range(self.max_steps_per_episode):
                # Random action
                action = self.env.action_space.sample()
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store data
                episode_obs.append(next_obs)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)
                
                if done:
                    break
                
                obs = next_obs
            
            # Store episode data
            self.observations.extend(episode_obs)
            self.actions.extend(episode_actions)
            self.rewards.extend(episode_rewards)
            self.dones.extend(episode_dones)
        
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones)
        }
    
    def collect_with_policy(self, policy, num_episodes=None, deterministic=False):
        """
        Collect rollouts using a policy.
        
        Args:
            policy: Policy function that takes observation and returns action
            num_episodes: Number of episodes to collect
            deterministic: Whether to use deterministic policy
        
        Returns:
            Dictionary containing collected data
        """
        if num_episodes is None:
            num_episodes = self.max_episodes
        
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_obs = [obs]
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            
            for step in range(self.max_steps_per_episode):
                # Get action from policy
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    if len(obs_tensor.shape) == 4:  # Image
                        obs_tensor = obs_tensor.permute(0, 3, 1, 2) / 255.0
                else:
                    obs_tensor = obs
                
                action = policy(obs_tensor, deterministic=deterministic)
                
                # Convert to numpy if needed
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                    if action.ndim > 1:
                        action = action[0]
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store data
                episode_obs.append(next_obs)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)
                
                if done:
                    break
                
                obs = next_obs
            
            # Store episode data
            self.observations.extend(episode_obs)
            self.actions.extend(episode_actions)
            self.rewards.extend(episode_rewards)
            self.dones.extend(episode_dones)
        
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones)
        }
    
    def close(self):
        """Close the environment."""
        self.env.close()

