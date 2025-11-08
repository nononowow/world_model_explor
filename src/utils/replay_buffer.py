"""
Replay buffer for storing and sampling experience.

Used for training world models and controllers.
"""

import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """
    Simple replay buffer for storing experience tuples.
    """
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, observation, action, reward, next_observation, done):
        """
        Add a transition to the buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        self.buffer.append((observation, action, reward, next_observation, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Batch of transitions as tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        for idx in indices:
            obs, act, rew, next_obs, done = self.buffer[idx]
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            dones.append(done)
        
        return {
            'observations': torch.FloatTensor(np.array(observations)),
            'actions': torch.FloatTensor(np.array(actions)),
            'rewards': torch.FloatTensor(rewards),
            'next_observations': torch.FloatTensor(np.array(next_observations)),
            'dones': torch.BoolTensor(dones)
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class SequenceReplayBuffer:
    """
    Replay buffer for storing sequences of transitions.
    
    Useful for training RNNs that need sequential data.
    """
    
    def __init__(self, capacity=10000, sequence_length=16):
        """
        Args:
            capacity: Maximum number of sequences to store
            sequence_length: Length of sequences
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
    
    def push_sequence(self, observations, actions, rewards, next_observations, dones):
        """
        Add a sequence to the buffer.
        
        Args:
            observations: Sequence of observations [seq_len, ...]
            actions: Sequence of actions [seq_len, ...]
            rewards: Sequence of rewards [seq_len]
            next_observations: Sequence of next observations [seq_len, ...]
            dones: Sequence of done flags [seq_len]
        """
        self.buffer.append({
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones
        })
    
    def sample(self, batch_size):
        """
        Sample a batch of sequences.
        
        Args:
            batch_size: Number of sequences to sample
        
        Returns:
            Batch of sequences as tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []
        
        for idx in indices:
            seq = self.buffer[idx]
            batch_obs.append(seq['observations'])
            batch_actions.append(seq['actions'])
            batch_rewards.append(seq['rewards'])
            batch_next_obs.append(seq['next_observations'])
            batch_dones.append(seq['dones'])
        
        return {
            'observations': torch.FloatTensor(np.array(batch_obs)),
            'actions': torch.FloatTensor(np.array(batch_actions)),
            'rewards': torch.FloatTensor(np.array(batch_rewards)),
            'next_observations': torch.FloatTensor(np.array(batch_next_obs)),
            'dones': torch.BoolTensor(np.array(batch_dones))
        }
    
    def __len__(self):
        return len(self.buffer)

