"""
Training pipeline for World Models.

Implements the three-stage training process:
1. Collect random rollouts
2. Train VAE on collected frames
3. Train RNN on latent sequences
4. Train controller in latent space
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from src.world_model.vae_model import WorldModelVAE
from src.world_model.rnn_model import WorldModelRNN
from src.world_model.controller import WorldModelController
from src.vae.basic_vae import vae_loss
from src.rnn.mdn_rnn import mdn_rnn_loss
from src.utils.data_collector import DataCollector
from src.utils.visualization import visualize_reconstruction, plot_training_curves


class WorldModelTrainer:
    """
    Trainer for World Models.
    
    Handles the complete training pipeline for world models.
    """
    
    def __init__(self, env_name='CarRacing-v0', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            env_name: Name of the environment
            device: Device to run on
        """
        self.env_name = env_name
        self.device = device
        
        # Models
        self.vae = None
        self.rnn = None
        self.controller = None
        
        # Training data
        self.observations = None
        self.actions = None
    
    def collect_data(self, num_episodes=1000):
        """
        Collect random rollouts from the environment.
        
        Args:
            num_episodes: Number of episodes to collect
        """
        print("Collecting random rollouts...")
        collector = DataCollector(self.env_name, max_episodes=num_episodes)
        data = collector.collect_random_rollouts(num_episodes)
        collector.close()
        
        self.observations = data['observations']
        self.actions = data['actions']
        
        print(f"Collected {len(self.observations)} frames")
        return data
    
    def train_vae(self, latent_dim=32, batch_size=32, num_epochs=50, lr=1e-4, beta=1.0):
        """
        Train VAE on collected observations.
        
        Args:
            latent_dim: Dimension of latent space
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            lr: Learning rate
            beta: Weight for KL divergence
        """
        print("Training VAE...")
        
        # Initialize VAE
        self.vae = WorldModelVAE(latent_dim=latent_dim, input_channels=3).to(self.device)
        
        # Prepare data
        # Convert observations to tensor and normalize
        obs_tensor = torch.FloatTensor(self.observations)
        if obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor / 255.0
        
        # Reshape if needed: [N, H, W, C] -> [N, C, H, W]
        if obs_tensor.shape[-1] == 3:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        
        # Resize to 64x64 if needed
        if obs_tensor.shape[2] != 64 or obs_tensor.shape[3] != 64:
            from torchvision.transforms import Resize
            resize = Resize((64, 64))
            obs_tensor = resize(obs_tensor)
        
        dataset = TensorDataset(obs_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        
        # Training loop
        losses = {'total': [], 'recon': [], 'kl': []}
        
        for epoch in range(num_epochs):
            epoch_losses = {'total': [], 'recon': [], 'kl': []}
            
            for batch_idx, (images,) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                images = images.to(self.device)
                
                # Forward pass
                recon_images, mu, logvar, z = self.vae(images)
                
                # Loss
                loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta=beta)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Store losses
                epoch_losses['total'].append(loss.item())
                epoch_losses['recon'].append(recon_loss.item())
                epoch_losses['kl'].append(kl_loss.item())
            
            # Average losses
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            for k, v in avg_losses.items():
                losses[k].append(v)
            
            print(f"Epoch {epoch+1}: Loss={avg_losses['total']:.4f}, "
                  f"Recon={avg_losses['recon']:.4f}, KL={avg_losses['kl']:.4f}")
        
        print("VAE training complete!")
        return losses
    
    def train_rnn(self, action_dim=3, hidden_dim=256, num_mixtures=5, 
                   batch_size=32, num_epochs=50, lr=1e-3, sequence_length=16):
        """
        Train RNN on latent sequences.
        
        Args:
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension of RNN
            num_mixtures: Number of Gaussian mixtures
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            lr: Learning rate
            sequence_length: Length of sequences for training
        """
        print("Training RNN...")
        
        if self.vae is None:
            raise ValueError("VAE must be trained first!")
        
        # Initialize RNN
        latent_dim = self.vae.latent_dim
        self.rnn = WorldModelRNN(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_mixtures=num_mixtures
        ).to(self.device)
        
        # Encode observations to latent space
        print("Encoding observations to latent space...")
        self.vae.eval()
        latent_states = []
        
        obs_tensor = torch.FloatTensor(self.observations)
        if obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor / 255.0
        if obs_tensor.shape[-1] == 3:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        if obs_tensor.shape[2] != 64 or obs_tensor.shape[3] != 64:
            from torchvision.transforms import Resize
            resize = Resize((64, 64))
            obs_tensor = resize(obs_tensor)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(obs_tensor), batch_size)):
                batch = obs_tensor[i:i+batch_size].to(self.device)
                mu, logvar = self.vae.encode(batch)
                z = self.vae.reparameterize(mu, logvar)
                latent_states.append(z.cpu())
        
        latent_states = torch.cat(latent_states, dim=0).numpy()
        
        # Prepare sequences
        print("Preparing sequences...")
        sequences = []
        for i in range(len(latent_states) - sequence_length):
            seq_latent = latent_states[i:i+sequence_length]
            seq_next_latent = latent_states[i+1:i+sequence_length+1]
            seq_actions = self.actions[i:i+sequence_length]
            
            sequences.append({
                'latent': seq_latent,
                'next_latent': seq_next_latent,
                'actions': seq_actions
            })
        
        # Create data loader
        def collate_fn(batch):
            latents = torch.FloatTensor([s['latent'] for s in batch])
            next_latents = torch.FloatTensor([s['next_latent'] for s in batch])
            actions = torch.FloatTensor([s['actions'] for s in batch])
            return latents, next_latents, actions
        
        dataloader = DataLoader(sequences, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Optimizer
        optimizer = optim.Adam(self.rnn.parameters(), lr=lr)
        
        # Training loop
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, (latents, next_latents, actions) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            ):
                latents = latents.to(self.device)
                next_latents = next_latents.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                pi, mu, logvar, _ = self.rnn(latents, actions)
                
                # Loss
                loss = mdn_rnn_loss(pi, mu, logvar, next_latents)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        print("RNN training complete!")
        return losses
    
    def train_controller(self, num_episodes=100, num_steps=1000, lr=1e-3, 
                        use_evolutionary=True, population_size=64, num_generations=20):
        """
        Train controller in latent space.
        
        Args:
            num_episodes: Number of episodes for evaluation
            num_steps: Maximum steps per episode
            lr: Learning rate (for gradient-based)
            use_evolutionary: Whether to use evolutionary strategies
            population_size: Population size for evolutionary strategies
            num_generations: Number of generations for evolutionary strategies
        """
        print("Training controller...")
        
        if self.vae is None or self.rnn is None:
            raise ValueError("VAE and RNN must be trained first!")
        
        # Initialize controller
        latent_dim = self.vae.latent_dim
        action_dim = len(self.actions[0]) if len(self.actions) > 0 else 3
        self.controller = WorldModelController(latent_dim=latent_dim, action_dim=action_dim).to(self.device)
        
        if use_evolutionary:
            return self._train_controller_evolutionary(num_episodes, num_steps, population_size, num_generations)
        else:
            return self._train_controller_gradient(num_episodes, num_steps, lr)
    
    def _train_controller_evolutionary(self, num_episodes, num_steps, population_size, num_generations):
        """Train controller using evolutionary strategies."""
        import gymnasium as gym
        env = gym.make(self.env_name, render_mode='rgb_array')
        
        # Get initial parameters
        initial_params = [p.clone() for p in self.controller.parameters()]
        
        best_reward = -float('inf')
        best_params = None
        
        for generation in range(num_generations):
            # Create population
            population = []
            for _ in range(population_size):
                # Add noise to parameters
                params = [p.clone() + torch.randn_like(p) * 0.1 for p in initial_params]
                population.append(params)
            
            # Evaluate population
            rewards = []
            for params in tqdm(population, desc=f"Generation {generation+1}/{num_generations}"):
                # Set parameters
                for param, new_param in zip(self.controller.parameters(), params):
                    param.data = new_param
                
                # Evaluate
                total_reward = 0
                for episode in range(num_episodes):
                    obs, _ = env.reset()
                    episode_reward = 0
                    
                    self.vae.eval()
                    self.rnn.eval()
                    self.controller.eval()
                    
                    with torch.no_grad():
                        # Encode initial observation
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        if obs_tensor.max() > 1.0:
                            obs_tensor = obs_tensor / 255.0
                        if obs_tensor.shape[-1] == 3:
                            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                        if obs_tensor.shape[2] != 64 or obs_tensor.shape[3] != 64:
                            from torchvision.transforms import Resize
                            resize = Resize((64, 64))
                            obs_tensor = resize(obs_tensor)
                        
                        obs_tensor = obs_tensor.to(self.device)
                        mu, logvar = self.vae.encode(obs_tensor)
                        z = self.vae.reparameterize(mu, logvar)
                        
                        hidden = None
                        
                        for step in range(num_steps):
                            # Get action from controller
                            action = self.controller.act(z, deterministic=True)
                            
                            # Convert to numpy
                            if isinstance(action, torch.Tensor):
                                action = action.detach().cpu().numpy()
                            
                            # Step environment
                            next_obs, reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            
                            episode_reward += reward
                            
                            if done:
                                break
                            
                            # Predict next latent state
                            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                            z, hidden = self.rnn.predict(z, action_tensor, hidden)
                    
                    total_reward += episode_reward
                
                rewards.append(total_reward / num_episodes)
            
            # Select best
            best_idx = np.argmax(rewards)
            if rewards[best_idx] > best_reward:
                best_reward = rewards[best_idx]
                best_params = [p.clone() for p in population[best_idx]]
                initial_params = best_params
            
            print(f"Generation {generation+1}: Best reward = {best_reward:.2f}, Mean = {np.mean(rewards):.2f}")
        
        # Set best parameters
        for param, best_param in zip(self.controller.parameters(), best_params):
            param.data = best_param
        
        env.close()
        print("Controller training complete!")
        return best_reward
    
    def _train_controller_gradient(self, num_episodes, num_steps, lr):
        """Train controller using gradient-based methods."""
        # This is a simplified version - full implementation would use policy gradients
        optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        
        # For now, just return a placeholder
        print("Gradient-based controller training not fully implemented yet.")
        return 0.0

