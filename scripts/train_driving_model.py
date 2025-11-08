"""
Training script for driving world model.

Usage:
    python scripts/train_driving_model.py --env SimpleDrivingEnv
"""

import argparse
import torch
import numpy as np
from src.driving.driving_world_model import DrivingWorldModel
from src.driving.driving_vae import DrivingVAE
from src.driving.driving_dynamics import DrivingDynamics
from src.driving.driving_controller import DrivingController
from src.utils.data_collector import DataCollector
from src.vae.basic_vae import vae_loss
from src.rnn.mdn_rnn import mdn_rnn_loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_driving_model(env_name='SimpleDrivingEnv', epochs_vae=50, epochs_rnn=50, 
                       epochs_controller=20, device='cuda'):
    """Train driving world model."""
    print("=" * 50)
    print("Driving World Model Training")
    print("=" * 50)
    
    # Initialize world model
    world_model = DrivingWorldModel(
        latent_dim=32,
        action_dim=3,
        state_dim=4,
        device=device
    )
    
    # Step 1: Collect data
    print("\n[Step 1] Collecting data...")
    from src.environments.driving_env import SimpleDrivingEnv
    env = SimpleDrivingEnv(render_mode='rgb_array')
    
    collector = DataCollector(env_name=None, max_episodes=500)
    collector.env = env
    data = collector.collect_random_rollouts(num_episodes=500)
    
    observations = data['observations']
    actions = data['actions']
    
    # Extract states (simplified - in real scenario would come from env)
    states = np.zeros((len(observations), 4))  # Dummy states
    
    print(f"Collected {len(observations)} frames")
    
    # Step 2: Train VAE
    print("\n[Step 2] Training VAE...")
    obs_tensor = torch.FloatTensor(observations)
    if obs_tensor.max() > 1.0:
        obs_tensor = obs_tensor / 255.0
    if obs_tensor.shape[-1] == 3:
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)
    
    # Resize to 64x64
    from torchvision.transforms import Resize
    resize = Resize((64, 64))
    obs_tensor = resize(obs_tensor)
    
    states_tensor = torch.FloatTensor(states)
    
    dataset = TensorDataset(obs_tensor, states_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer_vae = torch.optim.Adam(world_model.vae.parameters(), lr=1e-4)
    
    for epoch in range(epochs_vae):
        epoch_loss = 0
        for images, states_batch in tqdm(dataloader, desc=f"VAE Epoch {epoch+1}"):
            images = images.to(device)
            states_batch = states_batch.to(device)
            
            recon_image, recon_state, mu, logvar, z = world_model.vae(images, states_batch)
            
            # Loss: image reconstruction + state reconstruction + KL
            image_loss = torch.nn.functional.mse_loss(recon_image, images)
            state_loss = torch.nn.functional.mse_loss(recon_state, states_batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = image_loss + state_loss + 0.1 * kl_loss
            
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
            
            epoch_loss += loss.item()
        
        print(f"VAE Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")
    
    # Step 3: Train RNN
    print("\n[Step 3] Training RNN...")
    # Encode observations
    world_model.vae.eval()
    latent_states = []
    with torch.no_grad():
        for i in range(0, len(obs_tensor), 32):
            batch_images = obs_tensor[i:i+32].to(device)
            batch_states = states_tensor[i:i+32].to(device)
            mu, logvar = world_model.vae.encode(batch_images, batch_states)
            z = world_model.vae.reparameterize(mu, logvar)
            latent_states.append(z.cpu())
    
    latent_states = torch.cat(latent_states, dim=0).numpy()
    
    # Prepare sequences
    sequence_length = 16
    sequences = []
    for i in range(len(latent_states) - sequence_length):
        seq_latent = latent_states[i:i+sequence_length]
        seq_next_latent = latent_states[i+1:i+sequence_length+1]
        seq_actions = actions[i:i+sequence_length]
        sequences.append((seq_latent, seq_next_latent, seq_actions))
    
    optimizer_rnn = torch.optim.Adam(world_model.dynamics.parameters(), lr=1e-3)
    
    for epoch in range(epochs_rnn):
        epoch_loss = 0
        np.random.shuffle(sequences)
        
        for i in range(0, len(sequences), 32):
            batch = sequences[i:i+32]
            latents = torch.FloatTensor([s[0] for s in batch]).to(device)
            next_latents = torch.FloatTensor([s[1] for s in batch]).to(device)
            actions_batch = torch.FloatTensor([s[2] for s in batch]).to(device)
            
            pi, mu, logvar, _ = world_model.dynamics(latents, actions_batch)
            loss = mdn_rnn_loss(pi, mu, logvar, next_latents)
            
            optimizer_rnn.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.dynamics.parameters(), max_norm=1.0)
            optimizer_rnn.step()
            
            epoch_loss += loss.item()
        
        print(f"RNN Epoch {epoch+1}: Loss = {epoch_loss / (len(sequences) // 32):.4f}")
    
    # Step 4: Train Controller (simplified)
    print("\n[Step 4] Training Controller...")
    # Use evolutionary strategy similar to world model trainer
    from src.world_model.trainer import WorldModelTrainer
    trainer = WorldModelTrainer(env_name=None, device=device)
    trainer.vae = world_model.vae
    trainer.rnn = world_model.dynamics
    trainer.controller = world_model.controller
    trainer.observations = observations
    trainer.actions = actions
    
    # Create a simple environment wrapper
    class EnvWrapper:
        def __init__(self, env):
            self.env = env
            self.action_space = type('obj', (object,), {'sample': lambda: np.array([0.0, 0.5, 0.0])})()
    
    trainer.env_name = 'SimpleDrivingEnv'
    # Note: Controller training would need environment integration
    
    # Save model
    print("\n[Step 5] Saving model...")
    torch.save({
        'vae': world_model.vae.state_dict(),
        'dynamics': world_model.dynamics.state_dict(),
        'controller': world_model.controller.state_dict(),
    }, 'data/driving_world_model.pth')
    
    print("Training complete!")
    return world_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Driving World Model')
    parser.add_argument('--env', type=str, default='SimpleDrivingEnv')
    parser.add_argument('--epochs_vae', type=int, default=50)
    parser.add_argument('--epochs_rnn', type=int, default=50)
    parser.add_argument('--epochs_controller', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    train_driving_model(
        env_name=args.env,
        epochs_vae=args.epochs_vae,
        epochs_rnn=args.epochs_rnn,
        epochs_controller=args.epochs_controller,
        device=args.device
    )

