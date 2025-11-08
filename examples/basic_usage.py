"""
Basic usage example for World Models.

This script demonstrates how to use the world model components.
"""

import torch
import numpy as np
from src.vae.basic_vae import VAE, vae_loss
from src.rnn.mdn_rnn import MDNRNN, mdn_rnn_loss
from src.controller.simple_controller import SimpleController


def example_vae():
    """Example: Train and use a VAE."""
    print("=" * 50)
    print("VAE Example")
    print("=" * 50)
    
    # Create VAE
    vae = VAE(input_channels=3, latent_dim=32)
    
    # Create dummy image (batch_size=4, channels=3, height=64, width=64)
    dummy_image = torch.randn(4, 3, 64, 64)
    
    # Forward pass
    recon_image, mu, logvar, z = vae(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Reconstructed shape: {recon_image.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Latent mean shape: {mu.shape}")
    
    # Compute loss
    loss, recon_loss, kl_loss = vae_loss(recon_image, dummy_image, mu, logvar)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_loss.item():.4f}")


def example_rnn():
    """Example: Use MDN-RNN."""
    print("\n" + "=" * 50)
    print("MDN-RNN Example")
    print("=" * 50)
    
    # Create RNN
    rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_mixtures=5)
    
    # Create dummy sequences
    batch_size = 4
    seq_len = 10
    latent_states = torch.randn(batch_size, seq_len, 32)
    actions = torch.randn(batch_size, seq_len, 3)
    
    # Forward pass
    pi, mu, logvar, hidden = rnn(latent_states, actions)
    
    print(f"Latent states shape: {latent_states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Mixture weights shape: {pi.shape}")
    print(f"Mixture means shape: {mu.shape}")
    
    # Predict next state
    current_latent = latent_states[:, -1, :]  # Last timestep
    next_action = actions[:, -1, :]
    next_latent, _ = rnn.predict(current_latent, next_action)
    print(f"\nPredicted next latent shape: {next_latent.shape}")


def example_controller():
    """Example: Use controller."""
    print("\n" + "=" * 50)
    print("Controller Example")
    print("=" * 50)
    
    # Create controller
    controller = SimpleController(latent_dim=32, action_dim=3)
    
    # Create dummy latent state
    latent_state = torch.randn(32)
    
    # Get action
    action = controller.act(latent_state, deterministic=True)
    
    print(f"Latent state shape: {latent_state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action}")


def example_world_model_workflow():
    """Example: Complete world model workflow."""
    print("\n" + "=" * 50)
    print("World Model Workflow Example")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize components
    vae = VAE(input_channels=3, latent_dim=32).to(device)
    rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256).to(device)
    controller = SimpleController(latent_dim=32, action_dim=3).to(device)
    
    # Simulate environment step
    # 1. Get observation (image)
    observation = torch.randn(3, 64, 64).to(device)
    print(f"Observation shape: {observation.shape}")
    
    # 2. Encode to latent
    mu, logvar = vae.encode(observation.unsqueeze(0))
    z = vae.reparameterize(mu, logvar)
    print(f"Latent state shape: {z.shape}")
    
    # 3. Get action from controller
    action = controller.act(z.squeeze(0), deterministic=True)
    print(f"Action: {action}")
    
    # 4. Predict next latent state
    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
    next_z, hidden = rnn.predict(z.squeeze(0), action_tensor)
    print(f"Predicted next latent shape: {next_z.shape}")
    
    # 5. Decode to see predicted observation
    predicted_obs = vae.decode(next_z)
    print(f"Predicted observation shape: {predicted_obs.shape}")
    
    print("\nWorld model workflow complete!")


if __name__ == '__main__':
    # Run examples
    example_vae()
    example_rnn()
    example_controller()
    example_world_model_workflow()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)

