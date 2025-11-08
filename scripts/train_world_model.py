"""
Training script for World Models.

Usage:
    python scripts/train_world_model.py --env CarRacing-v0
"""

import argparse
import torch
from src.world_model.trainer import WorldModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Train World Model')
    parser.add_argument('--env', type=str, default='CarRacing-v0')
    parser.add_argument('--collect_episodes', type=int, default=1000)
    parser.add_argument('--vae_epochs', type=int, default=50)
    parser.add_argument('--rnn_epochs', type=int, default=50)
    parser.add_argument('--controller_episodes', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("World Models Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = WorldModelTrainer(env_name=args.env, device=args.device)
    
    # Step 1: Collect data
    print("\n[Step 1] Collecting random rollouts...")
    trainer.collect_data(num_episodes=args.collect_episodes)
    
    # Step 2: Train VAE
    print("\n[Step 2] Training VAE...")
    vae_losses = trainer.train_vae(
        latent_dim=args.latent_dim,
        batch_size=32,
        num_epochs=args.vae_epochs,
        lr=1e-4,
        beta=1.0
    )
    
    # Step 3: Train RNN
    print("\n[Step 3] Training RNN...")
    rnn_losses = trainer.train_rnn(
        action_dim=3,
        hidden_dim=256,
        num_mixtures=5,
        batch_size=32,
        num_epochs=args.rnn_epochs,
        lr=1e-3,
        sequence_length=16
    )
    
    # Step 4: Train Controller
    print("\n[Step 4] Training Controller...")
    controller_reward = trainer.train_controller(
        num_episodes=args.controller_episodes,
        num_steps=1000,
        use_evolutionary=True,
        population_size=64,
        num_generations=20
    )
    
    # Save models
    print("\n[Step 5] Saving models...")
    torch.save({
        'vae': trainer.vae.state_dict(),
        'rnn': trainer.rnn.state_dict(),
        'controller': trainer.controller.state_dict(),
    }, 'data/world_model.pth')
    
    print("\nTraining complete!")
    print(f"Final controller reward: {controller_reward:.2f}")


if __name__ == '__main__':
    main()

