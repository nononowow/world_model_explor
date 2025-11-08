"""
Training script for basic VAE.

Usage:
    python scripts/train_vae.py --dataset mnist --epochs 50
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from src.vae.basic_vae import VAE, vae_loss


def train_vae(dataset_name='mnist', epochs=50, batch_size=128, lr=1e-3, beta=1.0, latent_dim=32):
    """Train VAE on a dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    if dataset_name == 'mnist':
        transform = transforms.Compose([ToTensor()])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        input_channels = 1
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([ToTensor()])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        input_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize VAE
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    # Training loop
    losses = {'total': [], 'recon': [], 'kl': []}
    
    for epoch in range(epochs):
        epoch_losses = {'total': [], 'recon': [], 'kl': []}
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            recon_images, mu, logvar, z = vae(images)
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta=beta)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store losses
            epoch_losses['total'].append(loss.item())
            epoch_losses['recon'].append(recon_loss.item())
            epoch_losses['kl'].append(kl_loss.item())
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                      f"KL: {kl_loss.item():.4f}")
        
        # Average losses
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            losses[k].append(v)
        
        print(f"Epoch {epoch+1} completed: Avg Loss = {avg_losses['total']:.4f}")
    
    # Save model
    torch.save(vae.state_dict(), f'data/vae_{dataset_name}.pth')
    print(f"Model saved to data/vae_{dataset_name}.pth")
    
    return vae, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--latent_dim', type=int, default=32)
    
    args = parser.parse_args()
    
    vae, losses = train_vae(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        latent_dim=args.latent_dim
    )

