"""
Visualization utilities for world models.

Tools for visualizing latent space, reconstructions, and predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_reconstruction(vae, data_loader, device, num_samples=8):
    """
    Visualize VAE reconstructions.
    
    Args:
        vae: Trained VAE model
        data_loader: Data loader with images
        device: Device to run on
        num_samples: Number of samples to visualize
    """
    vae.eval()
    
    with torch.no_grad():
        # Get a batch
        images, _ = next(iter(data_loader))
        images = images.to(device)
        
        # Reconstruct
        recon_images, mu, logvar, z = vae(images)
        
        # Take first num_samples
        images = images[:num_samples].cpu()
        recon_images = recon_images[:num_samples].cpu()
        
        # Create figure
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        
        for i in range(num_samples):
            # Original
            if images[i].shape[0] == 1:  # Grayscale
                axes[0, i].imshow(images[i].squeeze(), cmap='gray')
            else:  # RGB
                axes[0, i].imshow(images[i].permute(1, 2, 0))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstruction
            if recon_images[i].shape[0] == 1:  # Grayscale
                axes[1, i].imshow(recon_images[i].squeeze(), cmap='gray')
            else:  # RGB
                axes[1, i].imshow(recon_images[i].permute(1, 2, 0))
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig


def visualize_latent_space(vae, data_loader, device, num_samples=1000):
    """
    Visualize latent space using t-SNE or PCA (2D projection).
    
    Args:
        vae: Trained VAE model
        data_loader: Data loader
        device: Device to run on
        num_samples: Number of samples to visualize
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    vae.eval()
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(data_loader):
            images = images.to(device)
            mu, logvar = vae.encode(images)
            z = vae.reparameterize(mu, logvar)
            
            latents.append(z.cpu().numpy())
            labels.append(batch_labels.numpy())
            
            if len(np.concatenate(latents)) >= num_samples:
                break
    
    latents = np.concatenate(latents)[:num_samples]
    labels = np.concatenate(labels)[:num_samples]
    
    # Project to 2D
    if latents.shape[1] > 2:
        # Use PCA first for dimensionality reduction, then t-SNE
        pca = PCA(n_components=50)
        latents_pca = pca.fit_transform(latents)
        
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(latents_pca)
    else:
        latents_2d = latents
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (2D projection)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    return plt.gcf()


def visualize_predictions(rnn, vae, initial_obs, actions, device, num_steps=10):
    """
    Visualize RNN predictions in observation space.
    
    Args:
        rnn: Trained RNN model
        vae: Trained VAE model
        initial_obs: Initial observation [channels, height, width]
        actions: Sequence of actions [num_steps, action_dim]
        device: Device to run on
        num_steps: Number of prediction steps
    """
    rnn.eval()
    vae.eval()
    
    with torch.no_grad():
        # Encode initial observation
        initial_obs_tensor = initial_obs.unsqueeze(0).to(device)
        mu, logvar = vae.encode(initial_obs_tensor)
        z = vae.reparameterize(mu, logvar)
        
        # Predict future latent states
        predicted_latents = [z]
        hidden = None
        
        actions_tensor = torch.FloatTensor(actions).to(device)
        
        for i in range(num_steps):
            action = actions_tensor[i:i+1]
            z_next, hidden = rnn.predict(z, action, hidden)
            predicted_latents.append(z_next)
            z = z_next
        
        # Decode predicted latents
        predicted_obs = []
        for z_pred in predicted_latents:
            obs_pred = vae.decode(z_pred)
            predicted_obs.append(obs_pred.cpu())
        
        # Visualize
        fig, axes = plt.subplots(1, len(predicted_obs), figsize=(2*len(predicted_obs), 2))
        
        for i, obs in enumerate(predicted_obs):
            if obs.shape[1] == 1:  # Grayscale
                axes[i].imshow(obs.squeeze().permute(1, 2, 0).squeeze(), cmap='gray')
            else:  # RGB
                axes[i].imshow(obs.squeeze().permute(1, 2, 0))
            axes[i].set_title(f'Step {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


def plot_training_curves(losses, title='Training Curves'):
    """
    Plot training loss curves.
    
    Args:
        losses: Dictionary of loss arrays {name: [values]}
        title: Plot title
    """
    fig, axes = plt.subplots(1, len(losses), figsize=(5*len(losses), 4))
    
    if len(losses) == 1:
        axes = [axes]
    
    for idx, (name, values) in enumerate(losses.items()):
        axes[idx].plot(values)
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Loss')
        axes[idx].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

