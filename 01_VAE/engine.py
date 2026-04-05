import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

class VAETrainer:
    def __init__(self, vae_model, dataloader, optimizer, device, config):
        self.vae_model = vae_model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        
        # Hyperparameters from config
        self.beta = config.get('beta', 1.0)
        self.epochs = config.get('epochs', 10)
        self.latent_dim = config.get('latent_dim', 25)

    def loss_function(self, recon_x, x, mu, log_var):
        """
        Calculates the VAE loss: Reconstruction (BCE) + KL Divergence.
        """
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss, kl_loss

    def train_epoch(self, epoch):
        self.vae_model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        # TQDM progress bar for real-time feedback
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}", leave=True)
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, log_var, _ = self.vae_model.forward(data)

            # Loss calculation
            recon_loss, kl_loss = self.loss_function(recon_batch, data, mu, log_var)
            loss = recon_loss + (self.beta * kl_loss)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            # TQDM update 
            pbar.set_postfix({'loss': loss.item() / len(data)})

        # Average epoch loss
        avg_loss = total_loss / len(self.dataloader.dataset)
        avg_recon = total_recon / len(self.dataloader.dataset)
        avg_kl = total_kl / len(self.dataloader.dataset)

        return avg_loss, avg_recon, avg_kl

    def fit(self):
        """
        The main training loop over all epochs.
        """
        for epoch in range(1, self.epochs + 1):
            avg_loss, avg_recon, avg_kl = self.train_epoch(epoch)

            # Log to weights & biases
            wandb.log({
                "epoch": epoch,
                "total_loss": avg_loss,
                "reconstruction_loss": avg_recon,
                "kl_divergence": avg_kl,
                "beta": self.beta
            })

            print(f"==> Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")