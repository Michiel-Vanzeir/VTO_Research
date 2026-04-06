import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import torch.nn.functional as F

class VAEGANTrainer:
    def __init__(self, vae_model, discriminator, train_loader, test_loader, optimizers, device, config):
        self.vae_model = vae_model
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.encoder_optimizer = optimizers['encoder']
        self.decoder_optimizer = optimizers['decoder']
        self.discriminator_optimizer = optimizers['discriminator']
        
        # Hyperparameters from config
        self.beta = config.get('beta', 1.0)
        self.gamma = config.get('gamma', 15.0)
        self.epochs = config.get('epochs', 10)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.latent_dim = config.get('latent_dim', 25)

        self.criterion = nn.BCEWithLogitsLoss()

    def VAE_warmup_epoch(self, epoch):
        self.vae_model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        # TQDM progress bar for real-time feedback
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Warmup epoch {epoch}", leave=True)
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, log_var, _ = self.vae_model.forward(data)

            # Loss calculation
            recon_loss = nn.functional.mse_loss(recon_batch, data)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + (self.beta * kl_loss)

            # Backward pass
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Update losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            # TQDM update 
            running_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': running_avg_loss})

        # Average epoch loss
        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon / len(self.train_loader)
        avg_kl = total_kl / len(self.train_loader)

        return avg_loss, avg_recon, avg_kl

    def train_epoch(self, epoch):
        self.vae_model.train()
        total_loss = 0
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_discriminator_loss = 0
        total_kl = 0
        total_feature_matching = 0
        total_gan = 0

        # TQDM progress bar for real-time feedback
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", leave=True)
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            batch_size = data.size(0)
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            # --- FORWARD PASSES ---
            recon_batch, mu, logvar, z = self.vae_model.forward(data)
            z_samples = torch.randn(batch_size, self.latent_dim).to(self.device)
            recon_samples = self.vae_model.generate(z_samples)

            # --- UPDATE DISCRIMINATOR ---
            self.discriminator_optimizer.zero_grad()

            p_real, f_real = self.discriminator(data)
            p_fake, f_fake = self.discriminator(recon_batch.detach())
            p_sample, _ = self.discriminator(recon_samples.detach())

            loss_d_real = self.criterion(p_real, real_labels)
            loss_d_fake = self.criterion(p_fake, fake_labels)
            loss_d_sample = self.criterion(p_sample, fake_labels)

            loss_disc = loss_d_real + loss_d_fake + loss_d_sample

            loss_disc.backward()
            self.discriminator_optimizer.step()

            # --- UPDATE ENCODER & DECODER ---
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            p_fake, f_fake = self.discriminator(recon_batch)    
            p_sample, _ = self.discriminator(recon_samples)

            loss_d_fake = self.criterion(p_fake, real_labels)
            loss_d_sample = self.criterion(p_sample, real_labels)

            loss_disc = loss_d_fake + loss_d_sample
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            fm_loss = F.mse_loss(f_fake, f_real.detach())

            pixel_mse = F.mse_loss(recon_batch, data)       

            # Calculate gradients
            loss_enc = kl_loss + fm_loss + 10.0*pixel_mse
            loss_enc.backward(retain_graph=True)

            loss_dec = (self.gamma * fm_loss) + loss_disc + 10.0*pixel_mse
            loss_dec.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Update losses
            total_loss += loss_enc.item() + loss_dec.item() + loss_disc.item()
            total_encoder_loss += loss_enc.item()
            total_decoder_loss += loss_dec.item()
            total_discriminator_loss += loss_disc.item()
            total_kl += kl_loss.item()
            total_feature_matching += fm_loss.item()
            total_gan += loss_disc.item()
            
            # TQDM update 
            running_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': running_avg_loss})

        # Average epoch loss
        batch_size = len(self.train_loader)
        avg_loss = (total_encoder_loss + total_decoder_loss + total_discriminator_loss) / batch_size
        avg_kl = total_kl / batch_size
        avg_fm = total_feature_matching / batch_size
        avg_gan = total_gan / batch_size
        avg_enc = total_encoder_loss / batch_size
        avg_dec = total_decoder_loss / batch_size
        avg_disc = total_discriminator_loss / batch_size

        return avg_loss, avg_enc, avg_dec, avg_disc, avg_kl, avg_fm, avg_gan

    def fit(self):
        """
        The main training loop over all epochs.
        """
        for epoch in range(1, self.warmup_epochs + 1):
            self.VAE_warmup_epoch(epoch)

        for epoch in range(1, self.epochs + 1):
            avg_loss, avg_enc, avg_dec, avg_disc, avg_kl, avg_fm, avg_gan = self.train_epoch(epoch)

            # Log to weights & biases
            wandb.log({
                "epoch": epoch,
                "total_loss": avg_loss,
                "encoder_loss": avg_enc,
                "decoder_loss": avg_dec,
                "discriminator_loss": avg_disc,
                "kl_divergence": avg_kl,
                "feature_matching_loss": avg_fm,
                "gan_loss": avg_gan,
                "beta": self.beta
            })

            print(f"==> Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")