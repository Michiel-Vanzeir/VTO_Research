import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb
import yaml
import torch

from models import VAE, Discriminator
from engine import VAEGANTrainer
from shared.dataloaders import get_fashion_mnist_loader


def main():
    # Load config & select device
    with open("02_VAE-GAN/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Weights & biases logging setup
    run = wandb.init(
        project='VTO-Research',
        entity='michiel-vanzeir-kuleuven',
        name='VAE-GAN-FashionMNIST',
        config = config
    )
    config = wandb.config

    # Data & vae model setup
    train_loader, test_loader = get_fashion_mnist_loader(batch_size=config.batch_size)
    vae = VAE(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizers = {
        'encoder': torch.optim.Adam(vae.encoder.parameters(), lr=config.learning_rates['encoder_lr']),
        'decoder': torch.optim.Adam(vae.decoder.parameters(), lr=config.learning_rates['decoder_lr']),
        'discriminator': torch.optim.Adam(discriminator.parameters(), lr=config.learning_rates['discriminator_lr'])
    }

    # Training 
    trainer = VAEGANTrainer(vae, discriminator, train_loader, test_loader, optimizers, device, config)
    trainer.fit()

    # Saving the weights
    save_path = f"weights/VAE_GAN/vae-gan_z{config.latent_dim}_e{config.epochs}_id{run.id}.pth"
    torch.save(vae.state_dict(), save_path)
    wandb.save(save_path)
    print("All weights are saved locally and in the cloud!")

    wandb.finish()


if __name__ == "__main__":
    main()