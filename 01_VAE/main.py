import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb
import yaml
import torch

from models import VAE
from engine import VAETrainer
from shared.dataloaders import get_fashion_mnist_loader


def main():
    # Load config & select device
    with open("01_VAE/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Weights & biases logging setup
    run = wandb.init(
        project='VTO-Research',
        entity='michiel-vanzeir-kuleuven',
        name='VAE-FashionMNIST',
        config = config
    )
    config = wandb.config

    # Data & vae model setup
    dataloader = get_fashion_mnist_loader(batch_size=config.batch_size)
    vae = VAE(config.latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=config.learning_rate)

    # Training 
    trainer = VAETrainer(vae, dataloader, optimizer, device, config)
    trainer.fit()

    # Saving the weights
    save_path = f"weights/VAE/vae_z{config.latent_dim}_e{config.epochs}_lr{config.learning_rate}_id{run.id}.pth"
    torch.save(vae.state_dict(), save_path)
    wandb.save(save_path)
    print("All weights are saved locally and in the cloud!")

    wandb.finish()


if __name__ == "__main__":
    main()