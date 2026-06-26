import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb
import yaml
import torch
import torch.optim as optim

from shared.dataloaders import get_fashion_mnist_loader
from models import DDPM
from engine import DDPMTrainer

def main():
    with open("03_DDPM/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project='VTO-Research',
        entity='michiel-vanzeir-kuleuven',
        name='DDPM-FashionMNIST',
        config=config
    )
    config = wandb.config

    train_loader, _ = get_fashion_mnist_loader(batch_size=config.batch_size)

    ddpm = DDPM(
        num_diffusion_timesteps=config.get('timesteps', 1000), 
        device=device
    )
    optimizer = optim.Adam(ddpm.model.parameters(), lr=float(config.get('learning_rate', 2e-4)))

    trainer = DDPMTrainer(
        model=ddpm, 
        dataloader=train_loader, 
        optimizer=optimizer, 
        device=device, 
        config=config
    )

    # Start the training loop
    trainer.fit()

    # Save the trained UNet weights
    save_path = f"weights/DDPM/ddpm_e{config.epochs}_id{run.id}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(ddpm.model.state_dict(), save_path)
    wandb.save(save_path)
    print("All weights are saved locally and in the cloud!")

    wandb.finish()


if __name__ == "__main__":
    main()