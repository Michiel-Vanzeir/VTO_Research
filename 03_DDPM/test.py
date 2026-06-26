import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import matplotlib.pyplot as plt

from models import DDPM
from visualization import DDPMVisualizer
from shared.dataloaders import get_fashion_mnist_loader


def main():
    # Load config & select device
    with open("03_DDPM/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader & VAE model setup
    train_loader, _ = get_fashion_mnist_loader(batch_size=6)
    ddpm = DDPM(num_diffusion_timesteps=config['timesteps'])
    

    # VAE visualization setup
    visualizer = DDPMVisualizer(ddpm, train_loader, device)

    visualizer.visualize_forward_process()


if __name__ == "__main__":
    main()