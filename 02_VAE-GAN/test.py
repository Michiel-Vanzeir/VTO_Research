import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import matplotlib.pyplot as plt

from models import VAE
from visualization import VAEGANVisualizer
from shared.dataloaders import get_fashion_mnist_loader


def main():
    # Load config & select device
    with open("02_VAE-GAN/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader & VAE model setup
    train_loader, test_loader = get_fashion_mnist_loader(batch_size=config['batch_size'])
    vae = VAE(config['latent_dim']).to(device)
    model_name = "vae-gan_z40_e100_idvldmxn8a.pth"
    model_path = os.path.join("weights", "VAE_GAN", model_name)

    if not os.path.exists(model_path):
        print(f"Model can't be found at: {model_path}")
        return

    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    print("Model loaded succesfully!")

    # VAE visualization setup
    visualizer = VAEGANVisualizer(vae, train_loader, device)

    with torch.no_grad():
        visualizer.compare_reconstruction(10)


if __name__ == "__main__":
    main()