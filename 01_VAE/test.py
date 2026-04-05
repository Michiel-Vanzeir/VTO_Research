import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import matplotlib.pyplot as plt

from models import VAE
from visualization import VAEVisualizer
from shared.dataloaders import get_fashion_mnist_loader


def main():
    # Load config & select device
    with open("01_VAE/config.yaml", "r") as f:
        config = yaml.safe_loadf(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader & VAE model setup
    dataloader = get_fashion_mnist_loader(batch_size=config['batch_size'])
    vae = VAE(config['latent_dim']).to(device)
    model_name = "vae_z64_e50_idqt8vq98z.pth"
    model_path = os.path.join("weights", "VAE", model_name)

    if not os.path.exists(model_path):
        print(f"Model can't be found at: {model_path}")
        return

    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    print("Model loaded succesfully!")

    # VAE visualization setup
    visualizer = VAEVisualizer(vae, dataloader, device)

    with torch.no_grad():
        visualizer.compare_reconstruction(110)


if __name__ == "__main__":
    main()