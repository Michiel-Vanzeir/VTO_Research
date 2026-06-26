import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class DDPMVisualizer:
    def __init__(self, model, dataloader, device, img_size=32):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.img_size = img_size

    @torch.no_grad()
    def inverse_transform(self, tensors):
      """Convert tensors from [-1., 1.] to [0., 255.]"""
      return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0


    @torch.no_grad()
    def visualize_forward_process(self):
        x0s, _ = next(iter(self.dataloader))
        noisy_images = []
        specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]
        
        for timestep in specific_timesteps:
            timestep = torch.as_tensor(timestep, dtype=torch.long)
        
            xts, _ = self.model.forward_diffusion(x0s, timestep)
            xts    = self.inverse_transform(xts) / 255.0
            xts    = make_grid(xts, nrow=1, padding=1)
            
            noisy_images.append(xts)
        
        _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')
 
        for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
            ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
            ax[i].set_title(f"t={timestep}", fontsize=8)
            ax[i].axis("off")
            ax[i].grid(False)
        
        plt.suptitle("Forward Diffusion Process", y=0.9)
        plt.axis("off")
        plt.show()