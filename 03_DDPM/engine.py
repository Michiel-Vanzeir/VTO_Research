import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
import wandb
from tqdm import tqdm

class DDPMTrainer:
    def __init__(self, model, dataloader, optimizer, device, config):
        self.ddpm = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.MSELoss()
        
        # Use device-agnostic GradScaler
        self.scaler = torch.amp.GradScaler()
        
        self.epochs = config.get('epochs', 800)
        self.timesteps = config.get('timesteps', 1000)

    def train_epoch(self, epoch=800):
        # Move metric to device to avoid CPU/GPU sync issues
        loss_record = MeanMetric().to(self.device)
        self.ddpm.model.train()
    
        with tqdm(total=len(self.dataloader), dynamic_ncols=True) as tq:
            tq.set_description(f"Train :: Epoch: {epoch}/{self.epochs}")
            
            for x0s, _ in self.dataloader: 
                tq.update(1)
                
                # Move inputs to device
                x0s = x0s.to(self.device)
                
                ts = torch.randint(low=1, high=self.timesteps, size=(x0s.shape[0],), device=self.device)
                xts, gt_noise = self.ddpm.forward_diffusion(x0s, ts) 
    
                # Universal autocast requires device_type as string (e.g. 'cuda')
                with torch.amp.autocast(device_type=self.device.type):
                    pred_noise = self.ddpm.model(xts, ts)
                    loss = self.loss_fn(gt_noise, pred_noise) 
    
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
    
                self.scaler.step(self.optimizer)
                self.scaler.update()
    
                loss_value = loss.detach()
                loss_record.update(loss_value)
    
                tq.set_postfix_str(s=f"Loss: {loss_value.item():.4f}")
    
            mean_loss = loss_record.compute().item()
            tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
        
        return mean_loss

    @torch.no_grad()
    def generate_images(self, num_images=16, img_channels=1, img_size=32):
        x = torch.randn((num_images, img_channels, img_size, img_size), device=self.device)
        self.ddpm.model.eval()

        for time_step in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="Generating Images", leave=False):
            ts = torch.ones(num_images, dtype=torch.long, device=self.device) * time_step
            z = torch.randn_like(x) if time_step > 0 else torch.zeros_like(x)

            with torch.amp.autocast(device_type=self.device.type):
                predicted_noise = self.ddpm.model(x, ts)

            beta_t = self.ddpm.beta[ts].view(-1, 1, 1, 1)
            one_by_sqrt_alpha_t = self.ddpm.one_by_sqrt_alpha[ts].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cum_t = self.ddpm.sqrt_one_minus_alpha_cumulative[ts].view(-1, 1, 1, 1)

            x = (
                one_by_sqrt_alpha_t
                * (x - (beta_t / sqrt_one_minus_alpha_cum_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
            )

        self.ddpm.model.train() 
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        return x

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            avg_loss = self.train_epoch(epoch)

            log_dict = {
                "epoch": epoch,
                "total_loss": avg_loss,
            }

            if epoch % 20 == 0 or epoch == self.epochs:
                samples = self.generate_images(num_images=16, img_channels=1, img_size=32)
                grid = make_grid(samples, nrow=4, padding=2, pad_value=1.0)
                log_dict["generated_samples"] = wandb.Image(grid, caption=f"Epoch {epoch}")

            wandb.log(log_dict)
            print(f"==> Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")