import torch
from unet import UNet

class DDPM:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device
        self.initialize()

        self.base_ch = 64
        self.base_ch_mult = (1, 2, 4, 4)
        self.apply_attention = (False, True, True, False)
        self.dropout_rate = 0.1
        self.time_emb_mult = 4

        self.model = UNet(
            input_channels          = 1,
            output_channels         = 1,
            base_channels           = self.base_ch,
            base_channels_multiples = self.base_ch_mult,
            apply_attention         = self.apply_attention,
            dropout_rate            = self.dropout_rate,
            time_multiple           = self.time_emb_mult,
        )
        self.model.to(self.device)
        
    def initialize(self):
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
         
        self.sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
        
    def get_betas(self):
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )

    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        eps = torch.randn_like(x0) 
        sqrt_alpha_cum = self.sqrt_alpha_cumulative[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cumulative[timesteps].view(-1, 1, 1, 1)   
        sample = sqrt_alpha_cum * x0 + sqrt_one_minus_alpha_cum * eps
  
        return sample, eps