import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] 
        h = h + time_emb
        h = self.bn2(self.relu(self.transform(h)))
        h = self.dropout(h)
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x_reshaped = x.view(batch_size, channels, h * w).swapaxes(1, 2)
        x_ln = self.ln(x_reshaped)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_reshaped
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(batch_size, channels, h, w)

class UNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        base_channels=64,
        base_channels_multiples=(1, 2, 4, 4),
        apply_attention=(False, True, True, False),
        dropout_rate=0.1,
        time_multiple=4,
    ):
        super().__init__()
        
        time_emb_dim = base_channels * time_multiple
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_ch = base_channels
        
        for i, mult in enumerate(base_channels_multiples):
            out_ch = base_channels * mult
            for _ in range(2):
                self.downs.append(nn.ModuleList([
                    Block(now_ch, out_ch, time_emb_dim, dropout_rate),
                    SelfAttention(out_ch) if apply_attention[i] else nn.Identity()
                ]))
                now_ch = out_ch
                channels.append(now_ch)
            
            if i != len(base_channels_multiples) - 1:
                self.downs.append(nn.Conv2d(now_ch, now_ch, 4, 2, 1))
                channels.append(now_ch)

        # Bottleneck
        self.mid_block1 = Block(now_ch, now_ch, time_emb_dim, dropout_rate)
        self.mid_attn = SelfAttention(now_ch)
        self.mid_block2 = Block(now_ch, now_ch, time_emb_dim, dropout_rate)

        # Decoder
        self.ups = nn.ModuleList()
        reversed_mults = list(reversed(base_channels_multiples))
        reversed_attn = list(reversed(apply_attention))
        
        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            # Changed to 3 to properly pop all encoder elements (2 blocks + 1 downsample)
            for _ in range(3): 
                self.ups.append(nn.ModuleList([
                    Block(now_ch + channels.pop(), out_ch, time_emb_dim, dropout_rate),
                    SelfAttention(out_ch) if reversed_attn[i] else nn.Identity()
                ]))
                now_ch = out_ch
                    
            if i != len(reversed_mults) - 1:
                self.ups.append(nn.ConvTranspose2d(now_ch, now_ch, 4, 2, 1))

        self.out_conv = nn.Conv2d(now_ch, output_channels, 3, padding=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.init_conv(x)
        
        skips = [x]
        for layer in self.downs:
            if isinstance(layer, nn.ModuleList):
                x = layer[0](x, t)
                x = layer[1](x)
            else:
                x = layer(x)
            skips.append(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for layer in self.ups:
            if isinstance(layer, nn.ModuleList):
                skip = skips.pop()
                
                # Autopad to fix FashionMNIST down/upsampling halving mismatches
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    
                x = torch.cat([x, skip], dim=1)
                x = layer[0](x, t)
                x = layer[1](x)
            else:
                x = layer(x)

        return self.out_conv(x)