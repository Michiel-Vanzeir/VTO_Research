import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # Input: (Batch, 1, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        # After two convolutions with stride 2: 32x32 -> 8x8 
        self.fc1 = nn.Linear(128 * 8 * 8, 16)
        self.fc_mean = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x), 0.2)
        
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 8 * 8 * 64)
        
        self.conv_t1 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_t3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 8, 8) # Reshape to (Batch, Channels, H, W)
        x = F.relu(self.bn1(self.conv_t1(x)))
        x = F.relu(self.bn2(self.conv_t2(x)))
        x = torch.tanh(self.conv_t3(x)) 
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var, z

    def generate(self, z):
        reconstruction = self.decoder(z)
        return reconstruction
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2)        
        ) # l-th layer for feature extraction
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1) # Classifies the images as fake/real
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        classification = self.classifier(features)
        return classification, features