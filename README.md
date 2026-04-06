# Virtual Try-On (VTO) Research

This repository contains the code, experiments, and research logs for an Artificial Intelligence project focused on Virtual Try-On (VTO) applications. This research is conducted as part of the Honours Programme at the Faculty of Engineering Science, KU Leuven.

## Project Objective
The primary goal of this project is to explore, evaluate, and implement generative AI architectures tailored for Virtual Try-On. The research progression moves from foundational latent variable models to state-of-the-art diffusion and attention-based architectures.

## Current Progress & Implementations

### 1. Variational Autoencoder (VAE)
- **Status:** Completed
- **Dataset:** FashionMNIST
- **Details:** Built entirely from scratch. Focuses on latent space representation, probabilistic mapping, and the mathematical foundations of the Evidence Lower Bound (ELBO).

### 2. VAEGAN
- **Status:** Completed
- **Dataset:** FashionMNIST
- **Details:** Custom implementation from scratch. Combines the structured latent space of a VAE with the adversarial training of a Generative Adversarial Network (GAN) to address the typical blurriness of standard VAEs and improve the sharpness of generated clothing items.

### 3. Advanced Generative Models (Theoretical Foundation)
Extensive theoretical groundwork has been completed for the upcoming implementation phases, covering:
- **Diffusion Models:** Forward and reverse processes in Denoising Diffusion Probabilistic Models (DDPM) and accelerated sampling via Denoising Diffusion Implicit Models (DDIM).
- **Attention Mechanisms:** Self-attention, cross-attention, and Vision Transformers (ViT), which are crucial for accurately conditioning clothing generation on human poses and features.

## Repository Structure
*(Note: Please adapt this to match your actual folder hierarchy)*

```text
├── data/                   # Scripts for downloading and formatting FashionMNIST
├── models/
│   ├── vae.py              # Custom VAE architecture from scratch
│   ├── vaegan.py           # Custom VAEGAN architecture from scratch
├── notebooks/              # Jupyter notebooks for exploratory data analysis and visualising latent spaces
├── utils/                  # Helper functions (loss calculations, logging, plotting)
├── train_vae.py            # Training script for the VAE
├── train_vaegan.py         # Training script for the VAEGAN
└── README.md