"""
---
title: Latent Diffusion Models
summary: >
 Annotated PyTorch implementation/tutorial of latent diffusion models from paper
 High-Resolution Image Synthesis with Latent Diffusion Models
---

# Latent Diffusion Models

Latent diffusion models use an auto-encoder to map between image space and
latent space. The diffusion model works on the latent space, which makes it
a lot easier to train.
It is based on paper
[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent
space of the pre-trained auto-encoder.

For a simpler diffusion implementation refer to our [DDPM implementation](../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import List

import torch
import torch.nn as nn
import lightning as L

from .model.autoencoder import Autoencoder
from .model.clip_embedder import CLIPTextEmbedder
from .model.unet import UNetModel


class DiffusionWrapper(nn.Module):
    """
    *This is an empty wrapper class around the [U-Net](model/unet.html).
    We keep this to have the same model structure as
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
    so that we do not have to map the checkpoint weights explicitly*.
    """

    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    * [CLIP embeddings generator](model/clip_embedder.html)
    """
    model: DiffusionWrapper
    first_stage_model: Autoencoder
    cond_stage_model: CLIPTextEmbedder

    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 ):
        r"""
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the [AutoEncoder](model/autoencoder.html)
        :param clip_embedder: is the [CLIP embeddings generator](model/clip_embedder.html)
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.model = DiffusionWrapper(unet_model)
        # Auto-encoder and scaling factor
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        # [CLIP embeddings generator](model/clip_embedder.html)
        self.cond_stage_model = clip_embedder

        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.model.parameters())).device

    def get_text_conditioning(self, prompts: List[str]):
        """
        ### Get [CLIP embeddings](model/clip_embedder.html) for a list of text prompts
        """
        return self.cond_stage_model(prompts)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        r"""
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, context)
    

class MyLatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    * [CLIP embeddings generator](model/clip_embedder.html)
    """
    model: DiffusionWrapper
    first_stage_model: Autoencoder

    def __init__(self,
                 unet_model: UNetModel,
                 latent_scaling_factor: float,
                 latent_dim: int,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 channels = 3
                 ):
        r"""
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the [AutoEncoder](model/autoencoder.html)
        :param clip_embedder: is the [CLIP embeddings generator](model/clip_embedder.html)
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.model = DiffusionWrapper(unet_model)
        # Auto-encoder and scaling factor
        self.latent_scaling_factor = latent_scaling_factor

        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)


        self.latent_dim = latent_dim
        self.c = channels
        self.h, self.w  = int((self.latent_dim // channels)**0.5), int((self.latent_dim // channels)**0.5)



    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.model.parameters())).device
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_bar.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_bar.to(original.device)[t].reshape(batch_size)

        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sequential_diffusion(self, x, t1, t2, noise = None):

        tricky_idxes = t1 == 0
        t1[tricky_idxes] = 1 # A CORRIGER
        # if t1 == 0:
        #     shrink_sqrt_cum_prod = self.sqrt_alpha_bar.to(x.device)[t2].reshape(x.shape[0])
        # else:    
        shrink_sqrt_cum_prod = self.sqrt_alpha_bar.to(x.device)[t2].reshape(x.shape[0]) / self.sqrt_alpha_bar.to(x.device)[t1 - 1].reshape(x.shape[0])
        
        
        shrink_sqrt_one_minus_cum_prod = torch.sqrt( 1 - shrink_sqrt_cum_prod**2 )


        if noise is None:
            noise = torch.randn_like(x)

        for _ in range(len(x.shape) - 1):
            shrink_sqrt_cum_prod = shrink_sqrt_cum_prod.unsqueeze(-1)
            shrink_sqrt_one_minus_cum_prod = shrink_sqrt_one_minus_cum_prod.unsqueeze(-1)

        return (shrink_sqrt_cum_prod * x + shrink_sqrt_one_minus_cum_prod * noise)


    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None, **kwargs):
        r"""
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, context)
    

class LitLDM(L.LightningModule):
    def __init__(self, ldm, vae, channels = 3, lr = 1e-3):
        super().__init__()
        self.ldm = ldm
        self.lr = lr
        self.vae = vae
        self.vae.eval()

        self.lat_dim = vae.latent_dim

        self.c = channels
        self.h, self.w  = int((self.lat_dim // channels)**0.5), int((self.lat_dim // channels)**0.5)

        for param in self.vae.parameters():
            param.requires_grad = False
        self.model_config = None

    def training_step(self, batch, batch_idx, **kwargs):
        # training_step defines the train loop.
        x = batch
        batch_size = x.shape[0]
        z = self.vae.encoder(x).embedding.reshape(-1, self.c, self.h, self.w)
        noise = torch.randn_like(z)

        t = torch.randint(0, self.ldm.n_steps, (z.shape[0],)).to(z.device)

        noisy_z = self.ldm.add_noise(z, noise, t).float()

        noise_pred = self.ldm(noisy_z, t).reshape(batch_size, self.c * self.h * self.w)
        noise = noise.reshape(batch_size, self.c * self.h * self.w)

        #z_pred = (noisy_z - (1- self.ldm.alpha_bar[t].reshape(batch_size, 1, 1, 1)) ** 0.5 * noise_pred) / (self.ldm.alpha_bar[t].reshape(batch_size, 1, 1, 1) ** 0.5)

        loss = ((noise_pred - noise)**2).sum(axis=1).mean()

        self.log("train_loss", loss, prog_bar=True)


        return loss
    
    def validation_step(self, batch, batch_idx, **kwargs):
        x = batch
        batch_size = x.shape[0]
        z = self.vae.encoder(x).embedding.reshape(-1, self.c, self.h, self.w)
        noise = torch.randn_like(z)

        t = torch.randint(0, self.ldm.n_steps, (batch_size,)).to(z.device)
        noisy_z = self.ldm.add_noise(z, noise, t).float()

        noise_pred = self.ldm(noisy_z, t).reshape(batch_size,self.c * self.h * self.w)
        noise = noise.reshape(batch_size, self.c * self.h * self.w)

        val_loss = ((noise_pred - noise)**2).sum(axis=1).mean()

        self.log("val_loss", val_loss, prog_bar = True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ldm.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.8)

        return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor':'train_loss'}
