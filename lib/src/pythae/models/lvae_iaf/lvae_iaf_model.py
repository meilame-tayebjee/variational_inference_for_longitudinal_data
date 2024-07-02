import os
from typing import Optional

import sys
sys.path.append("......")
sys.path.append(".....")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..normalizing_flows import IAF, IAFConfig
from ..vae import VAE
from .lvae_iaf_config import LVAE_IAF_Config
from diffusion.stable_diffusion.latent_diffusion import MyLatentDiffusion
from diffusion.stable_diffusion.sampler.ddim import DDIMSampler


class LVAE_IAF(VAE):
    """Longitudinal Variational Auto Encoder with Inverse Autoregressive Flows
    (:class:`~pythae.models.normalizing_flows.IAF`).

    Args:
        model_config(VAE_IAF_Config): The Variational Autoencoder configuration seting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: LVAE_IAF_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "LVAE_IAF"

        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim
        self.beta = model_config.beta

        self.flows = nn.ModuleList()

        iaf_config = IAFConfig(
            input_dim=(model_config.latent_dim,),
            n_blocks=model_config.n_made_blocks,
            n_hidden_in_made=model_config.n_hidden_in_made,
            hidden_size=model_config.hidden_size,
            include_batch_norm=False,
            context_dim=model_config.context_dim
        )

        for i in range(self.n_obs - 1):
            self.flows.append(IAF(iaf_config))

        self.linear_scheduling = self.model_config.linear_scheduling_steps

        self.prior = model_config.prior
        self.posterior = model_config.posterior

        if self.prior == "vamp":
            linear_layer = nn.Linear(
                model_config.vamp_number_components, int(np.prod(model_config.input_dim))
            )
            self.pseudo_inputs = nn.Sequential(linear_layer, nn.Hardtanh(0.0, 1.0))

            self.idle_input = torch.eye(
                model_config.vamp_number_components, requires_grad=False
            ).to(self.device)

            self.vamp_number_components = model_config.vamp_number_components

        if self.posterior == "iaf":
            self.posterior_iaf_config = IAFConfig(
                input_dim=(model_config.latent_dim,),
                n_blocks=3,
                n_hidden_in_made=2,
                hidden_size=model_config.hidden_size,
                context_dim=model_config.context_dim,
                include_batch_norm=False,
            )

            self.posterior_iaf_flow = IAF(self.posterior_iaf_config)

        

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE NF model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        seq_mask = inputs['seq_mask']
        pix_mask = inputs['pix_mask']
        epoch = kwargs.pop("epoch", 100)
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        if epoch < self.warmup:
            encoder_output = self.encoder(x)#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)

            z0 = z

            log_abs_det_jac_posterior = 0
            if self.posterior == 'iaf':
                z0 = z

                if self.posterior_iaf_config.context_dim is not None:
                    try:
                        h = encoder_output.context

                    except AttributeError as e:
                        raise AttributeError(
                            "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                            "something different from None please ensure that the encoder actually outputs "
                            f"the context vector 'h'. Exception caught: {e}."
                        )

                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                else:
                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac

            z_seq = z

            #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
            #z = torch.cat((t.unsqueeze(-1), z.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

            recon_x = self.decoder(z_seq)["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

            loss, recon_loss, kld = self.vae_loss_function(
                recon_x=recon_x,
                x=x.reshape((x.shape[0]*self.n_obs,) + x.shape[2:]),
                mu=mu,
                log_var=log_var,
                z0=z0, 
                zk=z,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask)

        else:

            # if missing data pick randomly index of non missing data
            seq_mask = seq_mask.reshape(-1, self.n_obs)
            if seq_mask[0].sum() < self.n_obs:
                probs = seq_mask[0].cpu().numpy()
                probs /= probs.sum()
                vi_index = np.random.choice(np.arange(self.n_obs), p=probs.reshape(-1))

            else:
                vi_index = np.random.randint(self.n_obs)
            
            encoder_output = self.encoder(x[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            h = None#encoder_output.context

            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)

            z_0_vi_index = z

            log_abs_det_jac_posterior = 0
            if self.posterior == 'iaf':

                if self.posterior_iaf_config.context_dim is not None:
                    try:
                        h = encoder_output.context

                    except AttributeError as e:
                        raise AttributeError(
                            "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                            "something different from None please ensure that the encoder actually outputs "
                            f"the context vector 'h'. Exception caught: {e}."
                        )

                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                else:
                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac

            z_vi_index = z

            ## propagate in past
            z_seq = []
            z_rev = z_vi_index
            log_abs_det_jac = 0
            for i in range(vi_index - 1, -1, -1):
                #print("past", i)
                flow_output = self.flows[i](z_rev)
                z_rev = flow_output.out
                log_abs_det_jac += flow_output.log_abs_det_jac
                z_seq.append(z_rev)
##
            z_seq.reverse()
#
            z_seq.append(z_vi_index)

            #proapagate in future
            z_for = z_vi_index
            for i in range(vi_index, self.n_obs - 1):
                #print("future", i)
                flow_output = self.flows[i].inverse(z_for)
                z_for = flow_output.out
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1)

            #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
            #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

            z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.latent_dim)

            loss, recon_loss, kld = self.loss_function(
                recon_x=recon_x,
                x=x,
                mu=mu,
                log_var=log_var,
                z_0_vi_index=z_0_vi_index,
                z_seq=z_seq,
                z_vi_index=z_vi_index,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                log_abs_det_jac=log_abs_det_jac,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask
            )

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape_as(x),
            z=z,
            z_seq=z_seq,
            x=x
        )

        return output

    def vae_loss_function(self, recon_x, x, mu, log_var, z0, epoch, zk=None, log_abs_det_jac_posterior=None, seq_mask=None, pix_mask=None):
        if self.prior == "standard" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vae_loss_function(recon_x, x, mu, log_var, z0, seq_mask, pix_mask)


        elif self.prior == "vamp" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vamp_loss_function(recon_x, x, mu, log_var, z0, epoch, seq_mask, pix_mask)


        elif self.posterior == "iaf":
            loss, recon_loss, kld = self._vae_iaf_loss_function(recon_x, x, mu, log_var, z0, zk, log_abs_det_jac_posterior, epoch, seq_mask, pix_mask)


        return loss, recon_loss, kld

    def _vae_loss_function(self, recon_x, x, mu, log_var, z, seq_mask=None, pix_mask=None):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _vae_iaf_loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac, epoch, seq_mask=None, pix_mask=None):
        if self.model_config.reconstruction_loss == "mse":

            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        # starting gaussian log-density
        log_prob_z0 = (
            -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
        ).sum(dim=1)

        # prior log-density
        log_prob_zk = self._log_p_z(zk)

        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vamp_loss_function(self, recon_x, x, mu, log_var, z, epoch, seq_mask=None, pix_mask=None):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        if self.linear_scheduling > 0:
            beta = 1.0 * epoch / self.linear_scheduling
            if beta > 1 or not self.training:
                beta = 1.0

        else:
            beta = 1.0

        
        #print((recon_loss * mask.reshape_as(recon_loss)).mean(), (KLD* mask.reshape_as(recon_loss)).mean())


        return (
            ((recon_loss + beta * KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )

    def loss_function(self, recon_x, x, mu, log_var, z_0_vi_index, z_seq, z_vi_index, log_abs_det_jac, log_abs_det_jac_posterior, epoch, seq_mask=None, pix_mask=None):

        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                0.5 * (
                    F.mse_loss(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).mean(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                (
                    F.binary_cross_entropy(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).mean(dim=-1)

        z0 = z_seq[:, 0]

        # starting gaussian log-density
        log_prob_z_vi_index = (
            -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / torch.exp(log_var))
        ).sum(dim=1) - log_abs_det_jac_posterior

        log_p_z = self._log_p_z(z0) 

        # prior log-density
        log_prior_z_vi_index = log_p_z + log_abs_det_jac

        KLD = log_prob_z_vi_index - log_prior_z_vi_index

        if self.linear_scheduling > 0:
            beta = self.beta * epoch / self.linear_scheduling
            if beta > self.beta or not self.training:
                beta = self.beta

        else:
            beta = self.beta

        return (recon_loss + beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _log_p_z(self, z):
        if self.prior == "standard":
            log_p_z = (-0.5 * torch.pow(z, 2)).sum(dim=1)
        
        elif self.prior == "vamp":
            C = self.vamp_number_components

            x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape(
                (C,) + self.model_config.input_dim
            )

            # we bound log_var to avoid unbounded optim
            encoder_output = self.encoder(x)
            prior_mu, prior_log_var = (
                encoder_output.embedding,
                encoder_output.log_covariance,
            )

            z_expand = z.unsqueeze(1)
            prior_mu = prior_mu.unsqueeze(0)
            prior_log_var = prior_log_var.unsqueeze(0)

            log_p_z = (
                torch.sum(
                    -0.5
                    * (
                        prior_log_var
                        + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)
                    ),
                    dim=2,
                )
                - torch.log(torch.tensor(C).type(torch.float))
            )

            log_p_z = torch.logsumexp(log_p_z, dim=1)

        return log_p_z


    def generate(self, z):
        z_for = z

        #_rec = self.decoder(z).reconstruction
        #h = self.encoder(_rec).context

        z_seq = [z]
        for i in range(self.n_obs - 1):
            flow_output = self.flows[i].inverse(z_for)
            z_for = flow_output.out
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1)

        #t = torch.linspace(0, 1, self.n_obs).repeat(z.shape[0], 1).to(z.device)
        #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(z.shape[0], -1, self.latent_dim)), dim=-1)
        return self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"].reshape((z.shape[0], self.n_obs,) + self.input_dim), z_seq #, torch.arange(0, self.n_obs).to(z.device).repeat(z.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"].reshape((z.shape[0], self.n_obs,) + self.input_dim)

    def infer_missing(self, x, seq_mask, pix_mask):
        # iterate on seen images in sequence and keep the one maximizing p(x_i^obs|z)
        
        p_x_given_z = []
        reconstructions = []
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vi_idx = []
        
        for vi_index in range(self.n_obs):

            if seq_mask[0][vi_index] != 0:

                encoder_output = self.encoder(x[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)

                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                h = None#encoder_output.context

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                z_0_vi_index = z

                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':

                    if self.posterior_iaf_config.context_dim is not None:
                        try:
                            h = encoder_output.context

                        except AttributeError as e:
                            raise AttributeError(
                                "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                                "something different from None please ensure that the encoder actually outputs "
                                f"the context vector 'h'. Exception caught: {e}."
                            )

                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                    else:
                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

                z_vi_index = z

                ## propagate in past
                z_seq = []
                z_rev = z_vi_index
                log_abs_det_jac = 0
                for i in range(vi_index - 1, -1, -1):
                    #print("past", i)
                    flow_output = self.flows[i](z_rev)
                    z_rev = flow_output.out
                    log_abs_det_jac += flow_output.log_abs_det_jac
                    z_seq.append(z_rev)

                z_seq.reverse()

                z_seq.append(z_vi_index)

                #proapagate in future
                z_for = z_vi_index
                for i in range(vi_index, self.n_obs - 1):
                    #print("future", i)
                    flow_output = self.flows[i].inverse(z_for)
                    z_for = flow_output.out
                    z_seq.append(z_for)

                z_seq = torch.cat(z_seq, dim=-1)

                #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
                #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

                z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.latent_dim)

                if self.model_config.reconstruction_loss == "mse":
                    recon_loss = (
                        0.5 * (
                            F.mse_loss(
                                recon_x.reshape(x.shape[0]*self.n_obs, -1),
                                x.reshape(x.shape[0]*self.n_obs, -1),
                                reduction="none"
                            ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                        ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
                    ).mean(dim=-1)

                elif self.model_config.reconstruction_loss == "bce":
                
                    recon_loss = (
                        (
                            F.binary_cross_entropy(
                                recon_x.reshape(x.shape[0]*self.n_obs, -1),
                                x.reshape(x.shape[0]*self.n_obs, -1),
                                reduction="none"
                            ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                        ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
                    ).mean(dim=-1)

                p_x_given_z.append(-recon_loss)
                reconstructions.append(recon_x)
                vi_idx.append(vi_index)

        #print(len(p_x_given_z), p_x_given_z[0].shape)
        #print(torch.cat(p_x_given_z).reshape(-1, x.shape[0]).shape)
        idx = torch.argsort(torch.cat(p_x_given_z).reshape(-1, x.shape[0]), dim=0, descending=True)
        #print(idx, torch.cat(reconstructions).reshape((-1, x.shape[0], self.n_obs,)+x.shape[2:]).shape)
        #idx = [int(i + self.n_obs*k) for (k, i) in enumerate(idx)]
        #print(idx, torch.cat(reconstructions).reshape((-1, x.shape[0], self.n_obs,)+x.shape[2:])[idx].shape)
        return torch.cat(reconstructions).reshape((-1, x.shape[0], self.n_obs,)+x.shape[2:]), idx, vi_idx

            



    #def infer_traj(self, x):

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for _ in range(n_full_batch):

                vi_index = np.random.randint(self.n_obs)

                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)

                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                h = None#encoder_output.context

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                z_0_vi_index = z

                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':

                    if self.posterior_iaf_config.context_dim is not None:
                        try:
                            h = encoder_output.context

                        except AttributeError as e:
                            raise AttributeError(
                                "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                                "something different from None please ensure that the encoder actually outputs "
                                f"the context vector 'h'. Exception caught: {e}."
                            )

                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                    else:
                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

                z_vi_index = z

                ## propagate in past
                z_seq = []
                z_rev = z_vi_index
                log_abs_det_jac = 0
                for j in range(vi_index - 1, -1, -1):
                    #print("past", i)
                    flow_output = self.flows[j](z_rev)
                    z_rev = flow_output.out
                    log_abs_det_jac += flow_output.log_abs_det_jac
                    z_seq.append(z_rev)
                z_seq.reverse()
                z_seq.append(z_vi_index)

                #proapagate in future
                z_for = z_vi_index
                for j in range(vi_index, self.n_obs - 1):
                    #print("future", i)
                    flow_output = self.flows[j].inverse(z_for)
                    z_for = flow_output.out
                    z_seq.append(z_for)

                z_seq = torch.cat(z_seq, dim=-1)

                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

                z_seq = z_seq.reshape(x_rep.shape[0], self.n_obs, self.latent_dim)

                
                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = (-0.5 * F.mse_loss(
                            recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                            x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                            reduction="none",
                        ).sum(dim=-1) - torch.tensor(
                            [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                        ).to(
                            data.device
                        )
                    ).reshape(x_rep.shape[0], -1).sum(dim=-1) # decoding distribution is assumed unit variance  N(mu, I)

                    #
                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                        x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                        reduction="none",
                    ).sum(dim=-1).reshape(x_rep.shape[0], -1).sum(dim=-1)

                z0 = z_seq[:, 0]

                # starting gaussian log-density
                log_prob_z_vi_index = (
                    -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / torch.exp(log_var))
                ).sum(dim=1) - log_abs_det_jac_posterior

                log_p_z = self._log_p_z(z0) 

                # prior log-density
                log_prior_z_vi_index = log_p_z + log_abs_det_jac

                log_p_x.append(
                    log_p_x_given_z + log_prior_z_vi_index - log_prob_z_vi_index
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

            if i % 100 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)



class LLDM_IAF(VAE):
    """
    """

    def __init__(
        self,
        model_config: LVAE_IAF_Config,
        encoder: Optional[BaseEncoder],
        decoder: Optional[BaseDecoder] ,
        pretrained_vae: Optional[VAE], #do not forget to call retrieveG on the vae beforehand !
        pretrained_ldm: Optional[MyLatentDiffusion],
        ddim_sampler: Optional[DDIMSampler],
        precomputed_zT_samples = None,
        GM = None,
        temp: Optional[float] = 1.,
        verbose = False
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "LLDM_IAF"

        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim
        self.beta = model_config.beta

        self.flows = nn.ModuleList()

        self.linear_scheduling = self.model_config.linear_scheduling_steps

        self.prior = model_config.prior
        self.posterior = model_config.posterior

        self.pretrained_vae = pretrained_vae
        self.pretrained_ldm = pretrained_ldm
        self.zT_samples = precomputed_zT_samples
        self.ddim_sampler = ddim_sampler
        self.diff_t_steps = np.flip(ddim_sampler.time_steps)
        print('Diffusion time steps ', self.diff_t_steps)
        self.device = self.pretrained_ldm.device
        if verbose:
            print('Running on ', self.device)

        self.temperature = temp
        if self.posterior == "iaf":
            self.posterior_iaf_config = IAFConfig(
                input_dim=(model_config.latent_dim,),
                n_blocks=3,
                n_hidden_in_made=2,
                hidden_size=model_config.hidden_size,
                context_dim=model_config.context_dim,
                include_batch_norm=False,
            )

            self.posterior_iaf_flow = IAF(self.posterior_iaf_config).to(self.device)


        if verbose:
            print('Freezing pre-trained VAE and pre-trained LDM...')
        
        for p in self.pretrained_vae.parameters():
            p.requires_grad = False
        for p in self.pretrained_ldm.parameters():
            p.requires_grad = False
        
        if verbose:
            print('Freezing done.')
            print('Number of trainable parameters: {:.1e}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
            print('Number of total parameters: {:.1e}'.format(sum(p.numel() for p in self.parameters())))


        self.GM = GM

        self.res_dict = {}
        for i in range(self.n_obs):
            self.res_dict[i] = {'rec_loss':0,
                                'reg_loss':0,
                                'count':0}

        

    def forward(self, inputs: BaseDataset, vi_index = None, **kwargs):
        """
        The VAE NF model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        device = self.device

        x = inputs['data'].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x

        if hasattr(inputs, 'seq_mask'):
            seq_mask = inputs['seq_mask'].to(device)
        else:
            seq_mask = torch.ones(x.shape[0], self.n_obs).to(device)
        
        if hasattr(inputs, 'pix_mask'):
            pix_mask = inputs['pix_mask'].to(device)
        else:
            pix_mask = torch.ones_like(x)

        epoch = kwargs.pop("epoch", 100)
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        batch_size = x.shape[0]

        if epoch < self.warmup:
            encoder_output = self.encoder(x)#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)

            z0 = z

            log_abs_det_jac_posterior = 0
            if self.posterior == 'iaf':
                z0 = z

                if self.posterior_iaf_config.context_dim is not None:
                    try:
                        h = encoder_output.context

                    except AttributeError as e:
                        raise AttributeError(
                            "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                            "something different from None please ensure that the encoder actually outputs "
                            f"the context vector 'h'. Exception caught: {e}."
                        )

                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                else:
                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac

            z_seq = z

            #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
            #z = torch.cat((t.unsqueeze(-1), z.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

            recon_x = self.decoder(z_seq)["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

            loss, recon_loss, kld = self.vae_loss_function(
                recon_x=recon_x,
                x=x.reshape((x.shape[0]*self.n_obs,) + x.shape[2:]),
                mu=mu,
                log_var=log_var,
                z0=z0, 
                zk=z,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask)

        else:

            # if missing data pick randomly index of non missing data
            seq_mask = seq_mask.reshape(-1, self.n_obs)
            if vi_index is None:
                if seq_mask[0].sum() < self.n_obs:
                    probs = seq_mask[0].cpu().numpy()
                    probs /= probs.sum()
                    vi_index = np.random.choice(np.arange(self.n_obs), p=probs.reshape(-1))

                else:
                    vi_index = np.random.randint(0, self.n_obs)
            

            encoder_output = self.encoder(x[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            h = None#encoder_output.context

            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)

            z_0_vi_index = z

            log_abs_det_jac_posterior = 0
            if self.posterior == 'iaf':

                if self.posterior_iaf_config.context_dim is not None:
                    try:
                        h = encoder_output.context

                    except AttributeError as e:
                        raise AttributeError(
                            "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                            "something different from None please ensure that the encoder actually outputs "
                            f"the context vector 'h'. Exception caught: {e}."
                        )

                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                else:
                    # Pass it through the Normalizing flows
                    flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                z = flow_output.out
                log_abs_det_jac_posterior += flow_output.log_abs_det_jac

            z_vi_index = z

            ##### FROM LVAE to LLDM ########

            ## propagate in past - Forward Diffusion (Noising)
            z_seq = []
            z_rev = z_vi_index
            for i in range(vi_index - 1, -1, -1): #noising in a sequential way

                #To keep the forward pass parallelisable, we repeat the same sampled vi_index
                t1 = self.diff_t_steps[(i+1)*np.ones(batch_size).astype(int)]
                t2 =  self.diff_t_steps[i*np.ones(batch_size).astype(int)]
                z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

                z_seq.append(z_rev)
##
            z_seq.reverse()
#
            z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

            #propagate in future - Backward Diffusion (Denoising)
            z_for = z_vi_index
            
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
                z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                 index = self.n_obs - 1- i,
                                                                 x = z_for,
                                                                 temperature=self.temperature,
                                                                 repeat_noise=False)
                
                z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for)




            z_seq = torch.cat(z_seq, dim=-1)

            ###############################

            #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
            #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

            z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.latent_dim)
            loss, recon_loss, kld = self.loss_function(
                recon_x=recon_x,
                x=x,
                mu=mu,
                log_var=log_var,
                z_0_vi_index=z_0_vi_index,
                z_seq=z_seq,
                vi_index=vi_index,
                z_vi_index=z_vi_index,
                log_abs_det_jac_posterior=log_abs_det_jac_posterior,
                epoch=epoch,
                seq_mask=seq_mask,
                pix_mask=pix_mask
            )
            self.res_dict[vi_index]['rec_loss'] += recon_loss.item()
            self.res_dict[vi_index]['reg_loss'] += kld.item()
            self.res_dict[vi_index]['count'] += 1


        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape_as(x),
            z=z,
            z_seq=z_seq,
            x=x,
            log_abs_det_jac_posterior=log_abs_det_jac_posterior
        )

        return output

    def vae_loss_function(self, recon_x, x, mu, log_var, z0, epoch, zk=None, log_abs_det_jac_posterior=None, seq_mask=None, pix_mask=None):
        if self.prior == "standard" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vae_loss_function(recon_x, x, mu, log_var, z0, seq_mask, pix_mask)


        elif self.prior == "vamp" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vamp_loss_function(recon_x, x, mu, log_var, z0, epoch, seq_mask, pix_mask)


        elif self.posterior == "iaf":
            loss, recon_loss, kld = self._vae_iaf_loss_function(recon_x, x, mu, log_var, z0, zk, log_abs_det_jac_posterior, epoch, seq_mask, pix_mask)


        return loss, recon_loss, kld

    def _vae_loss_function(self, recon_x, x, mu, log_var, z, seq_mask=None, pix_mask=None):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _vae_iaf_loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac, epoch, seq_mask=None, pix_mask=None):
        if self.model_config.reconstruction_loss == "mse":

            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        # starting gaussian log-density
        log_prob_z0 = (
            -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
        ).sum(dim=1)

        # prior log-density
        log_prob_zk = self._log_p_z(zk)

        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vamp_loss_function(self, recon_x, x, mu, log_var, z, epoch, seq_mask=None, pix_mask=None):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        if self.linear_scheduling > 0:
            beta = 1.0 * epoch / self.linear_scheduling
            if beta > 1 or not self.training:
                beta = 1.0

        else:
            beta = 1.0

        
        #print((recon_loss * mask.reshape_as(recon_loss)).mean(), (KLD* mask.reshape_as(recon_loss)).mean())


        return (
            ((recon_loss + beta * KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )


    def loss_function(self, recon_x, x, mu, log_var, z_0_vi_index, z_seq, vi_index, z_vi_index, log_abs_det_jac_posterior, epoch, seq_mask=None, pix_mask=None):

        assert not torch.isnan(x).any()
        
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                0.5 * (
                    F.mse_loss(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).mean(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                (
                    F.binary_cross_entropy(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).mean(dim=-1)

        z0 = z_seq[:, 0]

        # starting gaussian log-density
        # it is q_\phi ( z_j | x_j) - same as LVAE
        log_prob_z_vi_index = (
            -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / (torch.exp(log_var) + 1e-7)) #adding small constant to avoid numerical instability in the denominator
        ).sum(dim=1) - log_abs_det_jac_posterior


        # prior log-density

        # if vi_index == 0 or vi_index == self.n_obs - 1:
        #     log_prior_z_vi_index = self.log_p_j_hat(j= vi_index, z = z_vi_index)
        #     log_prior_z_vi_index = log_prior_z_vi_index.to(z_vi_index.device)
        #     KLD = log_prob_z_vi_index - log_prior_z_vi_index 
        #     KLD = torch.clamp(KLD, min = -2, max = 500)
        # else:
        #     KLD = torch.zeros_like(log_prob_z_vi_index)

        #if vi_index == 0 or vi_index == self.n_obs - 1:
                
        log_prior_z_vi_index = self.log_p_j_hat(j= vi_index, z = z_vi_index)
        log_prior_z_vi_index = log_prior_z_vi_index.to(z_vi_index.device)
        KLD = log_prob_z_vi_index - log_prior_z_vi_index
        # #########
        # KLD = - KLD #negative KL divergence
        # ###########
        KLD = torch.clamp(KLD, min = -2, max = 500)

        if self.linear_scheduling > 0:
            beta = self.beta * epoch / self.linear_scheduling
            if beta > self.beta or not self.training:
                beta = self.beta

        else:
            beta = self.beta

        return (recon_loss + beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _log_p_z(self, z):
        if self.prior == "standard":
            log_p_z = (-0.5 * torch.pow(z, 2)).sum(dim=1)
        
        elif self.prior == "vamp":
            C = self.vamp_number_components

            x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape(
                (C,) + self.model_config.input_dim
            )

            # we bound log_var to avoid unbounded optim
            encoder_output = self.encoder(x)
            prior_mu, prior_log_var = (
                encoder_output.embedding,
                encoder_output.log_covariance,
            )

            z_expand = z.unsqueeze(1)
            prior_mu = prior_mu.unsqueeze(0)
            prior_log_var = prior_log_var.unsqueeze(0)

            log_p_z = (
                torch.sum(
                    -0.5
                    * (
                        prior_log_var
                        + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)
                    ),
                    dim=2,
                )
                - torch.log(torch.tensor(C).type(torch.float))
            )

            log_p_z = torch.logsumexp(log_p_z, dim=1)

        return log_p_z
    
    def log_p_j_hat(self, j, z):
        """
        Prior on z_j

        Args:
            j (int, 0 <= j <= self.n_obs-1): index of the latent variable (within the sequence)
            z_j (torch.Tensor shape (batch_size, lat_dim)): latent variable
        
        Returns:
            torch.Tensor: prior log-density of z_j

        """

        #For these two special cases, we do not need the sampled z_T, as we know tractable priors

        assert j >= 0 and j < self.n_obs
        if j == 0:
            #z0 follows a standard normal prior
            return (-0.5 * torch.pow(z, 2)).sum(dim=1)
        
        #if j == self.n_obs-1:
        else:
            return self.pretrained_vae.log_pi(z) # log sqrt det G(z) = 0.5 log det G(z) with precompiled G !

            # z_np = z.clone().cpu().detach().numpy()
            # #GM prior
            # return torch.tensor(self.GM.score_samples(z_np)).to(z.device)
        
        # t_diff = self.diff_t_steps[j]
        
        # alpha_bar_t_diff = self.pretrained_ldm.alpha_bar[t_diff]
        # sqrt_alpha_bar_t_diff = self.pretrained_ldm.sqrt_alpha_bar[t_diff]
        # mean = sqrt_alpha_bar_t_diff * self.zT_samples
        # mean = mean.unsqueeze(0).repeat(z.shape[0], 1, 1)

        # mean = mean.to(z.device)

        # log_density = - torch.sum( (z.unsqueeze(1) - mean)**2 / (2 * (1 - alpha_bar_t_diff)), dim = -1)
        # log_density = log_density.mean(dim = -1) #Monte-Carlo average (over the 1000 zT samples)
        # log_density = log_density.mean() #average over the batch
        # return log_density

    def reconstruct(self, x, vi_index):

        device = self.device
        x = x["data"].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        batch_size = x.shape[0]

        encoder_output = self.encoder(x[:, vi_index])
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance


        #std = torch.exp(0.5 * log_var)
        std = torch.zeros_like(log_var)
        z, _ = self._sample_gauss(mu, std)
        z_0_vi_index = z

        log_abs_det_jac_posterior = 0
        if self.posterior == 'iaf':

            if self.posterior_iaf_config.context_dim is not None:
                try:
                    h = encoder_output.context

                except AttributeError as e:
                    raise AttributeError(
                        "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                        "something different from None please ensure that the encoder actually outputs "
                        f"the context vector 'h'. Exception caught: {e}."
                    )

                # Pass it through the Normalizing flows
                flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

            else:
                # Pass it through the Normalizing flows
                flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

            z = flow_output.out
            log_abs_det_jac_posterior += flow_output.log_abs_det_jac

        z_vi_index = z

                ##### FROM LVAE to LLDM ########

        ## propagate in past - Forward Diffusion (Noising)
        z_seq = []
        z_rev = z_vi_index
        for i in range(vi_index - 1, -1, -1): #noising in a sequential way

            #To keep the forward pass parallelisable, we repeat the same sampled vi_index
            t1 = self.diff_t_steps[(i+1)*np.ones(batch_size).astype(int)]
            t2 =  self.diff_t_steps[i*np.ones(batch_size).astype(int)]
            z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

            z_seq.append(z_rev)
##
        z_seq.reverse()
#
        z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

        #propagate in future - Backward Diffusion (Denoising)
        z_for = z_vi_index
        
        for i in range(vi_index, self.n_obs - 1):
            t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
            noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
            z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                index = self.n_obs -1- i,
                                                                x = z_for,
                                                                temperature=self.temperature,
                                                                repeat_noise=False)
            
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
        recon_x = self.decoder(z_seq)["reconstruction"]

        return z_seq, recon_x




    def generate(self, num_gen_seq = 1, start = None, temperature = 0.01):
        

        c, h, w = self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w
        #start = torch.randn(num_gen_seq,c,h,w).to(self.device) if start is None else start
        start = torch.randn(num_gen_seq,c*h*w).to(self.device) if start is None else start

        # z, all_z, all_pred_z0 = self.ddim_sampler.sample(shape=(num_gen_seq,c,h,w), x_last = start, 
        #                                                  temperature = 0)
        # all_z= all_z.reshape(num_gen_seq, self.n_obs , self.latent_dim)

        # all_recons_x = []

        # for i in range(num_gen_seq):
        #     recon_x = self.decoder(all_z[i])["reconstruction"]
        #     all_recons_x.append(recon_x)


        vi_index = 0
        z_for = start
        batch_size = start.shape[0]
        z_seq = [start]
        for i in range(vi_index, self.n_obs - 1):
            t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float()
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
            noise_pred = self.pretrained_ldm(z_for, t)
            z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                index = i,
                                                                x = z_for,
                                                                temperature=temperature,
                                                                repeat_noise=False)
            
            z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1)
        all_recons_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]
        
        #return torch.stack(all_recons_x, dim=0)
        input_c, input_h, input_w = self.pretrained_vae.input_dim
        return all_recons_x.reshape(num_gen_seq, self.n_obs, input_c, input_h, input_w)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps