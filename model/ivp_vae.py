import time
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from experiments.utils_metrics import compute_log_normal_pdf
from model.components import Z_to_mu_ReLU, Z_to_std_ReLU

import utils


class IVPVAE(nn.Module):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            ivp_solver):

        super(IVPVAE, self).__init__()

        self.args = args
        self.time_start = 0
        self.latent_dim = args.latent_dim
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.register_buffer('mu', torch.tensor([args.prior_mu]))
        self.register_buffer('std', torch.tensor([args.prior_std]))
        # basic models
        self.embedding_nn = embedding_nn
        self.ivp_solver = ivp_solver
        self.reconst_mapper = reconst_mapper
        self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
        self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

    def forward(self, batch, k_iwae=1):

        results = dict.fromkeys(
            ['likelihood', 'mse', 'forward_time', 'loss'])

        times_in = batch['times_in']
        data_in = batch['data_in']
        mask_in = batch['mask_in']
        if self.args.extrap_full == True:
            times_out = batch['times_out']
            data_out = batch['data_out']
            mask_out = batch['mask_out']
        else:
            times_out = batch['times_in']
            data_out = batch['data_in']
            mask_out = batch['mask_in']

        utils.check_mask(data_in, mask_in)

        self.time_start = time.time()
        # Encoder
        data_embeded = self.embedding_nn(data_in, mask_in)

        t_exist = times_in.gt(torch.zeros_like(times_in))
        lat_exist = t_exist.unsqueeze(-1).repeat(1, 1, self.latent_dim)

        back_time_steps = torch.neg(times_in)
        latent = self.ivp_solver(data_embeded.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        # To see if the variance of latents is becoming small
        lat_mu = torch.sum(latent * lat_exist, dim=-2,
                           keepdim=True) / lat_exist.sum(dim=-2, keepdim=True)
        lat_variance = torch.sum((latent - lat_mu)**2 * lat_exist,
                                 dim=-2, keepdim=True) / lat_exist.sum(dim=-2, keepdim=True)

        z0_mean = self.z2mu_mapper(latent)
        z0_std = self.z2std_mapper(latent) + 1e-8
        z0_mean = z0_mean * lat_exist

        t_loss_start = time.time()
        # KL Divergence Loss
        fp_distr = Normal(z0_mean, z0_std)
        kldiv_z0_all = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))
        assert not (torch.isinf(kldiv_z0_all).any() |
                    torch.isnan(kldiv_z0_all).any())
        kldiv_z0 = torch.sum(kldiv_z0_all * lat_exist, (1, 2)) / \
            lat_exist.sum((1, 2))
        t_loss_end = time.time()
        # do not count the time of computing loss
        self.time_start += t_loss_end - t_loss_start

        # Sampling
        z0_mean_iwae = z0_mean.repeat(k_iwae, 1, 1, 1)
        z0_std_iwae = z0_std.repeat(k_iwae, 1, 1, 1)
        initial_state = utils.sample_standard_gaussian(
            z0_mean_iwae, z0_std_iwae)

        # Integrate inference results
        if self.args.combine_methods == "average":
            initial_state = torch.sum(
                initial_state * lat_exist, dim=-2, keepdim=True) / lat_exist.sum(dim=-2, keepdim=True)
        elif self.args.combine_methods == "kl_weighted":
            # kl_r = 1 / kldiv_z0_all
            kl_r = kldiv_z0_all
            kl_w = kl_r / torch.sum(kl_r * lat_exist, dim=-2, keepdim=True)
            kl_w = (kl_w * lat_exist).repeat(k_iwae, 1, 1, 1)
            initial_state = torch.sum(
                initial_state * kl_w, dim=-2, keepdim=True)
        else:
            raise NotImplementedError

        # Decoder
        sol_z = self.ivp_solver(
            initial_state, times_out.unsqueeze(0))

        # Reconstruction/Modeling Loss
        data_out = data_out.repeat(k_iwae, 1, 1, 1)
        mask_out = mask_out.repeat(k_iwae, 1, 1, 1)

        pred_x = self.reconst_mapper(sol_z)
        rec_likelihood = compute_log_normal_pdf(
            data_out, mask_out, pred_x, self.args)

        t_loss_start = time.time()
        # Monitoring the reconstruction loss of Z
        ll_z = compute_log_normal_pdf(
            data_embeded, lat_exist, sol_z, self.args)
        loss_ll_z = -torch.logsumexp(ll_z, 0).mean(dim=0)
        # sum out the traj dim
        loss = -torch.logsumexp(rec_likelihood -
                                self.args.kl_coef * kldiv_z0, 0)
        # mean over the batch
        loss = torch.mean(loss, dim=0)
        assert not (torch.isnan(loss)).any()
        assert (not torch.isinf(loss).any())

        results["loss"] = loss + self.args.ratio_zz * loss_ll_z
        results['likelihood'] = torch.mean(rec_likelihood).detach()
        results['kldiv_z0'] = torch.mean(kldiv_z0).detach()
        results['loss_ll_z'] = loss_ll_z.detach()
        results["lat_variance"] = torch.mean(lat_variance).detach()

        t_loss_end = time.time()
        self.time_start += t_loss_end - t_loss_start

        forward_info = {'initial_state': initial_state,
                        'sol_z': sol_z,
                        'pred_x': pred_x}

        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch, k_iwae=self.args.k_iwae)
