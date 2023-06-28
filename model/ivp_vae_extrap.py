import time
import torch

from experiments.utils_metrics import compute_log_normal_pdf, mean_squared_error
from model.ivp_vae import IVPVAE


class IVPVAE_Extrap(IVPVAE):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver)

        self.args = args

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results['forward_time'] = time.time() - self.time_start
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            sol_z = self.ivp_solver(
                forward_info['initial_state'], batch['times_out'].unsqueeze(0))

            next_data = data_out.repeat(k_iwae, 1, 1, 1)
            next_mask = mask_out.repeat(k_iwae, 1, 1, 1)
            pred_x = self.reconst_mapper(sol_z)
            results['forward_time'] = time.time() - self.time_start
            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, dim=0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            assert (not torch.isnan(loss_next))

            if self.args.train_w_reconstr:
                results["loss"] = results["loss"] + \
                    self.args.ratio_nl * loss_next
            else:
                results["loss"] = loss_next

            mse_extrap = mean_squared_error(next_data, pred_x, mask=next_mask)
            results["mse_extrap"] = torch.mean(mse_extrap).detach()
            results['mse'] = torch.mean(mean_squared_error(
                batch['data_in'], forward_info['pred_x'], mask=batch['mask_in'])).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)
