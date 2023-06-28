import time
import torch

from model.ivp_vae import IVPVAE
from model.components import BinaryClassifier
from experiments.utils_metrics import compute_binary_CE_loss


class IVPVAE_BiClass(IVPVAE):
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

        # Classification
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["val_loss"] = results["ce_loss"]
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)
