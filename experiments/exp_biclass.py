import wandb

from experiments import BaseExperiment


class Exp_BiClass(BaseExperiment):

    def validation_step(self, epoch):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_auroc={results['auroc']:.5f}")
        self.logger.info(f"val_auprc={results['auprc']:.5f}")
        self.logger.info(f"val_ce_loss={results['ce_loss']:.5f}")
        self.logger.info(f"val_forward_time={results['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log({"val_auroc": results['auroc'], "epoch_id": epoch})
            wandb.log({"val_auprc": results['auprc'], "epoch_id": epoch})
            wandb.log({"val_ce_loss": results['ce_loss'], "epoch_id": epoch})
            wandb.log(
                {"val_forward_time": results['forward_time'], "epoch_id": epoch})
            wandb.log({"kldiv_z0": results["kldiv_z0"], "epoch_id": epoch})
            # temporally added
            wandb.log({"loss_ll_z": results["loss_ll_z"], "epoch_id": epoch})
            wandb.log(
                {"lat_variance": results["lat_variance"], "epoch_id": epoch})
        if results['val_loss'] != 0:
            return results['val_loss']
        else:
            return results['loss']

    def test_step(self):
        results = self.compute_results_all_batches(self.dltest)
        self.logger.info(f"test_auroc={results['auroc']:.5f}")
        self.logger.info(f"test_auprc={results['auprc']:.5f}")
        self.logger.info(f"test_ce_loss={results['ce_loss']:.5f}")
        self.logger.info(f"test_forward_time={results['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log({"test_auroc": results['auroc'], "run_id": 1})
            wandb.log({"test_auprc": results['auprc'], "run_id": 1})
            wandb.log({"test_ce_loss": results['ce_loss'], "run_id": 1})
            wandb.log(
                {"test_forward_time": results['forward_time'], "run_id": 1})
        if results['val_loss'] != 0:
            return results['val_loss']
        else:
            return results['loss']
