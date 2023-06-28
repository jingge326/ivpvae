import wandb

from experiments import BaseExperiment


class Exp_Extrap(BaseExperiment):

    def validation_step(self, epoch):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_mse={results['mse']:.5f}")
        self.logger.info(f"val_mse_extrap={results['mse_extrap']:.5f}")
        self.logger.info(f"val_forward_time={results['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log({"val_mse": results['mse'], "epoch_id": epoch})
            wandb.log(
                {"val_mse_extrap": results['mse_extrap'], "epoch_id": epoch})
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
        self.logger.info(f"test_mse={results['mse']:.5f}")
        self.logger.info(f"test_mse_extrap={results['mse_extrap']:.5f}")
        self.logger.info(f"test_forward_time={results['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log({"test_mse": results['mse'], "run_id": 1})
            wandb.log({"test_mse_extrap": results['mse_extrap'], "run_id": 1})
            wandb.log(
                {"test_forward_time": results['forward_time'], "run_id": 1})
        if results['val_loss'] != 0:
            return results['val_loss']
        else:
            return results['loss']
