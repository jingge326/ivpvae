import copy
import os
import re
import time
import logging
import random
from argparse import Namespace
from pathlib import Path
from copy import deepcopy

import numpy as np
import wandb
import sklearn as sk
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from experiments.data_eicu import get_eicu_tvt_datasets
from experiments.data_mimic import collate_fn_biclass, collate_fn_extrap, get_m4_tvt_datasets
from experiments.data_physionet12 import get_p12_tvt_datasets
from model.model_factory import ModelFactory

from utils import record_experiment


class BaseExperiment:
    ''' Base experiment class '''

    def __init__(self, args: Namespace):
        self.args = args
        self.epochs_max = args.epochs_max
        self.patience = args.patience
        self.proj_path = Path(args.proj_path)
        self.mf = ModelFactory(self.args)
        self.tags = [
            self.args.ml_task,
            self.args.data,
            self.args.leit_model,
            self.args.ivp_solver,
            self.args.test_info]

        self.args.exp_name = '_'.join(
            self.tags + [("r"+str(args.random_state))])

        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)

        self._init_logger()
        self.device = torch.device(args.device)
        self.logger.info(f'Device: {self.device}')

        self.variable_num, self.dltrain, self.dlval, self.dltest = self.get_data()
        self.model = self.get_model().to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f'num_params={num_params}')

        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = None
        if args.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim, args.lr_scheduler_step, args.lr_decay)

    def _init_logger(self):

        logging.basicConfig(filename=self.proj_path / 'log' / (self.args.exp_name+'.log'),
                            filemode='w',
                            level=logging.INFO,
                            force=True)

        self.logger = logging.getLogger()

        if self.args.log_tool == 'wandb':
            # initialize weight and bias
            os.environ["WANDB__SERVICE_WAIT"] = "1800"
            wandb.init(
                project="ivp_vae",
                config=copy.deepcopy(dict(self.args._get_kwargs())),
                group="_".join(self.tags),
                tags=self.tags,
                name="r"+str(self.args.random_state))

    def get_model(self):
        print('current_ehr_variable_num: ' + str(self.variable_num))
        if self.args.ml_task == 'extrap':
            model = self.mf.initialize_extrap_model()
        elif self.args.ml_task == 'biclass':
            model = self.mf.initialize_biclass_model()
        else:
            raise ValueError("Unknown")

        return model

    def get_data(self):
        self.args.num_times = self.args.time_max + 1
        if re.match("m4", self.args.data):
            train_data, val_data, test_data = get_m4_tvt_datasets(
                self.args, self.proj_path, self.logger)
        elif re.match("p12", self.args.data):
            train_data, val_data, test_data = get_p12_tvt_datasets(
                self.args, self.proj_path, self.logger)
        elif re.match("eicu", self.args.data):
            train_data, val_data, test_data = get_eicu_tvt_datasets(
                self.args, self.proj_path, self.logger)
        else:
            raise ValueError("Unsupported Dataset!")

        if self.args.ml_task == 'extrap':
            collate_fn = collate_fn_extrap
        elif self.args.ml_task == 'biclass':
            collate_fn = collate_fn_biclass
        else:
            raise ValueError("Unknown")

        dl_train = DataLoader(
            dataset=train_data,
            collate_fn=lambda batch: collate_fn(
                batch, train_data.variable_num, self.args),
            shuffle=True,
            batch_size=self.args.batch_size)
        dl_val = DataLoader(
            dataset=val_data,
            collate_fn=lambda batch: collate_fn(
                batch, val_data.variable_num, self.args),
            shuffle=True,
            batch_size=self.args.batch_size)
        dl_test = DataLoader(
            dataset=test_data,
            collate_fn=lambda batch: collate_fn(
                batch, test_data.variable_num, self.args),
            shuffle=True,
            batch_size=self.args.batch_size)

        return train_data.variable_num, dl_train, dl_val, dl_test

    def training_step(self, batch):
        results = self.model.compute_prediction_results(batch)
        return results['loss']

    def validation_step(self) -> Tensor:
        raise NotImplementedError

    def test_step(self) -> Tensor:
        raise NotImplementedError

    def compute_results_all_batches(self, dl):
        total = {}
        total['loss'] = 0
        total['likelihood'] = 0
        total['mse'] = 0
        total["auroc"] = 0
        total['kl_first_p'] = 0
        total['std_first_p'] = 0
        total['ce_loss'] = 0
        total['mse_reg'] = 0
        total['mae_reg'] = 0
        total['mse_extrap'] = 0
        total['forward_time'] = 0
        total['kldiv_z0'] = 0
        total['loss_ae'] = 0
        total['loss_vae'] = 0
        total['loss_ll_z'] = 0
        total["val_loss"] = 0
        total["lat_variance"] = 0

        n_test_batches = 0

        classif_predictions = torch.Tensor([]).to(self.args.device)
        all_test_labels = torch.Tensor([]).to(self.args.device)
        n_traj_samples = self.args.k_iwae

        for batch in dl:
            results = self.model.run_validation(batch)

            if self.args.ml_task == 'biclass':
                n_labels = 1  # batch['truth'].size(-1)
                classif_predictions = torch.cat(
                    (classif_predictions, results["label_predictions"].reshape(n_traj_samples, -1, n_labels)), 1)
                all_test_labels = torch.cat((all_test_labels,
                                            batch['truth'].reshape(-1, n_labels)), 0)

            for key in total.keys():
                if results.get(key) is not None:
                    var = results[key]
                    if isinstance(var, torch.Tensor):
                        var = var.detach()
                    total[key] += var

            n_test_batches += 1

        if n_test_batches > 0:
            for key, _ in total.items():
                total[key] = total[key] / n_test_batches

        if self.args.ml_task == 'biclass':
            all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)

            total["auroc"] = 0.0
            total["auprc"] = 0.0
            if torch.sum(all_test_labels) != 0.0:
                print("Number of labeled examples: {}".format(
                    int(len(all_test_labels.reshape(-1))/n_traj_samples)))
                print("Number of examples with mortality 1: {}".format(
                    int(torch.sum(all_test_labels == 1.0)/n_traj_samples)))

                array_truth = all_test_labels.cpu().numpy().reshape(-1)
                array_predict = classif_predictions.cpu().numpy().reshape(-1)
                total["auroc"] = sk.metrics.roc_auc_score(
                    array_truth, array_predict)
                total["auprc"] = sk.metrics.average_precision_score(
                    array_truth, array_predict)
            else:
                print(
                    "Warning: Couldn't compute AUC -- all examples are from the same class")

        return total

    def run(self) -> None:
        # Training loop parameters
        best_loss = float('inf')
        waiting = 0
        durations = []
        best_model = deepcopy(self.model.state_dict())

        for epoch in range(1, self.epochs_max):
            iteration = 1
            self.model.train()
            start_time = time.time()

            for batch in self.dltrain:
                # Single training step
                self.optim.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                if self.args.clip_gradient:
                    # Optional gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip)
                self.optim.step()

                self.logger.info(
                    f'[epoch={epoch:04d}|iter={iteration:04d}] train_loss={train_loss:.5f}')
                if self.args.log_tool == 'wandb':
                    wandb.log({"train_loss": train_loss})
                iteration += 1

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(
                f'[epoch={epoch:04d}] epoch_duration={epoch_duration:5f}')

            # Validation step
            self.model.eval()
            val_loss = self.validation_step(epoch)
            self.logger.info(f'[epoch={epoch:04d}] val_loss={val_loss:.5f}')
            if self.args.log_tool == 'wandb':
                wandb.log({"epoch_duration": epoch_duration, "epoch_id": epoch})
                wandb.log({"val_loss": val_loss, "epoch_id": epoch})

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early stopping procedure
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model.state_dict())
                waiting = 0
            else:
                waiting += 1

            if waiting >= self.patience:
                break

            if self.args.log_tool == 'wandb':
                wandb.log(
                    {"lr": self.optim.param_groups[0]['lr'], "epoch_id": epoch})

        # Load best model
        self.model.load_state_dict(best_model)
        # Held-out test set step
        test_loss = self.test_step()

        self.logger.info(f'epoch_duration_mean={np.mean(durations):.5f}')
        self.logger.info(f'test_loss={test_loss:.5f}')

        if self.args.log_tool == 'wandb':
            wandb.log({"epoch_duration_mean": np.mean(durations), "run_id": 1})
            wandb.log({"test_loss": test_loss, "run_id": 1})

    def finish(self):
        record_experiment(self.args, self.model)
        torch.save(self.model.state_dict(), self.proj_path /
                   'temp/model' / (self.args.exp_name+'.pt'))
        logging.shutdown()
