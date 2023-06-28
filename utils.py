from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import sklearn as sk

PROJ_FOLDER = Path(__file__).parents[0]
DATA_DIR = PROJ_FOLDER / "data"


# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):

    data_min, data_max = None, None
    inf = torch.Tensor([float('Inf')])[0]

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def init_network_weights(net, method='normal_'):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if method == 'xavier_uniform_':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming_uniform_':
                nn.init.kaiming_uniform_(m.weight)
            else:
                nn.init.normal_(m.weight, mean=0, std=0.1)

            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    epsilon = torch.distributions.normal.Normal(torch.Tensor(
        [0.0]).to(device), torch.Tensor([1.0]).to(device))
    r = epsilon.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def split_train_val_test(data, train_frac=0.6, val_frac=0.2):
    n_samples = len(data)
    data_train = data[: int(n_samples * train_frac)]
    data_val = data[
        int(n_samples * train_frac): int(n_samples * (train_frac + val_frac))
    ]
    data_test = data[int(n_samples * (train_frac + val_frac)):]
    return data_train, data_val, data_test


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert start.size() == end.size()
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat(
                (res, torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def reverse(tensor):
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    return tensor[idx]


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def get_dict_template():
    return {
        "observed_data": None,
        "observed_tp": None,
        "data_to_predict": None,
        "tp_to_predict": None,
        "observed_mask": None,
        "mask_predicted_data": None,
        'labels': None,
    }


def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[att_max == 0.0] = 1.0

    if (att_max != 0.0).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.0] = 1.0

    if (att_max != 0.0).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def shift_outputs(outputs, first_datapoint=None):
    outputs = outputs[:, :, :-1, :]

    if first_datapoint is not None:
        n_traj, n_dims = first_datapoint.size()
        first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
        outputs = torch.cat((first_datapoint, outputs), 2)
    return outputs


def split_data_extrap(data_dict, dataset=""):
    # data_dict = {
    # 'data': combined_vals,  # (batch, sequence length, dim=41)
    # 'time_steps': combined_tt,  # (batch, sequence length)
    # 'mask': combined_mask,  # (batch, sequence length, dim=41)
    # 'labels': combined_labels  # (batch, 1)
    # }
    device = get_device(data_dict["data"])

    n_observed_tp = data_dict["data"].size(1) // 2
    if dataset == "hopper":
        n_observed_tp = data_dict["data"].size(1) // 3

    split_dict = {
        "observed_data": data_dict["data"][:, :n_observed_tp, :].clone(),
        "observed_tp": data_dict["times"][:n_observed_tp].clone(),
        "data_to_predict": data_dict["data"][:, n_observed_tp:, :].clone(),
        "tp_to_predict": data_dict["times"][n_observed_tp:].clone(),
    }

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict['labels'] = None

    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:,
                                                        :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:,
                                                              n_observed_tp:].clone()

    if ('labels' in data_dict) and (data_dict['labels'] is not None):
        split_dict['labels'] = data_dict['labels'].clone()

    split_dict["mode"] = "extrap"
    return split_dict


def split_data_interp(data_dict):
    device = get_device(data_dict["data"])

    split_dict = {
        "observed_data": data_dict["data"].clone(),
        "observed_tp": data_dict["times"].clone(),
        "data_to_predict": data_dict["data"].clone(),
        "tp_to_predict": data_dict["times"].clone(),
    }

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict['labels'] = None

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ('labels' in data_dict) and (data_dict['labels'] is not None):
        split_dict['labels'] = data_dict['labels'].clone()

    split_dict["mode"] = "interp"
    return split_dict


def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict


def split_and_subsample_batch(data_dict, args, data_type="train"):
    if data_type == "train":
        # Training set
        if args.extrap == True:
            processed_dict = split_data_extrap(data_dict, dataset=args.data)
        else:
            processed_dict = split_data_interp(data_dict)
    else:
        # Test set
        if args.extrap == True:
            processed_dict = split_data_extrap(data_dict, dataset=args.data)
        else:
            processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)
    return processed_dict


def compute_results_all_batches(model, dl, args):
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

    n_test_batches = 0

    classif_predictions = torch.Tensor([]).to(args.device)
    all_test_labels = torch.Tensor([]).to(args.device)
    n_traj_samples = args.k_iwae

    for batch in dl:
        results = model.run_validation(batch)

        if args.ml_task == 'biclass':
            n_labels = 1  # batch['truth'].size(-1)
            classif_predictions = torch.cat(
                (classif_predictions, results["label_predictions"].reshape(n_traj_samples, -1, n_labels)), 1)
            all_test_labels = torch.cat((all_test_labels,
                                         batch['truth'].reshape(-1, n_labels)), 0)

        for key in total.keys():
            if key in results:
                var = results[key]
                if isinstance(var, torch.Tensor):
                    var = var.detach()
                total[key] += var

        n_test_batches += 1

    if n_test_batches > 0:
        for key, _ in total.items():
            total[key] = total[key] / n_test_batches

    if args.ml_task == 'biclass':
        all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)
        idx_not_nan = ~torch.isnan(all_test_labels)
        classif_predictions = classif_predictions[idx_not_nan]
        all_test_labels = all_test_labels[idx_not_nan]

        total["auroc"] = 0.0  # AUC score
        if torch.sum(all_test_labels) != 0.0:
            print("Number of labeled examples: {}".format(
                len(all_test_labels.reshape(-1))))
            print("Number of examples with mortality 1: {}".format(
                torch.sum(all_test_labels == 1.0)))

            # Cannot compute AUC with only 1 class
            total["auroc"] = sk.metrics.roc_auc_score(all_test_labels.cpu().numpy().reshape(-1),
                                                      classif_predictions.cpu().numpy().reshape(-1))
        else:
            print(
                "Warning: Couldn't compute AUC -- all examples are from the same class")

    return total


def check_mask(data, mask):
    # check that 'mask' argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.0).cpu().numpy()
    n_ones = torch.sum(mask == 1.0).cpu().numpy()

    # mask should contain only zeros and ones
    assert (n_zeros + n_ones) == np.prod(list(mask.size()))

    # all masked out elements should be zeros
    assert torch.sum(data[mask == 0.0] != 0.0) == 0


class SolverWrapper(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, x, t, backwards=False):
        assert len(x.shape) - len(t.shape) == 1
        t = t.unsqueeze(-1)
        if t.shape[-3] == 1:
            t = t.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4 and x.shape[0] != t.shape[0]:
            t = t.repeat_interleave(x.shape[0], dim=0)
        y = self.solver(x, t)  # (1, batch_size, times, dim)
        return y


def record_experiment(args, model):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    file_result = PROJ_FOLDER / 'results/{}.txt'.format(dt_string)
    with open(file_result, 'w') as fr:
        json.dump(vars(args), fr, indent=0)
        print(model, file=fr)


def calc_mean_std(df):
    mean = df.mean()
    if len(df) == 1 or df.std() == 0:
        std = 1
    else:
        std = df.std()
    return mean, std


# taken from https://github.com/ALRhub/rkn_share/ and not modified
class TimeDistributed(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def __init__(self, module, low_mem=False, num_outputs=1):
        """
        Makes a torch model time distributed. If the original model works with Tensors of size [batch_size] + data_shape
        this wrapper makes it work with Tensors of size [batch_size, sequence_length] + data_shape
        :param module: The module to wrap
        :param low_mem: Default is to the fast but high memory version. If you run out of memory set this to True
                        (it will be slower than)
            - low memory version: simple forloop over the time axis -> slower but consumes less memory
            - not low memory version: "reshape" and then process all at once -> faster but consumes more memory
        :param num_outputs: Number of outputs of the original module (really the number of outputs,
               not the dimensionality, e.g., for the normal RKN encoder that should be 2 (mean and variance))
        """

        super(TimeDistributed, self).__init__()
        self._module = module
        if num_outputs > 1:
            self.forward = self._forward_low_mem_multiple_outputs if low_mem else self._forward_multiple_outputs
        else:
            self.forward = self._forward_low_mem if low_mem else self._forward
        self._num_outputs = num_outputs

    def _forward(self, *args):
        input_shapes = [args[i].shape for i in range(len(args))]
        batch_size, seq_length = input_shapes[0][0], input_shapes[0][1]
        out = self._module(*[x.view(batch_size * seq_length,
                           *input_shapes[i][2:]) for i, x in enumerate(args)])
        return out.view(batch_size, seq_length, *out.shape[1:])

    def _forward_multiple_outputs(self, *args):
        input_shapes = [args[i].shape for i in range(len(args))]
        batch_size, seq_length = input_shapes[0][0], input_shapes[0][1]
        outs = self._module(
            *[x.view(batch_size * seq_length, *input_shapes[i][2:]) for i, x in enumerate(args)])
        out_shapes = [outs[i].shape for i in range(self._num_outputs)]
        return [outs[i].view(batch_size, seq_length, *out_shapes[i][1:]) for i in range(self._num_outputs)]

    def _forward_low_mem(self, x):
        out = []
        unbound_x = x.unbind(1)
        for x in unbound_x:
            out.append(self._module(x))
        return torch.stack(out, dim=1)

    def _forward_low_mem_multiple_outputs(self, x):
        out = [[] for _ in range(self._num_outputs)]
        unbound_x = x.unbind(1)
        for x in unbound_x:
            outs = self._module(x)
            [out[i].append(outs[i]) for i in range(self._num_outputs)]
        return [torch.stack(out[i], dim=1) for i in range(self._num_outputs)]


def log_lik_gaussian_simple(x, mu, logvar):
    """
    Return loglikelihood of x in gaussian specified by mu and logvar, taken from
    https://github.com/edebrouwer/gru_ode_bayes
    """
    return np.log(np.sqrt(2 * np.pi)) + (logvar / 2) + ((x - mu).pow(2) / (2 * logvar.exp()))
