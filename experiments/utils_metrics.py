import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Independent


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(
            Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(data_2d).squeeze()
    return log_prob


def compute_binary_CE_loss(label_predictions, mortality_label):
    # print('Computing binary classification loss: compute_CE_loss')

    mortality_label = mortality_label.reshape(-1)

    if len(label_predictions.size()) == 1:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj = label_predictions.size(0)
    label_predictions = label_predictions.reshape(n_traj, -1)

    idx_not_nan = ~torch.isnan(mortality_label)
    if len(idx_not_nan) == 0.0:
        print("All are labels are NaNs!")
        ce_loss = torch.Tensor(0.0).to(mortality_label)
    label_predictions = label_predictions[:, idx_not_nan]

    mortality_label = mortality_label[idx_not_nan]

    if torch.sum(mortality_label == 0.0) == 0 or torch.sum(mortality_label == 1.0) == 0:
        print(
            "Warning: all examples in a batch belong to the same class -- please increase the batch size."
        )

    assert not torch.isnan(label_predictions).any()
    assert not torch.isnan(mortality_label).any()

    # For each trajectory, we get n_traj samples from z0 -- compute loss on all of them
    mortality_label = mortality_label.repeat(n_traj, 1)
    ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

    # divide by number of patients in a batch
    ce_loss = ce_loss / n_traj
    return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj, n_samples, n_timepoints, n_dims = data.size()

    res = []
    for i in range(n_traj):
        for k in range(n_samples):
            for j in range(n_dims):
                data_masked = torch.masked_select(
                    data[i, k, :, j], mask[i, k, :, j].bool())
                mu_masked = torch.masked_select(
                    mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(
                    mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    # shape: [n_samples*n_traj, 1]

    res = torch.stack(res, 0).to(data)
    res = res.reshape((n_traj, n_samples, n_dims))
    # Take mean over the number of dimensions
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    # res = res.transpose(0, 1)
    return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj, n_samples, n_timepoints, n_dims = mu.size()

    assert data.size()[-1] == n_dims

    # Shape after permutation: [n_samples, n_traj, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj * n_samples, n_timepoints * n_dims)
        n_traj, n_samples, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(
            n_traj * n_samples, n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj, n_samples).transpose(0, 1)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        def func(mu, data, indices):
            return gaussian_log_likelihood(mu, data, obsrv_std=obsrv_std, indices=indices)
        res = compute_masked_likelihood(mu, data, mask, func)
    return res


def mse(mu, data, indices=None):
    n_data_points = mu.size()[-1]
    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).to(data).squeeze()
    return mse


def compute_mse(mu, data, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj, n_samples, n_timepoints, n_dims = mu.size()
    assert data.size()[-1] == n_dims

    # Shape after permutation: [n_samples, n_traj, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj * n_samples, n_timepoints * n_dims)
        n_traj, n_samples, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(
            n_traj * n_samples, n_timepoints * n_dims)
        res = mse(mu_flat, data_flat)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def compute_multiclass_CE_loss(label_predictions, true_label, mask):

    if len(label_predictions.size()) == 3:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj, n_samples, n_tp, n_dims = label_predictions.size()

    # assert(not torch.isnan(label_predictions).any())
    # assert(not torch.isnan(true_label).any())

    # For each trajectory, we get n_traj samples from z0 -- compute loss on all of them
    true_label = true_label.repeat(n_traj, 1, 1)

    label_predictions = label_predictions.reshape(
        n_traj * n_samples * n_tp, n_dims
    )
    true_label = true_label.reshape(n_traj * n_samples * n_tp, n_dims)

    # choose time points with at least one measurement
    mask = torch.sum(mask, -1) > 0

    # repeat the mask for each label to mark that the label for this time point is present
    pred_mask = mask.repeat(n_dims, 1, 1).permute(1, 2, 0)

    label_mask = mask
    pred_mask = pred_mask.repeat(n_traj, 1, 1, 1)
    label_mask = label_mask.repeat(n_traj, 1, 1, 1)

    pred_mask = pred_mask.reshape(n_traj * n_samples * n_tp, n_dims)
    label_mask = label_mask.reshape(n_traj * n_samples * n_tp, 1)

    if (label_predictions.size(-1) > 1) and (true_label.size(-1) > 1):
        assert label_predictions.size(-1) == true_label.size(-1)
        # targets are in one-hot encoding -- convert to indices
        _, true_label = true_label.max(-1)

    res = []
    for i in range(true_label.size(0)):
        pred_masked = torch.masked_select(
            label_predictions[i], pred_mask[i].bool())
        labels = torch.masked_select(true_label[i], label_mask[i].bool())

        pred_masked = pred_masked.reshape(-1, n_dims)

        if len(labels) == 0:
            continue

        ce_loss = nn.CrossEntropyLoss()(pred_masked, labels.long())
        res.append(ce_loss)

    ce_loss = torch.stack(res, 0).to(label_predictions)
    ce_loss = torch.mean(ce_loss)
    # # divide by number of patients in a batch
    # ce_loss = ce_loss / n_traj
    return ce_loss


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def compute_log_normal_pdf(observed_data, observed_mask, pred_x, args):
    obsrv_std = torch.zeros(pred_x.size()).to(
        observed_data.device) + args.obsrv_std
    noise_logvar = 2. * torch.log(obsrv_std).to(observed_data.device)
    pdf = log_normal_pdf(observed_data, pred_x, noise_logvar, observed_mask)
    logpx = pdf.sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)

    return logpx


def mean_squared_error(orig, pred, mask, mask_select=None):
    error = (orig - pred) ** 2
    if mask_select is not None:
        mask = mask * mask_select
    error = error * mask
    return error.sum() / mask.sum()
