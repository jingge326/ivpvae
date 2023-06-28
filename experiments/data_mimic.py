import re
import pandas as pd
import numpy as np
import sklearn.model_selection
import torch
from torch.utils.data import Dataset

import utils


def merge_timestamps(df_data, down_times):
    df_data["Time"] = df_data["Time"].div(
        down_times).apply(np.floor)*down_times
    # max() can integrate both value columns and mask columns
    df_data.reset_index(inplace=True)
    df_data = df_data.groupby(["ID", "Time"], as_index=False).max()
    df_data.set_index("ID", inplace=True)
    return df_data


# Randomly assign some elements of a Pandas DataFrame to zero according to a certain ratio
def mask_random(df_in, ratio):
    df_out = df_in.copy()
    mask = np.random.choice([0, 1], size=df_in.shape, p=[ratio, 1-ratio])
    df_out = df_out * mask
    return df_out


def filter_tvt(df_all, logger, args):
    # For Extrap: there are values before and after 24h
    # For Classf: the patient didn't die within the first 24h
    ids_before = df_all.loc[df_all['Time']
                            < args.next_start].index.unique()
    ids_after = df_all.loc[df_all['Time']
                           > args.next_start].index.unique()
    ids_selected = set(ids_before) & set(ids_after)
    df_all = df_all.loc[list(ids_selected)]
    logger.info("Number of samples: {}".format(len(ids_selected)))

    if args.time_max < df_all['Time'].max():
        df_all = df_all.loc[df_all['Time'] <= args.time_max]

    if args.mask_drop_rate > 0:
        value_cols = []
        mask_cols = []
        for col in df_all.columns:
            value_cols.append(col.startswith("Value"))
            mask_cols.append(col.startswith("Mask"))

        value = df_all.loc[:, value_cols].values
        mask = df_all.loc[:, mask_cols].values
        mask_new = mask_random(mask, args.mask_drop_rate)
        value_new = value * mask_new
        df_all.loc[:, value_cols] = value_new
        df_all.loc[:, mask_cols] = mask_new
        df_all = df_all.loc[df_all.loc[:, mask_cols].sum(axis=1) > 0]

    # Splitting
    ids_train, ids_vt = sklearn.model_selection.train_test_split(
        df_all.index.unique(),
        train_size=0.8,
        random_state=args.random_state,
        shuffle=True)

    ids_valid, ids_test = sklearn.model_selection.train_test_split(
        ids_vt,
        train_size=0.5,
        random_state=args.random_state,
        shuffle=True)

    data_train = df_all.loc[ids_train]
    data_validation = df_all.loc[ids_valid]
    data_test = df_all.loc[ids_test]

    return data_train, data_validation, data_test


def load_tvt(args, m4_path, logger):
    if re.search("_\d+$", args.data) is not None:
        m4_path = m4_path/("r"+str(args.random_state))
        name = args.data
        print("m4_path: " + str(m4_path))
        print("data: " + name)
        data_train = pd.read_csv(
            m4_path/(name + "_train.csv"), index_col=0)
        data_validation = pd.read_csv(
            m4_path/(name + "_valid.csv"), index_col=0)
        if re.search("mortality", name) is not None:
            data_test = pd.read_csv(
                m4_path/"data_mimic4_mortality_test.csv", index_col=0)
        elif re.search("length", name) is not None:
            data_test = pd.read_csv(
                m4_path/"data_mimic4_length_test.csv", index_col=0)
        elif re.search("next", name) is not None:
            data_test = pd.read_csv(
                m4_path/"data_mimic4_next_test.csv", index_col=0)
        else:
            raise ValueError("Unknown dataset")

    elif args.data == "m4_full":
        data_mimic = pd.read_csv(
            m4_path/'mimic4_full_dataset.csv', index_col=0)
        if args.num_samples != -1:
            tvt_ids = pd.DataFrame(data_mimic.index.unique(), columns=['ID']).sample(
                n=args.num_samples, random_state=args.random_state)
            data_tvt = data_mimic.loc[tvt_ids['ID']]
        else:
            data_tvt = data_mimic

        data_train, data_validation, data_test = filter_tvt(
            data_tvt, logger, args)
    else:
        raise ValueError("Unknown dataset")

    return data_train, data_validation, data_test


def scale_tvt(args, data_train, data_validation, data_test, logger):
    logger.info("Number of samples for training: {}".format(
        data_train.index.nunique()))
    logger.info("Number of samples for validation: {}".format(
        data_validation.index.nunique()))
    logger.info("Number of samples for testing: {}".format(
        data_test.index.nunique()))
    time_max = args.time_max
    if args.down_times > 1:
        data_train = merge_timestamps(data_train, args.down_times)
        data_validation = merge_timestamps(
            data_validation, args.down_times)
        data_test = merge_timestamps(data_test, args.down_times)

    if args.t_offset > 0:
        data_train.loc[:, "Time"] = data_train["Time"] + args.t_offset
        data_validation.loc[:,
                            "Time"] = data_validation["Time"] + args.t_offset
        data_test.loc[:, "Time"] = data_test["Time"] + args.t_offset
        time_max += args.t_offset

    # the timestamp was the time delta between the first chart time for each admission
    if args.time_scale == "time_max":
        print("time_max 1:{}".format(time_max))
        print("self_max 2:{}".format(data_test["Time"].max()))
        data_train.loc[:, "Time"] = data_train["Time"] / time_max
        data_validation.loc[:, "Time"] = data_validation["Time"] / time_max
        data_test.loc[:, "Time"] = data_test["Time"] / time_max

    # need to delete max
    elif args.time_scale == "self_max" or args.time_scale == "max":
        data_train.loc[:, "Time"] = data_train["Time"] / \
            data_train["Time"].max()
        data_validation.loc[:, "Time"] = data_validation["Time"] / \
            data_validation["Time"].max()
        data_test.loc[:, "Time"] = data_test["Time"] / data_test["Time"].max()

    elif args.time_scale == "constant":
        data_train.loc[:, "Time"] = data_train["Time"] / args.time_constant
        data_validation.loc[:,
                            "Time"] = data_validation["Time"] / args.time_constant
        data_test.loc[:, "Time"] = data_test["Time"] / args.time_constant

    value_cols = [c.startswith("Value") for c in data_train.columns]
    value_cols = data_train.iloc[:, value_cols]
    mask_cols = [("Mask" + x[5:]) for x in value_cols]

    data_train.dropna(inplace=True)

    # Normalizing values
    for item in zip(value_cols, mask_cols):
        val_train = data_train.loc[data_train[item[1]].astype("bool"), item[0]]
        val_validation = data_validation.loc[data_validation[item[1]].astype(
            "bool"), item[0]]
        val_test = data_test.loc[data_test[item[1]].astype("bool"), item[0]]

        df_tv = pd.concat([val_train, val_validation])
        df_tvt = pd.concat([val_train, val_validation, val_test])
        mean_tv, std_tv = utils.calc_mean_std(df_tv)
        mean_t, std_t = utils.calc_mean_std(df_tvt)

        data_train.loc[data_train[item[1]].astype("bool"), item[0]] = (
            val_train - mean_tv) / std_tv
        data_validation.loc[data_validation[item[1]].astype(
            "bool"), item[0]] = (val_validation - mean_tv) / std_tv

        data_test.loc[data_test[item[1]].astype("bool"), item[0]] = (
            val_test - mean_t) / std_t

    data_train.dropna(inplace=True)
    data_validation.dropna(inplace=True)
    data_test.dropna(inplace=True)

    return data_train, data_validation, data_test


def get_m4_tvt_datasets(args, proj_path, logger):
    m4_path = proj_path/"data/mimic4/processed/"
    data_train, data_validation, data_test = load_tvt(args, m4_path, logger)
    data_train, data_validation, data_test = scale_tvt(
        args, data_train, data_validation, data_test, logger)

    if args.ml_task == "biclass":
        mortality_m4_path = m4_path/"mortality_labels.csv"
        label_data = pd.read_csv(mortality_m4_path)
        label_data.loc["labels"] = label_data["labels"].astype(float)
        train = DatasetBiClass(
            data_train.reset_index(), label_df=label_data, ts_full=args.ts_full)
        val = DatasetBiClass(
            data_validation.reset_index(), label_df=label_data, ts_full=args.ts_full)
        test = DatasetBiClass(
            data_test.reset_index(), label_df=label_data, ts_full=args.ts_full)

    elif args.ml_task == "extrap":
        val_options = {"T_stop": (args.next_end+args.t_offset)/args.time_constant,
                       "T_val": (args.next_start+args.t_offset)/args.time_constant,
                       "max_val_samples": args.next_headn}
        train = DatasetExtrap(
            data_train.reset_index(), val_options, args.extrap_full, ts_full=args.ts_full)
        val = DatasetExtrap(data_validation.reset_index(),
                            val_options, args.extrap_full, ts_full=args.ts_full)
        test = DatasetExtrap(data_test.reset_index(),
                             val_options, args.extrap_full, ts_full=args.ts_full)

    else:
        raise ValueError("Unknown M4 mode!")

    return train, val, test


class DatasetBiClass(Dataset):
    def __init__(self, in_df, label_df, ts_full=False):

        self.in_df = in_df
        adm_ids = self.in_df.loc[:, "ID"].unique()
        self.label_df = label_df[label_df["ID"].isin(adm_ids)]

        # how many different ids are there
        self.length = len(adm_ids)

        # how many different variables are there
        self.variable_num = sum([col.startswith("Value")
                                for col in self.in_df.columns])

        # Rename all the admission ids
        map_dict = dict(zip(adm_ids, np.arange(self.length)))
        self.in_df.loc[:, "ID"] = self.in_df.loc[:, "ID"].map(map_dict)
        self.label_df.loc[:, "ID"] = self.label_df["ID"].map(
            map_dict)

        # data processing
        self.in_df = self.in_df.astype(np.float32)
        self.in_df.ID = self.in_df.ID.astype(int)
        self.label_df["labels"] = self.label_df["labels"].astype(np.float32)
        if ts_full:
            self.in_df.Time = self.in_df.Time.astype(int)
        # This step is important for the batch sampling on admission ids
        self.in_df.set_index("ID", inplace=True)
        self.in_df.sort_values("Time", inplace=True)
        self.label_df.set_index("ID", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # given the admission id, get the whole time series
        # samples = self.in_df.loc[idx]
        # if len(samples.shape) == 1:
        #     samples = self.in_df.loc[[idx]]
        samples = self.in_df.loc[[idx]]
        label = self.label_df.loc[idx].values
        return {"idx": idx, "truth": label, "samples": samples}


class DatasetExtrap(Dataset):
    def __init__(self, in_df, val_options, extrap_full, ts_full=False):

        self.in_df = in_df

        # how many different variables are there
        self.variable_num = sum([col.startswith("Value")
                                for col in self.in_df.columns])

        before_idx = self.in_df.loc[self.in_df["Time"]
                                    < val_options["T_val"], "ID"].unique()

        if val_options.get("T_stop") is not None:
            after_idx = self.in_df.loc[(self.in_df["Time"] >= val_options["T_val"]) & (
                self.in_df["Time"] < val_options["T_stop"]), "ID", ].unique()
        else:
            after_idx = self.in_df.loc[self.in_df["Time"]
                                       >= val_options["T_val"], "ID"].unique()

        valid_idx = np.intersect1d(before_idx, after_idx)

        # Fix the case for droping out observations with different rates
        if (len(valid_idx) / len(before_idx) < 0.8) or (len(valid_idx) / len(after_idx) < 0.8):
            raise ValueError("Wrong")
        # if (len(valid_idx) != len(before_idx)) or (len(valid_idx) != len(after_idx)):
        #     raise ValueError("Wrong")

        self.in_df = self.in_df.loc[self.in_df["ID"].isin(valid_idx)].copy()

        # how many different ids are there
        self.length = self.in_df["ID"].nunique()

        # Rename all the admission ids
        map_dict = dict(
            zip(self.in_df.loc[:, "ID"].unique(), np.arange(self.length)))
        self.in_df.loc[:, "ID"] = self.in_df.loc[:, "ID"].map(map_dict)

        # data processing
        self.in_df = self.in_df.astype(np.float32)
        self.in_df.ID = self.in_df.ID.astype(int)
        self.in_df.sort_values("Time", inplace=True)

        self.df_before = self.in_df.loc[self.in_df["Time"]
                                        < val_options["T_val"]].copy()
        self.df_after = self.in_df.loc[self.in_df["Time"]
                                       >= val_options["T_val"]].copy()

        if val_options.get("T_stop"):
            self.df_after = self.df_after.loc[self.df_after["Time"]
                                              < val_options["T_stop"]].copy()

        if val_options["max_val_samples"] > 0:
            self.df_after = (self.df_after.groupby("ID").head(
                val_options["max_val_samples"]).copy())

        if extrap_full == True:
            self.df_val = pd.concat((self.df_before, self.df_after))
        else:
            self.df_val = self.df_after

        self.df_val.sort_values(by=["ID", "Time"], inplace=True)
        if ts_full == True:
            self.df_before.Time = self.df_before.Time.astype(int)
            self.df_val.Time = self.df_val.Time.astype(int)
        # This step is important for the batch sampling on admissioin ids
        self.df_before.set_index("ID", inplace=True)
        self.df_val.set_index("ID", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # given the admission id, get the whole time series
        samples = self.df_before.loc[idx]
        if len(samples.shape) == 1:
            samples = self.df_before.loc[[idx]]

        val_samples = self.df_val.loc[idx]
        if len(val_samples.shape) == 1:
            val_samples = self.df_val.loc[[idx]]
        return {"idx": idx, "truth": val_samples, "samples": samples}


def collate_fn_biclass(batch, num_vars, args):
    if hasattr(args, "device"):
        device = args.device
    else:
        device = torch.device("cpu")
    value_cols = []
    mask_cols = []
    for col in batch[0]["samples"].columns:
        value_cols.append(col.startswith("Value"))
        mask_cols.append(col.startswith("Mask"))

    if args.ts_full == True:
        data_list = []
        truth_list = []
        for b in batch:
            df_new = pd.DataFrame(0.0, index=np.arange(
                args.num_times), columns=b["samples"].columns[1:])
            df_tmp = b["samples"].set_index("Time")
            df_new.loc[df_tmp.index] = df_tmp
            value = df_new.loc[:, value_cols[1:]].values
            mask = df_new.loc[:, mask_cols[1:]].values
            if args.mask_type == "cumsum":
                mask_cum = mask.cumsum(axis=0) * mask
            data_list.append(np.concatenate((value, mask_cum), axis=-1))
            truth_list.append(b["truth"])

        data_batch = torch.tensor(
            np.array(data_list, dtype=np.float32)).to(device).permute(0, 2, 1)

        combined_truth = torch.tensor(np.array(truth_list)).to(device)
        data_dict = {"data": data_batch, "truth": combined_truth, "mask": mask}

    else:
        values_list = []
        masks_list = []
        times_list = []
        len_list = []
        truth_list = []
        for b in batch:
            values_list.append(b["samples"].loc[:, value_cols].values)
            masks_list.append(b["samples"].loc[:, mask_cols].values)
            ts = b["samples"]["Time"].values
            times_list.append(ts)
            len_list.append(len(ts))
            truth_list.append(b["truth"])

        max_len = max(len_list)

        # shape = (batch_size, maximum sequence length, variables)
        combined_values = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
            (max_len-len_t, num_vars), dtype=np.float32)], 0) for values, len_t in zip(values_list, len_list)], 0,)).to(device)

        # shape = (batch_size, maximum sequence length, variables)
        combined_masks = torch.from_numpy(np.stack([np.concatenate([mask, np.zeros(
            (max_len-len_t, num_vars), dtype=np.float32)], 0) for mask, len_t in zip(masks_list, len_list)], 0)).to(device)

        # shape = (batch_size, maximum sequence length)
        combined_times = torch.from_numpy(np.stack([np.concatenate([times, np.zeros(
            max_len-len_t, dtype=np.float32)], 0) for times, len_t in zip(times_list, len_list)], 0,).astype(np.float32)).to(device)

        combined_truth = torch.tensor(np.array(truth_list)).to(device)

        lengths = torch.tensor(len_list).to(device)

        # The first time point should be larger than 0 after data processing
        assert combined_times[:, 0].gt(0).all()

        if args.first_dim == "time_series":
            combined_values = combined_values.permute(1, 0, 2)
            combined_masks = combined_masks.permute(1, 0, 2)
            combined_times = combined_times.permute(1, 0)

        data_dict = {
            "times_in": combined_times,
            "data_in": combined_values,
            "mask_in": combined_masks,
            "truth": combined_truth,
            "lengths": lengths
        }

    return data_dict


def collate_fn_extrap(batch, num_vars, args):
    if hasattr(args, "device"):
        device = args.device
    else:
        device = torch.device("cpu")
    value_cols = []
    mask_cols = []
    for col in batch[0]["samples"].columns:
        value_cols.append(col.startswith("Value"))
        mask_cols.append(col.startswith("Mask"))

    if args.ts_full == True:
        data_list = []
        truth_list = []
        mask_list = []
        for b in batch:
            df_in_new = pd.DataFrame(0.0, index=np.arange(
                args.next_start/args.down_times), columns=b["samples"].columns[1:])
            df_out_new = pd.DataFrame(0.0, index=np.arange(
                args.next_start, args.next_end), columns=b["truth"].columns[1:])
            df_in_tmp = b["samples"].set_index("Time")
            df_out_tmp = b["truth"].set_index("Time")
            df_in_new.loc[df_in_tmp.index] = df_in_tmp
            df_out_new.loc[df_out_tmp.index] = df_out_tmp
            value_in = df_in_new.loc[:, value_cols[1:]].values
            mask_in = df_in_new.loc[:, mask_cols[1:]].values
            value_out = df_out_new.loc[:, value_cols[1:]].values
            mask_out = df_out_new.loc[:, mask_cols[1:]].values
            if args.mask_type == "cumsum":
                mask_in = mask_in.cumsum(axis=0) * mask_in
            data_list.append(np.concatenate((value_in, mask_in), axis=-1))
            truth_list.append(value_out)
            mask_list.append(mask_out)

        combined_data = torch.from_numpy(
            np.array(data_list, dtype=np.float32)).to(device).permute(0, 2, 1).to(device)
        combined_truth = torch.from_numpy(np.array(truth_list)).to(device)
        combined_mask = torch.from_numpy(np.array(mask_list)).to(device)

        data_dict = {"data": combined_data,
                     "truth": combined_truth,
                     "mask": combined_mask}

    else:
        values_in_list = []
        masks_in_list = []
        ts_in_list = []
        len_in_list = []
        values_out_list = []
        masks_out_list = []
        ts_out_list = []
        len_out_list = []
        for b in batch:
            values_in = b["samples"].loc[:, value_cols].values
            masks_in = b["samples"].loc[:, mask_cols].values
            ts_in = b["samples"]["Time"].values
            values_out = b["truth"].loc[:, value_cols].values
            masks_out = b["truth"].loc[:, mask_cols].values
            ts_out = b["truth"]["Time"].values
            values_in_list.append(values_in)
            masks_in_list.append(masks_in)
            ts_in_list.append(ts_in)
            len_in_list.append(len(ts_in))
            values_out_list.append(values_out)
            masks_out_list.append(masks_out)
            ts_out_list.append(ts_out)
            len_out_list.append(len(ts_out))

        max_len_in = max(len_in_list)
        max_len_out = max(len_out_list)

        # shape = (batch_size, maximum sequence length, variables)
        data_in = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
            (max_len_in-len_t, num_vars), dtype=np.float32)], 0) for values, len_t in zip(values_in_list, len_in_list)], 0,)).to(device)

        # shape = (batch_size, maximum sequence length, variables)
        mask_in = torch.from_numpy(np.stack([np.concatenate([mask, np.zeros(
            (max_len_in-len_t, num_vars), dtype=np.float32)], 0) for mask, len_t in zip(masks_in_list, len_in_list)], 0)).to(device)

        # shape = (batch_size, maximum sequence length)
        times_in = torch.from_numpy(np.stack([np.concatenate([times, np.zeros(
            max_len_in-len_t, dtype=np.float32)], 0) for times, len_t in zip(ts_in_list, len_in_list)], 0,).astype(np.float32)).to(device)

        # shape = (batch_size, maximum sequence length, variables)
        data_out = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
            (max_len_out-len_t, num_vars), dtype=np.float32)], 0) for values, len_t in zip(values_out_list, len_out_list)], 0,)).to(device)

        # shape = (batch_size, maximum sequence length, variables)
        mask_out = torch.from_numpy(np.stack([np.concatenate([mask, np.zeros(
            (max_len_out-len_t, num_vars), dtype=np.float32)], 0) for mask, len_t in zip(masks_out_list, len_out_list)], 0)).to(device)

        # shape = (batch_size, maximum sequence length)
        times_out = torch.from_numpy(np.stack([np.concatenate([times, np.zeros(
            max_len_out-len_t, dtype=np.float32)], 0) for times, len_t in zip(ts_out_list, len_out_list)], 0,).astype(np.float32)).to(device)

        lengths_in = torch.tensor(len_in_list).to(device)
        lengths_out = torch.tensor(len_out_list).to(device)

        # The first time point should be larger than 0 after data processing
        assert times_in[:, 0].gt(0).all()

        data_dict = {}
        data_dict["times_in"] = times_in
        data_dict["data_in"] = data_in
        data_dict["mask_in"] = mask_in
        data_dict["lengths_in"] = lengths_in
        data_dict["times_out"] = times_out
        data_dict["data_out"] = data_out
        data_dict["mask_out"] = mask_out
        data_dict["lengths_out"] = lengths_out

        if args.extrap_full == True:
            mask_extrap = torch.zeros_like(times_out, dtype=torch.bool)
            mask_extrap[:, :times_in.size(1)] = times_in.gt(
                torch.zeros_like(times_in))
            data_dict["mask_extrap"] = mask_extrap

    return data_dict
