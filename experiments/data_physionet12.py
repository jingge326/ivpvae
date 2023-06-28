import os

import tarfile
import shutil
import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url
from experiments.data_mimic import DatasetBiClass, DatasetExtrap, filter_tvt, scale_tvt


def download_and_process_p12(path_p12):
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]
    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']
    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC']
    params_dict = {k: i for i, k in enumerate(params)}

    raw_folder = path_p12/"raw"
    processed_folder = path_p12/"processed"
    os.makedirs(raw_folder, exist_ok=True)

    # Download outcome data
    list_lab_df = []
    for url in outcome_urls:
        filename = url.rpartition('/')[2]
        download_url(url, raw_folder, filename, None)
        list_lab_df.append(pd.read_csv(raw_folder/filename, header=0).rename(
            columns={"RecordID": "ID", "In-hospital_death": "labels"})[["ID", "labels"]])

    labels_df = pd.concat(list_lab_df)

    os.makedirs(processed_folder, exist_ok=True)
    labels_df.to_csv(processed_folder/"p12_labels.csv", index=False)

    list_data_df = []
    for url in urls:
        filename = url.rpartition('/')[2]
        download_url(url, raw_folder, filename, None)
        tar = tarfile.open(os.path.join(raw_folder, filename), 'r:gz')
        tar.extractall(raw_folder)
        tar.close()
        print('Processing {}...'.format(filename))

        dirname = os.path.join(raw_folder, filename.split('.')[0])
        files_all = [fname.split('.')[0] for fname in os.listdir(dirname)]
        files_selected = list(set(files_all) & set(map(str, labels_df["ID"])))

        list_ids_dup = []
        list_vals = []
        list_masks = []
        list_times = []

        if len(files_selected) == 0:
            continue

        for record_id in files_selected:
            prev_time = -1
            num_obs = []
            with open(os.path.join(dirname, record_id + ".txt")) as f:
                for l in f.readlines()[1:]:
                    time, param, val = l.split(',')
                    # Time in minutes
                    time = float(time.split(':')[
                        0])*60 + float(time.split(':')[1])

                    if time != prev_time:
                        list_times.append(time)
                        list_vals.append(np.zeros(len(params)))
                        list_masks.append(np.zeros(len(params)))
                        num_obs.append(np.zeros(len(params)))
                        list_ids_dup.append(record_id)
                        prev_time = time

                    if param in params_dict:
                        n_observations = num_obs[-1][params_dict[param]]
                        # integration by average
                        if n_observations > 0:
                            prev_val = list_vals[-1][params_dict[param]]
                            new_val = (prev_val * n_observations +
                                       float(val)) / (n_observations + 1)
                            list_vals[-1][params_dict[param]] = new_val
                        else:
                            list_vals[-1][params_dict[param]] = float(val)
                        list_masks[-1][params_dict[param]] = 1
                        num_obs[-1][params_dict[param]] += 1
                    else:
                        assert param == 'RecordID', 'Unexpected param {}'.format(
                            param)

        arr_values = np.stack(list_vals, axis=0)
        arr_masks = np.stack(list_masks, axis=0)
        df_times = pd.DataFrame(list_times, columns=['Time'])

        df_values = pd.DataFrame(arr_values, columns=[
            'Value_'+str(i) for i in params_dict.values()])
        df_mask = pd.DataFrame(
            arr_masks, columns=['Mask_'+str(i) for i in params_dict.values()])

        df_p12 = pd.concat([pd.DataFrame(list_ids_dup, columns=[
            'ID']), df_times, df_values, df_mask], axis=1)
        list_data_df.append(df_p12)

    df_p12_data = pd.concat(list_data_df)
    df_p12_data.to_csv(processed_folder/'p12_data.csv', index=False)


def load_tvt(args, path_p12, logger):
    path_processed = path_p12/"processed"
    path_raw = path_p12/"raw"
    if os.path.exists(path_processed/'p12_data.csv') and os.path.exists(path_processed/'p12_labels.csv'):
        pass
    else:
        if os.path.exists(path_raw):
            shutil.rmtree(path_raw)
        if os.path.exists(path_processed):
            shutil.rmtree(path_processed)
        download_and_process_p12(path_p12)

    data_tvt = pd.read_csv(path_processed/'p12_data.csv', index_col=0)

    data_train, data_validation, data_test = filter_tvt(
        data_tvt, logger, args)

    return data_train, data_validation, data_test


def get_p12_tvt_datasets(args, proj_path, logger):
    path_p12 = proj_path/'data'/'PhysioNet12'
    data_train, data_validation, data_test = load_tvt(args, path_p12, logger)
    data_train, data_validation, data_test = scale_tvt(
        args, data_train, data_validation, data_test, logger)

    if args.ml_task == "biclass":
        label_data = pd.read_csv(
            proj_path/'data'/'PhysioNet12'/'processed'/'p12_labels.csv')
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
        raise ValueError("Unknown ML mode!")

    return train, val, test
