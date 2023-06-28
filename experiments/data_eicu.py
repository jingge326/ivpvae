import pandas as pd

from experiments.data_mimic import DatasetBiClass, DatasetExtrap, filter_tvt, scale_tvt


def load_tvt(args, path_eicu, logger):
    path_processed = path_eicu/'processed'
    data_eicu = pd.read_csv(path_processed/'eicu_data.csv', index_col=0)

    if args.num_samples != -1:
        tvt_ids = pd.DataFrame(data_eicu.index.unique(), columns=['ID']).sample(
            n=args.num_samples, random_state=args.random_state)
        data_tvt = data_eicu.loc[tvt_ids['ID']]
    else:
        data_tvt = data_eicu

    data_train, data_validation, data_test = filter_tvt(
        data_tvt, logger, args)

    return data_train, data_validation, data_test


def get_eicu_tvt_datasets(args, proj_path, logger):
    path_eicu = proj_path/'data/eicu'
    data_train, data_validation, data_test = load_tvt(args, path_eicu, logger)
    data_train, data_validation, data_test = scale_tvt(
        args, data_train, data_validation, data_test, logger)

    if args.ml_task == "biclass":
        label_data = pd.read_csv(
            proj_path/'data/eicu/processed/eicu_labels.csv')
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
