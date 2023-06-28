import os
import argparse
from pathlib import Path
import eicu_utils


def data_extraction_root(args):
    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    eicu_dir = os.path.join(args.eicu_dir)
    if not os.path.exists(eicu_dir):
        os.mkdir(eicu_dir)

    patients = eicu_utils.read_patients_table(args.eicu_dir, args.output_dir)
    stay_id = eicu_utils.cohort_stay_id(patients)

    eicu_utils.break_up_stays_by_unit_stay(
        patients, args.output_dir, stayid=stay_id, verbose=1)
    del patients

    print("reading lab table")
    lab = eicu_utils.read_lab_table(args.eicu_dir)
    eicu_utils.break_up_lab_by_unit_stay(
        lab, args.output_dir, stayid=stay_id, verbose=1)
    del lab

    print("reading nurseCharting table, might take some time")
    nc = eicu_utils.read_nc_table(args.eicu_dir)
    eicu_utils.break_up_stays_by_unit_stay_nc(
        nc, args.output_dir, stayid=stay_id, verbose=1)
    del nc

    # Write the timeseries data into folders
    eicu_utils.extract_time_series(args.output_dir, drop_outliers=True)

    # eicu_utils.delete_wo_timeseries(args.output_dir)
    # Write all the data into one dataframe
    eicu_utils.all_df_into_one_df(args.output_dir)


path_proj = Path(__file__).parents[2]
path_raw = path_proj/"data/eicu/raw/eicu-2.0"
path_extr = path_proj/"data/eicu/extracted"


def main():
    parser = argparse.ArgumentParser(description="Create data for all tasks")
    parser.add_argument('--eicu_dir', type=str,
                        default=path_raw, help="Path to eICU dataset")
    parser.add_argument('--output_dir', type=str, default=path_extr,
                        help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.eicu_dir):
        os.makedirs(args.eicu_dir)
    data_extraction_root(args)


if __name__ == '__main__':
    main()
