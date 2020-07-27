import os

import datetime as dt
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm

pd.set_option("max.rows", 10)
pd.set_option("max.columns", 50)
pd.set_option("display.width", 1000)

CSV_SUBSET = {"patients", "encounters"}


def main(args):
    dfs = get_dfs(synthea_path=args.synthea_path)

    patients: pd.DataFrame
    encounters: pd.DataFrame
    patients, encounters = dfs["patients"], dfs["encounters"]

    print("Unique patients: {}".format(len(patients["Id"].unique())))
    print("Unique encounters: {}".format(len(encounters["Id"].unique())))

    fmt_str = r"%Y-%m-%d"

    print("Converting birthdates to timestamps.")
    patient_birthdates = patients["BIRTHDATE"]
    patient_birthdates = [
        dt.datetime.strptime(x, fmt_str) for x in tqdm(patient_birthdates)]
    patient_birthdates = [x.timestamp() for x in patient_birthdates]
    patients["birthdate_ts"] = patient_birthdates

    fmt_str = r"%Y-%m-%dT%H:%M:%SZ"

    print("Converting encounter datetimes to timestamps.")
    encounter_datetimes = encounters["START"]
    encounter_datetimes = [
        dt.datetime.strptime(x, fmt_str) for x in tqdm(encounter_datetimes)]
    encounter_datetimes = [x.timestamp() for x in encounter_datetimes]
    encounters["start_ts"] = encounter_datetimes

    patients_subset = patients[["Id", "birthdate_ts", "BIRTHDATE"]]
    patients_subset = patients_subset.rename(columns={"Id": "PATIENT"})

    encounters_subset = encounters[["Id", "PATIENT", "start_ts", "START"]]

    encounters_patients = pd.merge(encounters_subset, patients_subset)
    problem_idx = (encounters_patients.start_ts <
                   encounters_patients.birthdate_ts)

    print("Found {} encounters with issues.".format(sum(problem_idx)))

    print("Issues per patient:")
    encounters_patients = encounters_patients[problem_idx]
    print(encounters_patients.groupby("PATIENT").count()["Id"])

    print()
    print("Specific issues:")

    for k, v in encounters_patients.groupby("PATIENT"):
        print("PATIENT: {}".format(k))
        print("BIRTHDATE: {}".format(list(v["BIRTHDATE"])[0]))
        for enc_id, enc_date in zip(v["Id"], v["START"]):
            print("    ENCOUNTER_ID: {} ENCOUNTER_DATE: {}".format(
                enc_id, enc_date))
        print()


def get_dfs(synthea_path):
    csv_files = os.listdir(synthea_path)
    csv_files = [x for x in csv_files if x in [y + ".csv" for y in CSV_SUBSET]]
    dfs = {
        f[:-len(".csv")]: pd.read_csv(os.path.join(synthea_path, f))
        for f in csv_files}
    return dfs


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--synthea-path", type=str,
                        default=None,
                        help="The directory where Synthea csvs are.")
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    parsed_args = parse_args()
    main(args=parsed_args)
