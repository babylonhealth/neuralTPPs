import datetime as dt
import json
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

pd.set_option("max.rows", 10)
pd.set_option("max.columns", 20)
pd.set_option("display.width", 200)

# SYNTHEA_SIMULATION_DATE = "2020-02-12 14:18"
SYNTHEA_SIMULATION_DATE = "2020-02-12 14:00"
SYNTHEA_SIMULATION_DATE_TS = dt.datetime.strptime(
    "2020-02-12 14:00", r"%Y-%m-%d %H:%M")
SYNTHEA_SIMULATION_DATE_TS = dt.datetime(
    year=SYNTHEA_SIMULATION_DATE_TS.year,
    month=SYNTHEA_SIMULATION_DATE_TS.month,
    day=SYNTHEA_SIMULATION_DATE_TS.day,
    hour=SYNTHEA_SIMULATION_DATE_TS.hour,
    minute=SYNTHEA_SIMULATION_DATE_TS.minute).timestamp()
CSV_SUBSET = {"patients", "conditions", "medications", "encounters"}


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    # Run configuration
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--ratio-train", type=float, default=0.6,
                        help="Ratio between train and total size.")
    parser.add_argument("--ratio-val", type=float, default=0.8,
                        help="Ratio between train+val and total size.")
    parser.add_argument("--synthea-path", type=str,
                        default='~/neural-tpps/rawdata/ear_infection',
                        help="The directory where synthea is saved")
    parser.add_argument("--save-path", type=str,
                        default='~/neural-tpps/data/synthea/ear_infection',
                        help="Path where the preprocessed data is saved.")
    args, _ = parser.parse_known_args()
    return args


def prepare_synthea_csvs(synthea_path):
    csv_files = os.listdir(synthea_path)
    csv_files = [x for x in csv_files if x in [y + ".csv" for y in CSV_SUBSET]]

    print("Loading data.")
    dfs = {
        f[:-len(".csv")]: pd.read_csv(os.path.join(synthea_path, f))
        for f in tqdm(csv_files)}
    # >>> pprint({k: len(v) for k, v in dfs.items()})
    # {'allergies': 0,
    #  'careplans': 0,
    #  'conditions': 15162,
    #  'encounters': 404011,
    #  'imaging_studies': 0,
    #  'immunizations': 470900,
    #  'medications': 28202,
    #  'observations': 1265370,
    #  'organizations': 4282,
    #  'patients': 10282,
    #  'payer_transitions': 10232,
    #  'payers': 10,
    #  'procedures': 5410,
    #  'providers': 25320}

    # important cols to keep
    dfs["conditions"] = dfs["conditions"][
        ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]]
    dfs["conditions"].columns = [
        "CONDITION_" + x if x not in {"PATIENT", "ENCOUNTER"} else x
        for x in dfs["conditions"].columns]
    dfs["conditions"]["CONDITION_CODE"] = dfs["conditions"][
        "CONDITION_CODE"].apply(lambda x: "SNOMED-CT_{}".format(x))

    dfs["encounters"] = dfs["encounters"][
        ["Id", "START", "STOP", "PATIENT", "CODE", "DESCRIPTION"]]
    dfs["encounters"].columns = [
        "ENCOUNTER_" + x if x not in {"Id", "PATIENT"} else x
        for x in dfs["encounters"].columns]
    dfs["encounters"] = dfs["encounters"].rename(columns={"Id": "ENCOUNTER"})
    dfs["encounters"]["ENCOUNTER_CODE"] = dfs["encounters"][
        "ENCOUNTER_CODE"].apply(lambda x: "SNOMED-CT_{}".format(x))
    #
    # dfs["immunizations"] = dfs["immunizations"][
    #     ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]]
    # dfs["immunizations"].columns = [
    #     "IMMUNIZATION_" + x if x not in {"PATIENT", "ENCOUNTER"} else x
    #     for x in dfs["immunizations"].columns]

    dfs["medications"] = dfs["medications"][
        ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]]
    dfs["medications"].columns = [
        "MEDICATION_" + x if x not in {"PATIENT", "ENCOUNTER"} else x
        for x in dfs["medications"].columns]
    dfs["medications"]["MEDICATION_CODE"] = dfs["medications"][
        "MEDICATION_CODE"].apply(lambda x: "RxNorm_{}".format(x))

    dfs["patients"] = dfs["patients"][["Id", "BIRTHDATE", "DEATHDATE"]]
    dfs["patients"] = dfs["patients"].rename(columns={"Id": "PATIENT"})
    #
    # dfs["procedures"] = dfs["procedures"][
    #     ["DATE", "PATIENT", "ENCOUNTER", "CODE",
    #      "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"]]
    # dfs["procedures"].columns = [
    #     "PROCEDURE_" + x if x not in {"PATIENT", "ENCOUNTER"} else x
    #     for x in dfs["procedures"].columns]

    return dfs


def main(
        seed: int,
        ratio_train: float,
        ratio_val: float,
        synthea_path: str,
        save_path: str):
    synthea_path = os.path.expanduser(synthea_path)
    save_path = os.path.expanduser(save_path)

    dfs = prepare_synthea_csvs(synthea_path)

    # To speed things up, keep only needed encounters
    encounters_to_keep = set(
        dfs["conditions"]["ENCOUNTER"].values).union(
        set(dfs["medications"]["ENCOUNTER"].values))
    dfs["encounters"] = dfs["encounters"][
        [x in encounters_to_keep for x in dfs["encounters"]["ENCOUNTER"]]]

    # Get unique patients
    unique_patients = pd.unique(dfs["patients"]["PATIENT"])
    print("Unique patients: {}.".format(len(unique_patients)))

    # Get map from codes to unique ids for our classes
    all_code_instances = np.concatenate([
        dfs["encounters"]["ENCOUNTER_CODE"],
        dfs["conditions"]["CONDITION_CODE"],
        dfs["medications"]["MEDICATION_CODE"]
    ])
    unique_codes = np.unique(all_code_instances)
    unique_codes.sort()

    codes_to_int = {k: i for i, k in enumerate(unique_codes)}
    int_to_codes = {v: k for k, v in codes_to_int.items()}

    codes_to_names = dict()
    for k, v in zip(dfs["encounters"]["ENCOUNTER_CODE"],
                    dfs["encounters"]["ENCOUNTER_DESCRIPTION"]):
        codes_to_names[k] = v
    for k, v in zip(dfs["conditions"]["CONDITION_CODE"],
                    dfs["conditions"]["CONDITION_DESCRIPTION"]):
        codes_to_names[k] = v
    for k, v in zip(dfs["medications"]["MEDICATION_CODE"],
                    dfs["medications"]["MEDICATION_DESCRIPTION"]):
        codes_to_names[k] = v
    names_to_codes = {v: k for k, v in codes_to_names.items()}

    fmt_str = r"%Y-%m-%d"

    def to_timestamp(x):
        if not isinstance(x, str):
            return x
        dtf = dt.datetime.strptime(x, fmt_str)
        return dt.datetime(
            year=dtf.year,
            month=dtf.month,
            day=dtf.day,
        ).timestamp()

    print("Processing patients.")
    all_patients = {k: dict() for k in unique_patients}
    for r in tqdm(list(dfs["patients"].itertuples())):
        all_patients[r.PATIENT] = {
            "patient_id": r.PATIENT,
            "birthdate": r.BIRTHDATE,
            "deathdate": r.DEATHDATE,
            "birthdate_timestamp": to_timestamp(r.BIRTHDATE),
            "deathdate_timestamp": to_timestamp(r.DEATHDATE)}
    
    for k, v in all_patients.items():
        still_living = np.isnan(v["deathdate_timestamp"])

        if still_living:
            v["window_end_timestamp"] = SYNTHEA_SIMULATION_DATE_TS
        else:
            v["window_end_timestamp"] = v["deathdate_timestamp"]

    dfs["encounters"]["ENCOUNTER_START_TS"] = dfs["encounters"][
        "ENCOUNTER_START"].copy()

    fmt_str = r"%Y-%m-%dT%H:%M:%SZ"

    def to_timestamp(x):
        dtf = dt.datetime.strptime(x, fmt_str)
        return dt.datetime(
            year=dtf.year,
            month=dtf.month,
            day=dtf.day,
            hour=dtf.hour              # ,
            # Hourly granularity is probably fine for now
            # minute=dtf.minute,
            # second=dtf.second
        ).timestamp()

    # Changes times to timestamps
    dfs["encounters"]["ENCOUNTER_START_TS"] = dfs[
        "encounters"]["ENCOUNTER_START_TS"].apply(to_timestamp)

    dfs["encounters"]["AGE_TS"] = dfs["encounters"].apply(
        lambda x: (x.ENCOUNTER_START_TS -
                   all_patients[x.PATIENT]["birthdate_timestamp"]),
        axis=1)

    # Normalise
    secs_in_hour = 60 * 60
    dfs["encounters"]["AGE_TS"] = dfs["encounters"]["AGE_TS"] / secs_in_hour

    # Find patients with negative ages at encounters
    rows_neg_age = dfs["encounters"][dfs["encounters"]["AGE_TS"] < 0]
    print("Found {} encounters where the patient is negative in age!".format(
        rows_neg_age.shape[0]))
    patients_to_remove = set(rows_neg_age.PATIENT)
    print("Removing {} patients.".format(len(patients_to_remove)))

    dfs = {
        k: v.iloc[[x not in patients_to_remove
                   for x in v.PATIENT]] for k, v in dfs.items()}

    assert dfs["encounters"]["AGE_TS"].min() >= 0

    all_encounters = {k: dict() for k in unique_patients}
    print("Processing encounters.")
    for r in tqdm(list(dfs["encounters"].itertuples())):
        all_encounters[r.PATIENT][r.ENCOUNTER] = {
            "time": r.AGE_TS,
            "labels": [codes_to_int[r.ENCOUNTER_CODE]],
            "encounter": {
                "code": r.ENCOUNTER_CODE,
                "description": r.ENCOUNTER_DESCRIPTION,
                "start": r.ENCOUNTER_START,
                "stop": r.ENCOUNTER_STOP},
            "conditions": list(),
            "medications": list()}

    spec_encounters = {k: dict() for k in unique_patients}
    print("Processing conditions.")
    for r in tqdm(list(dfs["conditions"].itertuples())):
        if r.ENCOUNTER not in spec_encounters[r.PATIENT]:
            spec_encounters[r.PATIENT][r.ENCOUNTER] = all_encounters[
                r.PATIENT][r.ENCOUNTER]
        spec_encounters[r.PATIENT][r.ENCOUNTER]["labels"].append(
            codes_to_int[r.CONDITION_CODE])
        spec_encounters[r.PATIENT][r.ENCOUNTER]["conditions"].append(
            {"code": r.CONDITION_CODE,
             "description": r.CONDITION_DESCRIPTION,
             "start": r.CONDITION_START,
             "stop": r.CONDITION_STOP})

    print("Processing medications.")
    for r in tqdm(list(dfs["medications"].itertuples())):
        if r.ENCOUNTER not in spec_encounters[r.PATIENT]:
            spec_encounters[r.PATIENT][r.ENCOUNTER] = all_encounters[
                r.PATIENT][r.ENCOUNTER]
        spec_encounters[r.PATIENT][r.ENCOUNTER]["labels"].append(
            codes_to_int[r.MEDICATION_CODE])
        spec_encounters[r.PATIENT][r.ENCOUNTER]["medications"].append(
            {"code": r.MEDICATION_CODE,
             "description": r.MEDICATION_DESCRIPTION,
             "start": r.MEDICATION_START,
             "stop": r.MEDICATION_STOP})

    # Make sure that all labels are unique and add on the encounter id
    for pat_encs in spec_encounters.values():
        for enc_id, enc in pat_encs.items():
            enc["labels"] = list(set(enc["labels"]))
            enc["encounter_id"] = enc_id

    # Listify the encounters and (re)sort by time
    for k in spec_encounters:
        spec_encounters[k] = list(spec_encounters[k].values())
        if len(spec_encounters[k]) > 0:
            spec_encounters[k] = sorted(
                spec_encounters[k], key=lambda x: x["time"])
        # Cut sequences longer than 400
        spec_encounters[k] = spec_encounters[k][:400]

    patient_records = [
        {"sequence_id": pat_id,
         "events": pat_encs,
         "metadata": all_patients[pat_id]}
        for pat_id, pat_encs in spec_encounters.items()]

    np.random.seed(seed=seed)
    np.random.shuffle(patient_records)
    n_patients = len(patient_records)
    train_idx = int(ratio_train * n_patients)
    val_idx = int(ratio_val * n_patients)

    data_train = patient_records[:train_idx]
    data_valid = patient_records[train_idx:val_idx]
    data_test = patient_records[val_idx:]

    # Save data
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_dict = {
        "train": data_train,
        "val": data_valid,
        "test": data_test}

    for key, value in save_dict.items():
        with open(os.path.join(save_path, key + ".json"), "w") as h:
            h.write(
                '[' + ',\n'.join(json.dumps(i) for i in value) + ']\n')

    args_dict = {
        "seed": seed,
        "window": None,
        "train_size": len(data_train),
        "val_size": len(data_valid),
        "test_size": len(data_test),
        "ratio_train": ratio_train,
        "ratio_val": ratio_val,
        "marks": len(codes_to_int)}
    with open(os.path.join(save_path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    with open(os.path.join(save_path, 'codes_to_int.json'), 'w') as fp:
        json.dump(codes_to_int, fp)
    with open(os.path.join(save_path, 'int_to_codes.json'), 'w') as fp:
        json.dump(int_to_codes, fp)
    with open(os.path.join(save_path, 'codes_to_names.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(save_path, 'names_to_codes.json'), 'w') as fp:
        json.dump(names_to_codes, fp)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(
        seed=parsed_args.seed,
        ratio_train=parsed_args.ratio_train,
        ratio_val=parsed_args.ratio_val,
        synthea_path=parsed_args.synthea_path,
        save_path=parsed_args.save_path)
