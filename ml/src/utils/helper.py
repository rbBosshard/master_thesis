import os

import numpy as np
import pandas as pd
import pubchempy as pcp
import multiprocessing
import joblib

from ml.src.pipeline.constants import REMOTE_METADATA_DIR_PATH, MASSBANK_DIR_PATH, FILE_FORMAT, INPUT_DIR_PATH, \
    OUTPUT_DIR_PATH, INPUT_FINGERPRINTS_DIR_PATH, METADATA_DIR_PATH, METADATA_ALL_DIR_PATH, METADATA_SUBSET_DIR_PATH, \
    VALIDATION_COVERAGE_DIR_PATH, VALIDATION_COVERAGE_PLOTS_DIR_PATH, CONC_DIR_PATH, INPUT_VALIDATION_DIR_PATH, \
    INPUT_ML_DIR_PATH, COMPOUNDS_DIR_PATH, FINGERPRINT_FILE


def get_subset_aeids():
    aeids_path = os.path.join(REMOTE_METADATA_DIR_PATH, 'subset', f"aeids_target_assays{FILE_FORMAT}")
    aeids = pd.read_parquet(aeids_path)
    return aeids


def get_validation_compounds():
    validation_compounds_safe_and_unsafe = pd.read_parquet(os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe_and_unsafe{FILE_FORMAT}"))['dsstox_substance_id']
    validation_compounds_safe = pd.read_parquet(os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}"))['dsstox_substance_id']
    validation_compounds_unsafe = pd.read_parquet(os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_unsafe{FILE_FORMAT}"))['dsstox_substance_id']
    return validation_compounds_safe_and_unsafe, validation_compounds_safe, validation_compounds_unsafe


def calculate_hitcall_statistics(hitcall_infos, aeid, df_aeid):
    hitcall_values = df_aeid['hitcall']
    hitcall_values = (hitcall_values >= 0.5).astype(int)
    total_size = len(hitcall_values)
    num_active = hitcall_values.sum()
    num_inactive = total_size - num_active
    hit_ratio = num_active / total_size
    hitcall_infos[aeid] = {"total_size": total_size, "num_active": num_active, "num_inactive": num_inactive,
                           "hit_ratio": hit_ratio}


def init_directories():
    print("init_directories")
    os.makedirs(INPUT_DIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    os.makedirs(INPUT_ML_DIR_PATH, exist_ok=True)
    os.makedirs(INPUT_FINGERPRINTS_DIR_PATH, exist_ok=True)
    os.makedirs(INPUT_VALIDATION_DIR_PATH, exist_ok=True)
    os.makedirs(METADATA_DIR_PATH, exist_ok=True)
    os.makedirs(METADATA_ALL_DIR_PATH, exist_ok=True)
    os.makedirs(METADATA_SUBSET_DIR_PATH, exist_ok=True)
    os.makedirs(MASSBANK_DIR_PATH, exist_ok=True)
    os.makedirs(VALIDATION_COVERAGE_DIR_PATH, exist_ok=True)
    os.makedirs(VALIDATION_COVERAGE_PLOTS_DIR_PATH, exist_ok=True)
    os.makedirs(CONC_DIR_PATH, exist_ok=True)


def compute_compounds_intersection(directory, compounds, compounds_with_zero_count, compounds_with_fingerprint):
    compounds = set(compounds)
    compounds_path = os.path.join(directory, f"compounds_tested{FILE_FORMAT}")
    pd.DataFrame(compounds, columns=['dsstox_substance_id']).to_parquet(compounds_path, compression='gzip')

    with open(os.path.join(directory, 'compounds_tested.out'), 'w') as f:
        for compound in compounds:
            f.write(str(compound) + '\n')

    with open(os.path.join(directory, 'compounds_absent.out'), 'w') as f:
        for compound in compounds_with_zero_count:
            f.write(str(compound) + '\n')

    intersection = compounds_with_fingerprint.intersection(compounds)
    compounds_not_tested = compounds_with_fingerprint.difference(compounds)
    compounds_without_fingerprint = compounds.difference(compounds_with_fingerprint)

    with open(os.path.join(directory, 'compounds_count.out'), 'w') as f:
        f.write(f"Number of compounds tested: {len(compounds)} \n")
        f.write(f"Number of compounds with fingerprint available: {len(compounds_with_fingerprint)} \n")
        f.write(f"Number of compounds tested and fingerprint available: {len(intersection)} \n")
        f.write(f"Number of compounds tested and no fingerprint available: {len(compounds_without_fingerprint)} \n")
        f.write(f"Number of compounds not tested but fingerprint available: {len(compounds_not_tested)} \n")

    dest_path = os.path.join(directory, f'compounds_tested_with_fingerprint{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(intersection)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(directory, f'compounds_tested_with_fingerprint.out'), 'w') as f:
        for compound in intersection:
            f.write(compound + '\n')

    dest_path = os.path.join(directory, f'compounds_not_tested{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(compounds_not_tested)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(directory, f'compounds_not_tested.out'), 'w') as f:
        for compound in compounds_not_tested:
            f.write(compound + '\n')

    dest_path = os.path.join(directory, f'compounds_without_fingerprint{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(compounds_without_fingerprint)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(directory, f'compounds_without_fingerprint.out'), 'w') as f:
        for compound in compounds_without_fingerprint:
            f.write(compound + '\n')

    return compounds_without_fingerprint


def csv_to_parquet_converter():
    # print("Get mapping: GUID -> DTXSID from smiles (using pubchempy)")
    # src_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"massbank_smiles_guid_acc_20231005.csv")
    # df = pd.read_csv(src_path)
    # df = df.drop_duplicates()
    #
    #
    # # Initialize an empty dictionary to store the mapping
    # guid_to_dtxsid = {}
    #
    # def apply_get_dtxsid_parallel(row):
    #     smiles = row['CH$SMILES']
    #     guid = row['GUID']
    #     c = pcp.get_compounds(smiles, 'smiles')
    #     synonyms = c[0].synonyms
    #     dtxsid_values = [item for item in synonyms if item.startswith("DTXSID")]
    #     num_compounds = len(dtxsid_values)
    #     if num_compounds == 1:
    #         print("Unique DTXSID found")
    #         dtxsid = dtxsid_values[0]
    #     elif num_compounds > 2:
    #         print("No unique DTXSID found")
    #         dtxsid = dtxsid_values[0]
    #     else:
    #         dtxsid = None
    #         print("No DTXSID found")
    #
    # result = joblib.Parallel(n_jobs=-1)(joblib.delayed(apply_get_dtxsid_parallel)(row) for _, row in df.iterrows())

    print("Preprocess fingerprint from structure input file")
    src_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}.csv")
    dest_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}{FILE_FORMAT}")

    df = pd.read_csv(src_path)
    # Old: Skip the first 3 columns (Unnamed: 0, relativeIndex, absoluteIndex) and transpose the dataframe
    # df = df.iloc[:, 3:].T
    # data = df.iloc[1:].values.astype(int)
    # index = df.index[1:]
    # columns = df.iloc[0]

    # New: Skip  first column (Unnamed: 0), drop duplicated index.1
    if 'index.1' in df.columns:
        df = df.drop('index.1', axis=1)
    df = df.iloc[:, 1:]
    index = df['index']
    data = df.iloc[:, 1:].values.astype(np.uint8)
    columns = df.columns[1:]

    df = pd.DataFrame(data=data, index=index, columns=columns).reset_index()
    df = df.rename(columns={"index": "dsstox_substance_id"})
    df.to_parquet(dest_path, compression='gzip')
    df.sample(n=100, replace=False).to_csv(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"test_sample.csv"), index=False)

    unique_chemicals = df['dsstox_substance_id'].unique()
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}_compounds.out"), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))



