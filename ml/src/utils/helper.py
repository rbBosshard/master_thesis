import os

import numpy as np
import pandas as pd


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


def calculate_binarized_hitcall_statistics(hitcall_infos, aeid, df_aeid, hitcall):
    hitcall_values = df_aeid[f"{hitcall}"]
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


def get_sirius_fingerprints():
    massbank_sirius_df = pd.read_csv(
        os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_from-sirius_fps_pos_curated_20231009_withDTXSID.csv"))
    fps_toxcast_df = pd.read_csv(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"ToxCast_20231006_fingerprints.csv"))
    df = massbank_sirius_df.merge(fps_toxcast_df[['index']], how='inner', left_on='DTXSID', right_on='index')
    cols_to_select = df.filter(regex='^\d+$').columns.to_list() + ['index']  # only fingerprint

    df = df[cols_to_select]
    # df = df.drop_duplicates()
    df = df.groupby("index").mean()
    df = df.round().astype(int)  # Convert the mean values to binary fingerprints again
    df = df.reset_index().rename(columns={'index': 'dsstox_substance_id'})
    return df


def get_guid_dtxsid_mapping():
    global massbank_guid_acc_df, massbank_metadata_df
    # Table with DTXSID and accession column
    massbank_dtxsid_acc_df = pd.read_csv(
        os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_quality_filtered_full_table_20231005.csv"))
    massbank_dtxsid_acc_df = massbank_dtxsid_acc_df.rename(columns={'CH$LINK': 'DTXSID'})
    massbank_dtxsid_acc_df['DTXSID'] = massbank_dtxsid_acc_df['DTXSID'].str.replace('COMPTOX ', '')
    # Table with GUID and accession column
    massbank_guid_acc_df = pd.read_csv(
        os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_smiles_guid_acc_20231005.csv"))
    # Merge both tables on accession column
    massbank_metadata_df = massbank_dtxsid_acc_df.merge(massbank_guid_acc_df, on="accession")
    # Get GUID/DTXSID pairs
    pairs = massbank_metadata_df[['GUID', 'DTXSID']].drop_duplicates()
    # Check if chemical with DTXSID have a unique relationship to GUID
    unique_guid = pairs['GUID'].nunique()  # = 1783
    unique_dtxsid = pairs['DTXSID'].nunique()  # = 1466
    is_one_to_one_relationship = (unique_guid == unique_dtxsid) and (unique_guid == len(pairs))  # False
    print("'GUID'/'DTXSID' is_one_to_one_relationship:", is_one_to_one_relationship)
    return pairs


def csv_to_parquet_converter():
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

    # get_guid_dtxsid_mapping()
    sirius_fingerprints = get_sirius_fingerprints()
    sirius_fingerprints.to_csv(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_massbank_fingerprints.csv"), index=False)
    sirius_fingerprints.to_parquet(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_massbank_fingerprints{FILE_FORMAT}"), compression='gzip')

    unique_chemicals = sirius_fingerprints['dsstox_substance_id'].unique()
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_fingerprints_compounds.out"), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))