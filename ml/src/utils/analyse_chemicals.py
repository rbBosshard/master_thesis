import os
import pandas as pd
from ml.src.pipeline.constants import FINGERPRINT_FILE, METADATA_DIR_PATH, FILE_FORMAT, COMPOUNDS_DIR_PATH, \
    INPUT_FINGERPRINTS_DIR_PATH


def analyse_chemicals():
    print("Get intersection: compounds with fingerprint from structure & compounds tested in assay endpoints")
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}_compounds.out"), 'r') as f:
        compounds_with_fingerprint = set(line.strip() for line in f)

    with open(os.path.join(METADATA_DIR_PATH, "all", f"compounds_tested.out"), 'r') as f:
        compounds_tested = set(line.strip() for line in f)
        with open(os.path.join(METADATA_DIR_PATH, "all", f"compounds_tested.out"), 'w') as dest_file:
            for compound in compounds_tested:
                dest_file.write(compound + '\n')

    intersection = compounds_with_fingerprint.intersection(compounds_tested)
    compounds_not_tested = compounds_with_fingerprint.difference(compounds_tested)
    compounds_without_fingerprint = compounds_tested.difference(compounds_with_fingerprint)

    os.makedirs(COMPOUNDS_DIR_PATH, exist_ok=True)
    with open(os.path.join(COMPOUNDS_DIR_PATH, 'compounds_count.out'), 'w') as f:
        f.write(f"Number of compounds tested: {len(compounds_tested)} \n")
        f.write(f"Number of compounds with fingerprint available: {len(compounds_with_fingerprint)} \n")
        f.write(f"Number of compounds tested and fingerprint available: {len(intersection)} \n")
        f.write(f"Number of compounds tested and no fingerprint available: {len(compounds_without_fingerprint)} \n")
        f.write(f"Number of compounds not tested but fingerprint available: {len(compounds_not_tested)} \n")

    dest_path = os.path.join(COMPOUNDS_DIR_PATH, f'compounds_tested_with_fingerprint{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(intersection)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(COMPOUNDS_DIR_PATH, f'compounds_tested_with_fingerprint.out'), 'w') as f:
        for compound in intersection:
            f.write(compound + '\n')

    dest_path = os.path.join(COMPOUNDS_DIR_PATH, f'compounds_not_tested{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(compounds_not_tested)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(COMPOUNDS_DIR_PATH, f'compounds_not_tested.out'), 'w') as f:
        for compound in compounds_not_tested:
            f.write(compound + '\n')

    dest_path = os.path.join(COMPOUNDS_DIR_PATH, f'compounds_without_fingerprint{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(compounds_without_fingerprint)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(COMPOUNDS_DIR_PATH, f'compounds_without_fingerprint.out'), 'w') as f:
        for compound in compounds_without_fingerprint:
            f.write(compound + '\n')
