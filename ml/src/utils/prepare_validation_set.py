import os
import pandas as pd

from ml.src.pipeline.constants import MASS_BANK_DIR_PATH, FILE_FORMAT, INPUT_VALIDATION_DIR_PATH


def prepare_validation_set():
    print("Prepare validation set (massbank)")
    massbank_dtxsid_with_records_path = os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_dtxsid_with_records.csv")
    massbank_dtxsid_with_records_sirius_training_path = os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_dtxsid_with_records_sirius_training.csv")

    massbank_dtxsid_with_records_df = pd.read_csv(massbank_dtxsid_with_records_path)
    massbank_dtxsid_with_records_sirius_training_df = pd.read_csv(massbank_dtxsid_with_records_sirius_training_path)

    massbank_dtxsid_with_records = massbank_dtxsid_with_records_df['dtxsid']
    massbank_dtxsid_with_records_sirius_training = massbank_dtxsid_with_records_sirius_training_df['dtxsid']

    massbank_dtxsid_with_records = set(massbank_dtxsid_with_records)
    massbank_dtxsid_with_records_sirius_training = set(massbank_dtxsid_with_records_sirius_training)

    compounds_intersection = massbank_dtxsid_with_records.intersection(massbank_dtxsid_with_records_sirius_training)
    compounds_safe_for_validation = massbank_dtxsid_with_records.difference(massbank_dtxsid_with_records_sirius_training)
    compounds_unsafe_for_validation = massbank_dtxsid_with_records_sirius_training

    with open(os.path.join(MASS_BANK_DIR_PATH, 'compounds_count.out'), 'w') as f:
        f.write(f"massbank_dtxsid_with_records: {len(massbank_dtxsid_with_records)} \n")
        f.write(f"massbank_dtxsid_with_records_sirius_training: {len(massbank_dtxsid_with_records_sirius_training)} \n")
        f.write("\n")
        f.write(f"intersection: {len(compounds_intersection)} \n")
        f.write("\n")
        f.write(f"validation_compounds_safe: {len(compounds_safe_for_validation)} \n")
        f.write(f"validation_compounds_unsafe: {len(compounds_unsafe_for_validation)} \n")

    path = os.path.join(MASS_BANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}")
    df = pd.DataFrame({'dsstox_substance_id': list(compounds_safe_for_validation)})
    df.to_parquet(path, compression='gzip')
    with open(os.path.join(MASS_BANK_DIR_PATH, 'validation_compounds_safe.out'), 'w') as f:
        for compound in compounds_safe_for_validation:
            f.write(compound + '\n')

    path = os.path.join(MASS_BANK_DIR_PATH, f"validation_compounds_unsafe{FILE_FORMAT}")
    df = pd.DataFrame({'dsstox_substance_id': list(compounds_unsafe_for_validation)})
    df.to_parquet(path, compression='gzip')
    with open(os.path.join(MASS_BANK_DIR_PATH, 'validation_compounds_unsafe.out'), 'w') as f:
        for compound in compounds_unsafe_for_validation:
            f.write(compound + '\n')

    path = os.path.join(MASS_BANK_DIR_PATH, f"validation_compounds_safe_and_unsafe{FILE_FORMAT}")
    df = pd.DataFrame({'dsstox_substance_id': list(massbank_dtxsid_with_records)})
    df.to_parquet(path, compression='gzip')
    with open(os.path.join(MASS_BANK_DIR_PATH, 'validation_compounds_safe_and_unsafe.out'), 'w') as f:
        for compound in massbank_dtxsid_with_records:
            f.write(compound + '\n')



