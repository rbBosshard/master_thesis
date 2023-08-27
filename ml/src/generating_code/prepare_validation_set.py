import os
import pandas as pd
import time

from ml.src.ml_helper import load_config
from ml.src.constants import METADATA_DIR_PATH, MASS_BANK_DIR_PATH

CONFIG = load_config()[0]

print(f'Reading...')

massbank_dtxsid_with_records_path = os.path.join(MASS_BANK_DIR_PATH, f"massbank_dtxsid_with_records.csv")
massbank_dtxsid_with_records_sirius_training_path = os.path.join(MASS_BANK_DIR_PATH, f"massbank_dtxsid_with_records_sirius_training.csv")

massbank_dtxsid_with_records_df = pd.read_csv(massbank_dtxsid_with_records_path)
massbank_dtxsid_with_records_sirius_training_df = pd.read_csv(massbank_dtxsid_with_records_sirius_training_path)

massbank_dtxsid_with_records = massbank_dtxsid_with_records_df['dtxsid']
massbank_dtxsid_with_records_sirius_training = massbank_dtxsid_with_records_sirius_training_df['dtxsid']

massbank_dtxsid_with_records = set(massbank_dtxsid_with_records)
massbank_dtxsid_with_records_sirius_training = set(massbank_dtxsid_with_records_sirius_training)

# Intersection of IDs in both files
compounds_intersection = massbank_dtxsid_with_records.intersection(massbank_dtxsid_with_records_sirius_training)
compounds_safe_for_validation = massbank_dtxsid_with_records.difference(massbank_dtxsid_with_records_sirius_training)
compounds_unsafe_for_validation = massbank_dtxsid_with_records_sirius_training


with open(os.path.join(MASS_BANK_DIR_PATH, 'compounds_count.out'), 'w') as f:
    f.write(f"massbank_dtxsid_with_records: {len(massbank_dtxsid_with_records)} \n")
    f.write(f"massbank_dtxsid_with_records_sirius_training: {len(massbank_dtxsid_with_records_sirius_training)} \n")
    f.write("\n")
    f.write(f"intersection: {len(compounds_intersection)} \n")
    f.write("\n")
    f.write(f"compounds_safe_for_validation: {len(compounds_safe_for_validation)} \n")
    f.write(f"compounds_unsafe_for_validation: {len(compounds_unsafe_for_validation)} \n")

compounds = compounds_safe_for_validation
path = os.path.join(MASS_BANK_DIR_PATH, f"compounds_safe_for_validation{CONFIG['file_format']}")
df = pd.DataFrame({'dsstox_substance_id': list(compounds_safe_for_validation)})
df.to_parquet(path, compression='gzip')
with open(os.path.join(MASS_BANK_DIR_PATH, 'compounds_safe_for_validation.out'), 'w') as f:
    for id_ in compounds:
        f.write(id_ + '\n')

compounds = compounds_unsafe_for_validation
path = os.path.join(MASS_BANK_DIR_PATH, f"compounds_unsafe_for_validation{CONFIG['file_format']}")
df = pd.DataFrame({'dsstox_substance_id': list(compounds_unsafe_for_validation)})
df.to_parquet(path, compression='gzip')
with open(os.path.join(MASS_BANK_DIR_PATH, 'compounds_unsafe_for_validation.out'), 'w') as f:
    for id_ in compounds_unsafe_for_validation:
        f.write(id_ + '\n')

compounds = massbank_dtxsid_with_records
path = os.path.join(MASS_BANK_DIR_PATH, f"compounds_safe_and_unsafe_for_validation{CONFIG['file_format']}")
df = pd.DataFrame({'dsstox_substance_id': list(massbank_dtxsid_with_records)})
df.to_parquet(path, compression='gzip')
with open(os.path.join(MASS_BANK_DIR_PATH, 'compounds_safe_and_unsafe_for_validation.out'), 'w') as f:
    for id_ in massbank_dtxsid_with_records:
        f.write(id_ + '\n')

print("Results written to files.")



