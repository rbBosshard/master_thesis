import os

import pandas as pd

from ml.src.pipeline.constants import REMOTE_METADATA_DIR_PATH, MASS_BANK_DIR_PATH, FILE_FORMAT, INPUT_DIR_PATH, \
    OUTPUT_DIR_PATH, INPUT_FINGERPRINTS_DIR_PATH, METADATA_DIR_PATH, METADATA_ALL_DIR_PATH, METADATA_SUBSET_DIR_PATH, \
    VALIDATION_COVERAGE_DIR_PATH, VALIDATION_COVERAGE_PLOTS_DIR_PATH, CONC_DIR_PATH, INPUT_VALIDATION_DIR_PATH, \
    INPUT_ML_DIR_PATH


def get_subset_aeids():
    aeids_path = os.path.join(REMOTE_METADATA_DIR_PATH, 'subset', f"aeids{FILE_FORMAT}")
    aeids = pd.read_parquet(aeids_path)
    return aeids


def get_validation_compounds():
    validation_compounds_safe_and_unsafe = pd.read_parquet(os.path.join(MASS_BANK_DIR_PATH, f"validation_compounds_safe_and_unsafe{FILE_FORMAT}"))['dsstox_substance_id']
    validation_compounds_safe = pd.read_parquet(os.path.join(MASS_BANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}"))['dsstox_substance_id']
    validation_compounds_unsafe = pd.read_parquet(os.path.join(MASS_BANK_DIR_PATH, f"validation_compounds_unsafe{FILE_FORMAT}"))['dsstox_substance_id']
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
    os.makedirs(MASS_BANK_DIR_PATH, exist_ok=True)
    os.makedirs(VALIDATION_COVERAGE_DIR_PATH, exist_ok=True)
    os.makedirs(VALIDATION_COVERAGE_PLOTS_DIR_PATH, exist_ok=True)
    os.makedirs(CONC_DIR_PATH, exist_ok=True)
