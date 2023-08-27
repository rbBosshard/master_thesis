import os

import pandas as pd

from ml.src.constants import REMOTE_METADATA_DIR_PATH, MASS_BANK_DIR_PATH, FILE_FORMAT


def get_subset_aeids():
    aeids_path = os.path.join(REMOTE_METADATA_DIR_PATH, 'subset', f"aeids{FILE_FORMAT}")
    aeids = pd.read_parquet(aeids_path)
    return aeids


def get_validation_compounds():
    compounds_safe_and_unsafe_for_validation_path = os.path.join(MASS_BANK_DIR_PATH,
                                                                 f"compounds_safe_and_unsafe_for_validation{FILE_FORMAT}")
    compounds_safe_for_validation_path = os.path.join(MASS_BANK_DIR_PATH,
                                                      f"compounds_safe_for_validation{FILE_FORMAT}")
    compounds_unsafe_for_validation_path = os.path.join(MASS_BANK_DIR_PATH,
                                                        f"compounds_unsafe_for_validation{FILE_FORMAT}")
    compounds_safe_and_unsafe_for_validation = pd.read_parquet(compounds_safe_and_unsafe_for_validation_path)[
        'dsstox_substance_id']
    compounds_safe_for_validation = pd.read_parquet(compounds_safe_for_validation_path)[
        'dsstox_substance_id']
    compounds_unsafe_for_validation = pd.read_parquet(compounds_unsafe_for_validation_path)[
        'dsstox_substance_id']

    return compounds_safe_and_unsafe_for_validation, compounds_safe_for_validation, compounds_unsafe_for_validation
