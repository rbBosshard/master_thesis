import os
import pandas as pd

from ml.src.utils.helper import get_subset_aeids, calculate_hitcall_statistics
from ml.src.pipeline.constants import REMOTE_DATA_DIR_PATH, INPUT_DIR_PATH, FILE_FORMAT, METADATA_SUBSET_DIR_PATH, \
    INPUT_ML_DIR_PATH


def extract_ml_relevant_pytcpl_output():
    print("Get output of pytcpl pipeline")
    src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'merged', 'output', f"0{FILE_FORMAT}")
    df = pd.read_parquet(src_path)[['aeid', 'dsstox_substance_id', 'hitcall']]

    dest_path = os.path.join(INPUT_ML_DIR_PATH, f"{0}{FILE_FORMAT}")
    df.to_parquet(dest_path, compression='gzip')

    # Get set of all compounds tested
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_without_fingerprint{FILE_FORMAT}")
    compounds_tested_without_fingerprint = pd.read_parquet(compounds_path)['dsstox_substance_id']

    aeids = get_subset_aeids()['aeid']
    hitcall_infos = {}

    for aeid in aeids:
        df_aeid = df[df['aeid'] == aeid]
        df_aeid = df_aeid[['dsstox_substance_id', 'hitcall']]
        df_aeid = df_aeid[~df_aeid['dsstox_substance_id'].isin(compounds_tested_without_fingerprint)]

        dest_path = os.path.join(INPUT_ML_DIR_PATH, f"{aeid}{FILE_FORMAT}")
        df_aeid.to_parquet(dest_path, compression='gzip')

        calculate_hitcall_statistics(hitcall_infos, aeid, df_aeid)

    hitcall_infos_df = pd.DataFrame(hitcall_infos, index=['total_size', 'num_active', 'num_inactive', 'hit_ratio']).T
    hitcall_infos_df = hitcall_infos_df.reset_index().rename(columns={'index': 'aeid'})
    hitcall_infos_df.to_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"hitcall_infos{FILE_FORMAT}"), compression='gzip')

