import os

import numpy as np
import pandas as pd
import json

from ml.src.pipeline.constants import REMOTE_DATA_DIR_PATH, FILE_FORMAT, METADATA_SUBSET_DIR_PATH

import matplotlib.pyplot as plt

from ml.src.utils.helper import get_subset_aeids

print(f'Reading...')
dest_folder_path = os.path.join(METADATA_SUBSET_DIR_PATH, 'conc')
os.makedirs(dest_folder_path, exist_ok=True)
dest_path = os.path.join(dest_folder_path, f"{0}{FILE_FORMAT}")

metrics_to_plot = ['num_points', 'num_groups', 'num_replicates', 'range_min', 'range_max']
metric_rename_dict = {
    'num_points': '# Datapoints',
    'num_groups': '# Concentration Groups',
    'num_replicates': '# Replicates',
    'range_min': 'Lowest Concentration',
    'range_max': 'Highest Concentration',
}

if not os.path.exists(dest_path):
    src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'merged', 'output', f"0{FILE_FORMAT}")
    df = pd.read_parquet(src_path)
    df = df[['aeid', 'dsstox_substance_id', 'conc']]
    df['conc'] = df['conc'].apply(json.loads)


    def calculate_metrics(conc):
        num_groups = len(set(conc))
        num_replicates = len(conc) // num_groups
        num_points = len(conc)
        min_val = np.min(conc)
        max_val = np.max(conc)
        return pd.Series([num_points, num_groups, num_replicates, min_val, max_val], index=metrics_to_plot)


    df[metrics_to_plot] = df['conc'].apply(calculate_metrics)
    df = df.drop(columns=['conc'])
    df.to_parquet(dest_path, compression='gzip')

else:
    df = pd.read_parquet(dest_path)

print(f"Shape: {df.shape}")


dest_folder_path = os.path.join(METADATA_SUBSET_DIR_PATH, 'conc')
os.makedirs(dest_folder_path)
dest_path = os.path.join(dest_folder_path, f"{0}{FILE_FORMAT}")

df.to_parquet(dest_path, compression='gzip')

aeids = get_subset_aeids()['aeid']
for aeid in aeids:
    df_aeid = df[df['aeid'] == aeid]
    df_aeid = df_aeid[['dsstox_substance_id', 'conc']]
    dest_path = os.path.join(dest_folder_path, f"{aeid}{FILE_FORMAT}")
    df_aeid.to_parquet(dest_path, compression='gzip')
