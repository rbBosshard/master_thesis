import os
import pandas as pd

from ml.src.utils.helper import get_subset_aeids
from ml.src.pipeline.constants import REMOTE_DATA_DIR_PATH, INPUT_DIR_PATH, FILE_FORMAT, METADATA_SUBSET_DIR_PATH

print(f'Reading...')
src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'merged', 'output', f"0{FILE_FORMAT}")
df = pd.read_parquet(src_path)[['aeid', 'dsstox_substance_id', 'hitcall']]
dest_path = os.path.join(INPUT_DIR_PATH, f"{0}{FILE_FORMAT}")
df.to_parquet(dest_path, compression='gzip')

aeids = get_subset_aeids()['aeid']
hitcall_infos = {}
for aeid in aeids:
    df_aeid = df[df['aeid'] == aeid]
    df_aeid = df_aeid[['dsstox_substance_id', 'hitcall']]

    dest_path = os.path.join(INPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")
    df_aeid.to_parquet(dest_path, compression='gzip')

    # hitcall_infos
    hitcall_values = df_aeid['hitcall']
    total_size = len(hitcall_values)
    num_active = hitcall_values.sum()
    num_inactive = total_size - num_active
    hit_ratio = num_active / total_size
    hitcall_infos[aeid] = {"total_size": total_size, "num_active": num_active, "num_inactive": num_inactive,
                       "hit_ratio": hit_ratio}


hitcall_infos_df = pd.DataFrame(hitcall_infos, index=['total_size', 'num_active', 'num_inactive', 'hit_ratio']).T
hitcall_infos_df.reset_index(inplace=True)
hitcall_infos_df.rename(columns={'index': 'aeid'}, inplace=True)
dest_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"hitcall_infos{FILE_FORMAT}")
hitcall_infos_df.to_parquet(dest_path, compression='gzip')
