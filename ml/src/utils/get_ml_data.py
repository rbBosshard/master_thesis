import os
import pandas as pd

from ml.src.utils.helper import get_subset_aeids
from ml.src.pipeline.constants import REMOTE_DATA_DIR_PATH, INPUT_DIR_PATH, FILE_FORMAT


print(f'Reading...')
src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'merged', 'output', f"0{FILE_FORMAT}")
df = pd.read_parquet(src_path)[['aeid', 'dsstox_substance_id', 'hitcall']]
dest_path = os.path.join(INPUT_DIR_PATH, f"{0}{FILE_FORMAT}")
df.to_parquet(dest_path, compression='gzip')

aeids = get_subset_aeids()['aeid']
for aeid in aeids:
    df_aeid = df[df['aeid'] == aeid]
    df_aeid = df_aeid[['dsstox_substance_id', 'hitcall']]
    dest_path = os.path.join(INPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")
    df_aeid.to_parquet(dest_path, compression='gzip')




