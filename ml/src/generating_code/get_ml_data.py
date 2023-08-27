import os
import pandas as pd

from ml.src.generating_code.helper import get_subset_aeids
from ml.src.ml_helper import load_config
from ml.src.constants import REMOTE_DATA_DIR_PATH, INPUT_DIR_PATH

CONFIG = load_config()[0]

print(f'Reading...')
src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'all', 'output', f"0{CONFIG['file_format']}")
df = pd.read_parquet(src_path)[['aeid', 'dsstox_substance_id', 'hitcall']]
dest_path = os.path.join(INPUT_DIR_PATH, f"{0}{CONFIG['file_format']}")
df.to_parquet(dest_path, compression='gzip')

aeids = get_subset_aeids()['aeid']
for aeid in aeids:
    df_aeid = df[df['aeid'] == aeid]
    df_aeid = df_aeid[['dsstox_substance_id', 'hitcall']]
    dest_path = os.path.join(INPUT_DIR_PATH, f"{aeid}{CONFIG['file_format']}")
    df_aeid.to_parquet(dest_path, compression='gzip')




