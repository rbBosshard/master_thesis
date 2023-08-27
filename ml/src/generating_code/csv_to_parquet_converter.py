import os
import pandas as pd
import time

from ml.src.ml_helper import load_config
from ml.src.constants import METADATA_DIR_PATH

CONFIG, CONFIG_CLASSIFIERS, START_TIME, AEID, LOGGER = load_config()

src_path = os.path.join(METADATA_DIR_PATH, 'fps', f"{CONFIG['fingerprint_file']}.csv")
dest_path = os.path.join(METADATA_DIR_PATH, 'fps', f"{CONFIG['fingerprint_file']}{CONFIG['file_format']}")

start_time = time.time()

print(f'Reading...')
df = pd.read_csv(src_path)
# Skip the first 3 columns (relativeIndex, absoluteIndex, index) and transpose the dataframe
df = df.iloc[:, 3:].T
data = df.iloc[1:].values.astype(int)
index = df.index[1:]
columns = df.iloc[0]
df = pd.DataFrame(data=data, index=index, columns=columns).reset_index()
df = df.rename(columns={"index": "dsstox_substance_id"})

print(f'Writing...')
df.to_parquet(dest_path, compression='gzip')

unique_chemicals = df['dsstox_substance_id'].unique()
with open(os.path.join(METADATA_DIR_PATH, 'fps', f"{CONFIG['fingerprint_file']}_compounds.out"), 'w') as f:
    f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))

print(f'CSV file "{src_path}" converted to parquet file "{dest_path}" with gzip compression.')



elapsed_time = time.time() - start_time

print(f'Time taken: {elapsed_time:.2f} seconds')