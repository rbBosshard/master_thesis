import os
import pandas as pd
import time

from ml.src.ml_helper import load_config
from ml.src.constants import INPUT_DIR_PATH

CONFIG, CONFIG_CLASSIFIERS, START_TIME, AEID, LOGGER = load_config()

csv_file_path = os.path.join(INPUT_DIR_PATH, f"{CONFIG['fingerprint_file']}.csv")
parquet_file_path = os.path.join(INPUT_DIR_PATH, f"{CONFIG['fingerprint_file']}{CONFIG['file_format']}")

start_time = time.time()

print(f'Reading...')
df = pd.read_csv(csv_file_path)
print(f'Writing...')
df.to_parquet(parquet_file_path, compression='gzip')
print(f'CSV file "{csv_file_path}" converted to Parquet file "{parquet_file_path}" with GZIP compression.')
elapsed_time = time.time() - start_time

print(f'Time taken: {elapsed_time:.2f} seconds')