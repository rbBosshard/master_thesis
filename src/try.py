import os
import time

import pandas as pd

from helper import CSV_DIR_PATH


csv_input = os.path.join(CSV_DIR_PATH, 'input_mapping_aeid_chid.csv')
df = pd.read_csv(csv_input)
presence_matrix = pd.crosstab(df['aeid'], df['chid'])  # count presence
suffix = ".parquet.gzip"
csv_output = os.path.join(CSV_DIR_PATH, f"presence_matrix_aeid_chid{suffix}")
presence_matrix.to_parquet(csv_output, compression='gzip')