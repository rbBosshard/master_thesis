import os
import time

import pandas as pd

from helper import CSV_DIR_PATH

start_time = time.time()
print(f"Start computing aeid chid matrix..")

print(f"Loading input..")
csv_input = os.path.join(CSV_DIR_PATH, 'input_mapping_aeid_chid.csv')
df = pd.read_csv(csv_input)

print(f"Computing..")
df['aeid']
presence_matrix = pd.crosstab(df['aeid'], df['chid'])  # count presence
# binary_presence_matrix = presence_matrix.applymap(lambda x: 1 if x > 0 else 0)  # binary presence

print(f"Writing output..")
suffix = ".parquet.gzip"

csv_output = os.path.join(CSV_DIR_PATH, f"presence_matrix_aeid_chid{suffix}")
presence_matrix.to_parquet(csv_output, compression='gzip')

num = 100
small_subset_sample_matrix = presence_matrix.iloc[:num, :num]
csv_output = os.path.join(CSV_DIR_PATH, f"small_subset_sample_matrix{suffix}")
small_subset_sample_matrix.to_parquet(csv_output, compression='gzip')

print(f"Total time taken: {(time.time() - start_time):.2f} seconds")
