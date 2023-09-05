import os

import pandas as pd

from ml.src.pipeline.constants import FILE_FORMAT, FINGERPRINT_FILE, INPUT_FINGERPRINTS_DIR_PATH


def csv_to_parquet_converter():
    print("Preprocess fingerprint from structure input file")
    src_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}.csv")
    dest_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}{FILE_FORMAT}")

    df = pd.read_csv(src_path)
    # Skip the first 3 columns (relativeIndex, absoluteIndex, index) and transpose the dataframe
    df = df.iloc[:, 3:].T
    data = df.iloc[1:].values.astype(int)
    index = df.index[1:]
    columns = df.iloc[0]
    df = pd.DataFrame(data=data, index=index, columns=columns).reset_index()
    df = df.rename(columns={"index": "dsstox_substance_id"})
    df.to_parquet(dest_path, compression='gzip')

    unique_chemicals = df['dsstox_substance_id'].unique()
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}_compounds.out"), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))


