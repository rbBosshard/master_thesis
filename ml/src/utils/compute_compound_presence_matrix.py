import os
import time

import pandas as pd

import sys

sys.path.append(r"C:\Users\bossh\Documents\GitHub")
from pytcpl.src.pipeline.pipeline_helper import query_db
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

from ml.src.utils.helper import get_subset_aeids
from ml.src.pipeline.constants import METADATA_DIR_PATH, METADATA_SUBSET_DIR_PATH, FILE_FORMAT

BUILD = 1
ALL = 0
SUBSET = 1


if ALL:
    print("Compute presence matrix for: all assay endpoints")
    start_time = time.time()
    print(f"Loading data..")
    aeid_compound_mapping_path = os.path.join(METADATA_DIR_PATH, f"aeid_compound_mapping{FILE_FORMAT}")

    if os.path.exists(aeid_compound_mapping_path):
        df = pd.read_parquet(aeid_compound_mapping_path)
    else:
        query_all = f"SELECT * FROM " \
                    f"(SELECT DISTINCT mc4.aeid, dsstox_substance_id " \
                    f"FROM (SELECT aeid, spid " \
                    f"FROM invitrodb_v3o5.mc4) as mc4 " \
                    f"JOIN sample ON mc4.spid = sample.spid " \
                    f"JOIN chemical ON sample.chid = chemical.chid) AS aeid_compound_map"
        df = query_db(query_all)
        df.to_parquet(aeid_compound_mapping_path, compression='gzip')

    # Get presence matrix
    print(f"Computing..")
    presence_matrix = pd.crosstab(df['aeid'], df['dsstox_substance_id'])  # count presence
    print(f"Writing output..")
    aeid_compound_presence_matrix_path = os.path.join(METADATA_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
    presence_matrix.to_parquet(aeid_compound_presence_matrix_path, compression='gzip')

    # Get compounds
    compounds = presence_matrix.columns.tolist()
    compounds_path = os.path.join(METADATA_DIR_PATH, f"compounds_tested{FILE_FORMAT}")
    pd.DataFrame(compounds, columns=['dsstox_substance_id']).to_parquet(compounds_path, compression='gzip')

    with open(os.path.join(METADATA_DIR_PATH, 'compounds_tested.out'), 'w') as f:
        for compound in compounds:
            f.write(str(compound) + '\n')

    print(f"Total time taken: {(time.time() - start_time):.2f} seconds")

if SUBSET:
    print("Compute presence matrix for: subset of assay endpoints")
    start_time = time.time()
    print(f"Loading data..")
    aeid_compound_presence_matrix_path = os.path.join(METADATA_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
    presence_matrix = pd.read_parquet(aeid_compound_presence_matrix_path)
    aeids = get_subset_aeids()['aeid'].tolist()
    presence_matrix = presence_matrix[presence_matrix.index.isin(aeids)].reset_index(drop=True)

    cols_with_zero_count = presence_matrix.columns[presence_matrix.sum(axis=0) == 0]
    presence_matrix = presence_matrix.drop(columns=cols_with_zero_count)

    print(f"Writing output..")
    aeid_compound_presence_matrix_path = os.path.join(METADATA_SUBSET_DIR_PATH,
                                                  f"aeid_compound_presence_matrix{FILE_FORMAT}")
    presence_matrix.to_parquet(aeid_compound_presence_matrix_path, compression='gzip')

    # Get compounds
    compounds = presence_matrix.columns.tolist()
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested{FILE_FORMAT}")
    pd.DataFrame(compounds, columns=['dsstox_substance_id']).to_parquet(compounds_path, compression='gzip')

    with open(os.path.join(METADATA_SUBSET_DIR_PATH, 'compounds_tested.out'), 'w') as f:
        for compound in compounds:
            f.write(str(compound) + '\n')

    compounds_with_zero_count = cols_with_zero_count.tolist()
    with open(os.path.join(METADATA_SUBSET_DIR_PATH, 'compounds_absent.out'), 'w') as f:
        for compound in compounds_with_zero_count:
            f.write(str(compound) + '\n')

    print(f"Total time taken: {(time.time() - start_time):.2f} seconds")