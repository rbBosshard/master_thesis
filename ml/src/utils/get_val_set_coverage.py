import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ml.src.utils.helper import get_subset_aeids, get_validation_compounds
from ml.src.pipeline.constants import INPUT_DIR_PATH, MASS_BANK_DIR_PATH, FILE_FORMAT, \
    METADATA_SUBSET_DIR_PATH

COLLECT_STATS = 1
COMPUTE_PRESENCE_MATRIX = 1

VALIDATION_COVERAGE_DIR_PATH = os.path.join(MASS_BANK_DIR_PATH, 'validation_coverage')
VALIDATION_COVERAGE_PLOTS_DIR_PATH = os.path.join(MASS_BANK_DIR_PATH, 'validation_coverage_plots')

os.makedirs(VALIDATION_COVERAGE_DIR_PATH, exist_ok=True)

print(f'Reading...')

# Get assay endpoint ML data for each aeid
assay_dfs = []
aeids = get_subset_aeids()['aeid']
for aeid in aeids:
    src_path = os.path.join(INPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")
    assay_dfs.append((aeid, pd.read_parquet(src_path)))

# Get assay endpoint ML data merged
src_path = os.path.join(INPUT_DIR_PATH, f"{0}{FILE_FORMAT}")
df_all = pd.read_parquet(src_path).drop(columns=['hitcall']) # dropped hitcall column

# Get subset compounds
compounds_safe_and_unsafe_for_validation, compounds_safe_for_validation, compounds_unsafe_for_validation = get_validation_compounds()
subset_ids_list = [compounds_safe_and_unsafe_for_validation, compounds_safe_for_validation, compounds_unsafe_for_validation]
subset_ids_list_names = ["compounds_safe_and_unsafe_for_validation", "compounds_safe_for_validation", "compounds_unsafe_for_validation"]
for subset_ids_list_name in subset_ids_list_names:
    os.makedirs(os.path.join(VALIDATION_COVERAGE_DIR_PATH, subset_ids_list_name), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, subset_ids_list_name), exist_ok=True)

# Get set of all compounds tested
compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested{FILE_FORMAT}")
all_compounds = pd.read_parquet(compounds_path)['dsstox_substance_id']


if COLLECT_STATS:
    for j, subset_ids in enumerate(subset_ids_list):
        coverage_info = {}
        compound_list_name = subset_ids_list_names[j]
        for i, (aeid, df) in enumerate(assay_dfs):
            intersection = df[df['dsstox_substance_id'].isin(subset_ids)]
            num_compounds = len(subset_ids)
            overlap = len(intersection)
            relative_coverage = overlap / num_compounds

            coverage_info[aeid] = {'compounds': num_compounds,
                                            'overlap': overlap,
                                            'relative_coverage': relative_coverage}

            filename = os.path.join(VALIDATION_COVERAGE_DIR_PATH, compound_list_name, f"{aeid}{FILE_FORMAT}")
            intersection.to_parquet(filename, compression='gzip')

            filename = os.path.join(VALIDATION_COVERAGE_DIR_PATH, compound_list_name, f"{aeid}.out")
            with open(filename, 'w') as file:
                file.write(f"Compounds: {num_compounds}\n")
                file.write(f"Overlap: {overlap}\n")
                file.write(f"Relative_coverage: {relative_coverage}\n")
    
        coverage_info = pd.DataFrame(coverage_info).T
        coverage_info.to_parquet(os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"coverage_info{FILE_FORMAT}"), compression='gzip')


if COMPUTE_PRESENCE_MATRIX:
    print("Compute presence_matrix")
    for j, subset_ids in enumerate(subset_ids_list):
        compound_list_name = subset_ids_list_names[j]

        presence_matrix = np.zeros((len(assay_dfs), len(all_compounds)), dtype=int)

        results = []
        def process(aeid, df):
            mask_a = np.isin(all_compounds, df['dsstox_substance_id'])
            mask_b = np.isin(all_compounds, subset_ids)
            final_mask = np.logical_and(mask_a, mask_b)
            presence_row = np.where(final_mask, 2, mask_a).astype(int)
            return aeid, presence_row

        num_cores = -1
        with Parallel(n_jobs=num_cores) as parallel:
            results = parallel(
                delayed(process)(aeid, df) for aeid, df in assay_dfs
            )

        for i, (aeid, presence_row) in enumerate(results):
            presence_matrix[i, :] = presence_row
            print(aeid)

        save_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"presence_matrix{FILE_FORMAT}")
        presence_matrix = pd.DataFrame(presence_matrix, index=[aeid for aeid, df in assay_dfs], columns=all_compounds)
        presence_matrix.to_parquet(save_path, compression='gzip')








