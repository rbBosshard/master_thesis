import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ml.src.utils.helper import get_subset_aeids, get_validation_compounds
from ml.src.pipeline.constants import INPUT_DIR_PATH, FILE_FORMAT, \
    METADATA_SUBSET_DIR_PATH, VALIDATION_COVERAGE_DIR_PATH, VALIDATION_COVERAGE_PLOTS_DIR_PATH

COLLECT_STATS = 0
COMPUTE_PRESENCE_MATRIX = 0

COMPUTE_HIT_CALL_MATRIX = 1

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
df_all = pd.read_parquet(src_path)
# df_all = df_all.drop(columns=['hitcall']) # dropped hitcall column

# Get subset compounds
compounds_safe_and_unsafe_for_validation, compounds_safe_for_validation, compounds_unsafe_for_validation = get_validation_compounds()
subset_ids_list = [compounds_safe_and_unsafe_for_validation, compounds_safe_for_validation,
                   compounds_unsafe_for_validation]
subset_ids_list_names = ["compounds_safe_and_unsafe_for_validation", "compounds_safe_for_validation",
                         "compounds_unsafe_for_validation"]
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
        coverage_info.to_parquet(
            os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"coverage_info{FILE_FORMAT}"),
            compression='gzip')

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

        save_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name,
                                 f"presence_matrix{FILE_FORMAT}")
        presence_matrix = pd.DataFrame(presence_matrix, index=[aeid for aeid, df in assay_dfs], columns=all_compounds)
        presence_matrix.to_parquet(save_path, compression='gzip')

if COMPUTE_HIT_CALL_MATRIX:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    data_dict = {}

    assay_dfs = []
    aeids = get_subset_aeids()['aeid']
    for aeid in aeids:
        src_path = os.path.join(INPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")
        data_dict[aeid] = pd.read_parquet(src_path)

    for j, subset_ids in enumerate(subset_ids_list):
        compound_list_name = subset_ids_list_names[j]
        print(compound_list_name)
        # Create a matrix filled with NaN values
        presence_matrix = pd.DataFrame(np.nan, index=data_dict.keys(), columns=subset_ids)

        results = []
        # Create a dictionary to store values for the heatmap
        heatmap_data = {}
        hitcall_infos = {}

        special_value = -1  # You can change this to any value you prefer
        for aeid, df in data_dict.items():
            hitcall_values = df[df['dsstox_substance_id'].isin(subset_ids)].set_index('dsstox_substance_id')['hitcall']
            # Binarize the 'hitcall' values
            # hitcall_values = (hitcall_values >= 0.5).astype(int)

            total_size = len(hitcall_values)
            num_active = hitcall_values.sum()
            num_inactive = total_size - num_active
            hit_ratio = num_active / total_size
            hitcall_infos[aeid] = {"total_size": total_size, "num_active": num_active, "num_inactive": num_inactive,
                                   "hit_ratio": hit_ratio}

            missing_compounds = set(subset_ids) - set(hitcall_values.index)
            missing_data = pd.Series([special_value] * len(missing_compounds), index=missing_compounds)
            heatmap_data[aeid] = pd.concat([hitcall_values, missing_data])

        hitcall_infos_df = pd.DataFrame(hitcall_infos, index=['total_size', 'num_active', 'num_inactive', 'hit_ratio']).T
        hitcall_infos_df.reset_index(inplace=True)
        hitcall_infos_df.rename(columns={'index': 'aeid'}, inplace=True)
        dest_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"hitcall_infos{FILE_FORMAT}")
        hitcall_infos_df.to_parquet(dest_path, compression='gzip')

        hitcall_infos_df_all_compounds = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"hitcall_infos{FILE_FORMAT}"))

        merged_df = hitcall_infos_df_all_compounds.merge(hitcall_infos_df, on='aeid', suffixes=('_df1', '_df2'))

        plt.figure(figsize=(8, 6))
        plt.scatter(merged_df['hit_ratio_df1'], merged_df['hit_ratio_df2'], marker='o', alpha=0.5)
        plt.xlabel('hit_ratio_df1')
        plt.ylabel('hit_ratio_df2')
        plt.title('Scatter Plot of hit_ratio from df1 vs. df2')

        # Add the equation line y = x
        max_value = max(merged_df['hit_ratio_df1'].max(), merged_df['hit_ratio_df2'].max())
        plt.plot([0, max_value], [0, max_value], linestyle='--', color='gray', label='y=x')
        plt.plot([0, max_value/2], [0, max_value], linestyle='--', color='blue', label='y=2x')

        plt.grid(True)
        plt.legend()
        plt.show()


        def custom_sum(row):
            return row.apply(lambda x: 0 if x < 0 else 1).sum()


        heatmap_df = pd.DataFrame(heatmap_data).T
        column_ones_counts = heatmap_df.apply(custom_sum, axis=0)
        sorted_columns = column_ones_counts.sort_values(ascending=False).index
        heatmap_df = heatmap_df[sorted_columns]

        heatmap_df['ones_count'] = heatmap_df.apply(custom_sum, axis=1)
        heatmap_df = heatmap_df.sort_values(by='ones_count', ascending=False)
        heatmap_df = heatmap_df.drop(columns='ones_count')

        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_df.values, cmap='viridis', interpolation='nearest', aspect='auto', vmin=-1, vmax=1)

        # Set x and y-axis labels
        plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90)
        plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

        # Show colorbar
        plt.colorbar()

        # Show the heatmap
        plt.show()

        # for i, (aeid, presence_row) in enumerate(results):
        #     presence_matrix[i, :] = presence_row

        # save_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"presence_matrix{FILE_FORMAT}")
        # presence_matrix = pd.DataFrame(presence_matrix, index=[aeid for aeid, df in data_dict.items()], columns=all_compounds)
        # presence_matrix.to_parquet(save_path, compression='gzip')
