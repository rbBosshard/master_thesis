import os
import pandas as pd
from datetime import datetime

from ml.src.pipeline.constants import LOG_DIR_PATH, METADATA_SUBSET_DIR_PATH, FILE_FORMAT

MOST_RECENT = 1
TARGET_RUN = "2023-09-11_11-32-31"

algo = "binary_classification"
logs_folder = os.path.join(LOG_DIR_PATH, f"runs_{algo}")
subfolders = [f for f in os.listdir(logs_folder)]


def folder_name_to_datetime(folder_name):
    return datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')


sorted_subfolders = sorted(subfolders, key=folder_name_to_datetime, reverse=True)

target_run_folder = sorted_subfolders[0] if MOST_RECENT else TARGET_RUN

print("Target run:", target_run_folder)

model_paths = {}
precision_vs_recall_validation_results = {}

target_run_folder_path = os.path.join(logs_folder, target_run_folder)
for aeid in os.listdir(target_run_folder_path):
    if aeid != ".log":
        aeid_path = os.path.join(target_run_folder_path, aeid)

        for model in os.listdir(aeid_path):

            model_path = os.path.join(aeid_path, model)
            best_estimator_path = os.path.join(model_path, 'best_estimator.joblib')
            model_paths[aeid] = best_estimator_path

            val_results = {}
            for validation_type in os.listdir(model_path):
                validation_result_path = os.path.join(model_path, validation_type)
                if os.path.isdir(validation_result_path):
                    report_path = os.path.join(validation_result_path, 'report.csv')
                    val_results[validation_type] = pd.read_csv(report_path).reset_index(drop=True)

            precision_vs_recall_validation_results[aeid] = {}
            for validation_type, val_result in val_results.items():
                macro_avg_row = val_result[val_result['Unnamed: 0'] == 'macro avg']
                precision_macro_avg = macro_avg_row['precision'].values[0]
                recall_macro_avg = macro_avg_row['recall'].values[0]
                f1_macro_avg = macro_avg_row['f1-score'].values[0]
                precision_vs_recall_validation_results[aeid][validation_type] = {
                    'precision': precision_macro_avg,
                    'recall': recall_macro_avg,
                    'f1': f1_macro_avg
                }

precision_vs_recall_validation_results = pd.DataFrame(precision_vs_recall_validation_results)
path = os.path.join(METADATA_SUBSET_DIR_PATH, f"precision_vs_recall_validation_results_{algo}{FILE_FORMAT}")
precision_vs_recall_validation_results.to_parquet(path, compression='gzip')


model_paths = pd.DataFrame(model_paths.items(), columns=['aeid', 'model_path'])
path = os.path.join(METADATA_SUBSET_DIR_PATH, f"model_paths{algo}{FILE_FORMAT}")
model_paths.to_parquet(path, compression='gzip')

print("Done")

