import os
import pandas as pd
from datetime import datetime

from ml.src.pipeline.constants import LOG_DIR_PATH

MOST_RECENT = 0
TARGET_RUN = "2023-09-11_11-32-31"

algo = "binary_classification"
logs_folder = os.path.join(LOG_DIR_PATH, f"runs_{algo}")
subfolders = [f for f in os.listdir(logs_folder)]


def folder_name_to_datetime(folder_name):
    return datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')


sorted_subfolders = sorted(subfolders, key=folder_name_to_datetime, reverse=True)

target_run_folder = sorted_subfolders[0] if MOST_RECENT else TARGET_RUN

print("Target run:", target_run_folder)

results_classifier = {}

target_run_folder_path = os.path.join(logs_folder, target_run_folder)
for aeid in os.listdir(target_run_folder_path):
    if aeid != ".log":
        aeid_path = os.path.join(target_run_folder_path, aeid)

        for classifier in os.listdir(aeid_path):

            classifier_path = os.path.join(aeid_path, classifier)

            val_res = {}
            for validation_type in os.listdir(classifier_path):
                validation_result_path = os.path.join(classifier_path, validation_type)
                # Read confusion matrix and classification report data into dataframes
                confusion_matrix_file = os.path.join(validation_result_path, 'confusion_matrix.png')
                classification_report_file = os.path.join(validation_result_path, 'classification_report.png')
                val_res[validation_type] = classification_report_file

            results_classifier[aeid] = {val_res}




