import os
import pandas as pd
import json

from ml.src.pipeline.constants import LOG_DIR_PATH, OUTPUT_DIR_PATH
from ml.src.utils.helper import folder_name_to_datetime

MOST_RECENT = 0
TARGET_RUN = "2023-10-14_16-38-47"
NUM_AEIDS = 1000

logs_folder = os.path.join(LOG_DIR_PATH)
subfolders = [f for f in os.listdir(logs_folder)]
sorted_subfolders = sorted(subfolders, key=folder_name_to_datetime, reverse=True)
target_run_folder = sorted_subfolders[0] if MOST_RECENT else TARGET_RUN
print("Target run:", target_run_folder)
print("#" * 50)

model_paths = {}
validation_results = {}
feature_importances_paths = {}
aeid_paths = {}
success_counter = 0
failed_counter = 0

target_run_folder_path = os.path.join(logs_folder, target_run_folder)
# Iterate over target variables (hitcall, hitcall_c)
for target_variable in os.listdir(target_run_folder_path):
    if ".log" not in target_variable:
        target_variable_path = os.path.join(target_run_folder_path, target_variable)
        validation_results[target_variable] = {}
        model_paths[target_variable] = {}
        feature_importances_paths[target_variable] = {}
        aeid_paths[target_variable] = {}
        
        # Iterate over ML algorithms (binary classification, regression)
        for ml_algorithm in os.listdir(target_variable_path):
            ml_algorithm_path = os.path.join(target_variable_path, ml_algorithm)
            validation_results[target_variable][ml_algorithm] = {}
            model_paths[target_variable][ml_algorithm] = {}
            feature_importances_paths[target_variable][ml_algorithm] = {}
            aeid_paths[target_variable][ml_algorithm] = {}

            # Iterate over aeids
            print(f"target_variable = {target_variable}, ml_algorithm = {ml_algorithm}")
            for aeid in os.listdir(ml_algorithm_path)[:NUM_AEIDS]:
                aeid_path = os.path.join(ml_algorithm_path, aeid)
                
                # Exclude aeids that did not finish successfully    
                if not os.path.isfile(os.path.join(aeid_path, 'failed.txt')):
                    validation_results[target_variable][ml_algorithm][aeid] = {}
                    model_paths[target_variable][ml_algorithm][aeid] = {}
                    feature_importances_paths[target_variable][ml_algorithm][aeid] = {}
                    aeid_paths[target_variable][ml_algorithm][aeid] = aeid_path
                    
                    # Iterate over preprocessing models (feature selection) (SelectFromModel: XGBoost & RandomForest)
                    for preprocessing_model in os.listdir(aeid_path):
                        preprocessing_model_path = os.path.join(aeid_path, preprocessing_model)
                        if os.path.isdir(preprocessing_model_path):
                            validation_results[target_variable][ml_algorithm][aeid][preprocessing_model] = {}
                            model_paths[target_variable][ml_algorithm][aeid][preprocessing_model] = {}
                            feature_importances_paths[target_variable][ml_algorithm][aeid][preprocessing_model] = {}

                            # Iterate over estimator models (LogisticRegression, SVC, MLPClassifier, RandomForest, XGBoost) on top of a preprocessing models
                            for model in os.listdir(preprocessing_model_path):
                                model_path = os.path.join(preprocessing_model_path, model)
                                if os.path.isdir(model_path):
                                    model_paths[target_variable][ml_algorithm][aeid][preprocessing_model][model] = os.path.join(model_path, 'best_estimator_all.joblib')
                                    validation_results[target_variable][ml_algorithm][aeid][preprocessing_model][model] = {}

                                    sorted_feature_importance_path = os.path.join(model_path, 'sorted_feature_importances.csv')
                                    if os.path.exists(sorted_feature_importance_path):
                                        feature_importances_paths[target_variable][ml_algorithm][aeid][preprocessing_model][model] = sorted_feature_importance_path

                                    # Iterate over validation results (validation, massbank_validation_from_structure, massbank_validation_from_sirius)
                                    for validation_type in os.listdir(model_path):
                                        validation_type_path = os.path.join(model_path, validation_type)
                                        if os.path.isdir(validation_type_path):
                                            validation_results[target_variable][ml_algorithm][aeid][preprocessing_model][model][validation_type] = {}
                                            
                                            # Iterate over the chosen classification thresholds and get respective validation report metrics
                                            for threshold_report in os.listdir(validation_type_path):
                                                if 'report' in threshold_report:
                                                    threshold = threshold_report.split("_")[1].split(".")[0]
                                                    report_path = os.path.join(validation_type_path, threshold_report)
                                                    report = pd.read_csv(report_path).reset_index(drop=True).rename(columns={'Unnamed: 0': 'class'})

                                                    accuracy_row = report[report['class'] == 'accuracy']
                                                    accuracy = accuracy_row.iloc[0, 1]

                                                    macro_avg_row = report[report['class'] == 'macro avg']
                                                    precision_macro_avg = macro_avg_row['precision'].values[0]
                                                    recall_macro_avg = macro_avg_row['recall'].values[0]
                                                    f1_macro_avg = macro_avg_row['f1-score'].values[0]
                                                    support_macro_avg = macro_avg_row['support'].values[0]

                                                    true_row = report[report['class'] == 'True']
                                                    support_true = true_row['support'].values[0]

                                                    false_row = report[report['class'] == 'False']
                                                    support_false = false_row['support'].values[0]

                                                    validation_results[target_variable][ml_algorithm][aeid][preprocessing_model][model][validation_type][threshold] = {
                                                        'accuracy': [accuracy],
                                                        'precision': [precision_macro_avg],
                                                        'recall': [recall_macro_avg],
                                                        'f1': [f1_macro_avg],
                                                        'support': [int(support_macro_avg)],
                                                        'support_true':  [int(support_true)],
                                                        'support_false': [int(support_false)]
                                                    }

                    print(f"SUCCESS: aeid={aeid}, Results successfully collected")
                    success_counter += 1
                else:
                    print(f"FAILED: aeid={aeid}, Incomplete validation results")
                    failed_counter += 1

            print("-" * 100)
            print(f"# Success: {success_counter} ")
            print(f"# Failed: {failed_counter}")

            success_counter = 0
            failed_counter = 0

# Create folder for target run
folder_output_path = os.path.join(OUTPUT_DIR_PATH, f"{target_run_folder}")
os.makedirs(folder_output_path, exist_ok=True)

# Save dictionaries as json files
path = os.path.join(folder_output_path, f"aeid_paths.json")
with open(path, 'w') as fp:
    json.dump(aeid_paths, fp)

path = os.path.join(folder_output_path, f"model_paths.json")
with open(path, 'w') as fp:
    json.dump(model_paths, fp)

path = os.path.join(folder_output_path, f"validation_results.json")
with open(path, 'w') as fp:
    json.dump(validation_results, fp)

path = os.path.join(folder_output_path, f"feature_importances_paths.json")
with open(path, 'w') as fp:
    json.dump(feature_importances_paths, fp)

print("=" * 50)

print("Done.")





