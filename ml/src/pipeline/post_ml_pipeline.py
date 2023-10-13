import os
import pandas as pd
from datetime import datetime
import json

from ml.src.pipeline.constants import LOG_DIR_PATH, METADATA_SUBSET_DIR_PATH, FILE_FORMAT, OUTPUT_DIR_PATH

MOST_RECENT = 0
TARGET_RUN = "2023-10-13_02-07-39"

algo = "classification"
logs_folder = os.path.join(LOG_DIR_PATH, f"runs_{algo}")
subfolders = [f for f in os.listdir(logs_folder)]

NUM_AEIDS = 3


def folder_name_to_datetime(folder_name):
    return datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')


sorted_subfolders = sorted(subfolders, key=folder_name_to_datetime, reverse=True)

target_run_folder = sorted_subfolders[0] if MOST_RECENT else TARGET_RUN

print("Target run:", target_run_folder)

model_paths = {}
validation_results = {}
feature_importances_paths = {}
aeid_paths = {}

target_run_folder_path = os.path.join(logs_folder, target_run_folder)
for aeid in os.listdir(target_run_folder_path)[:NUM_AEIDS]:
    if aeid != ".log":
        aeid_path = os.path.join(target_run_folder_path, aeid)
        
        # Exclude aeids that did not finish successfully    
        if os.path.isfile(os.path.join(aeid_path, 'success.txt')):
            validation_results[aeid] = {}
            model_paths[aeid] = {}
            feature_importances_paths[aeid] = {}
            aeid_paths[aeid] = aeid_path
            
            # Iterate over preprocessing models (feature selection) (SelectFromModel: XGBoost & RandomForest)
            for preprocessing_model in os.listdir(aeid_path):
                preprocessing_model_path = os.path.join(aeid_path, preprocessing_model)
                if os.path.isdir(preprocessing_model_path):
                    validation_results[aeid][preprocessing_model] = {}
                    model_paths[aeid][preprocessing_model] = {}
                    feature_importances_paths[aeid][preprocessing_model] = {}

                    # Iterate over estimator models (LogisticRegression, SVC, MLPClassifier, RandomForest, XGBoost) on top of a preprocessing models
                    for model in os.listdir(preprocessing_model_path):
                        model_path = os.path.join(preprocessing_model_path, model)
                        if os.path.isdir(model_path):
                            model_paths[aeid][preprocessing_model][model] = {}
                            validation_results[aeid][preprocessing_model][model] = {}
                            best_estimator_path = os.path.join(model_path, 'best_estimator_all.joblib')
                            optimal_threshold_path = os.path.join(model_path, 'optimal_threshold_validation.joblib')
                            model_paths[aeid][preprocessing_model][model] = (best_estimator_path, optimal_threshold_path)
                            sorted_feature_importance_path = os.path.join(model_path, 'sorted_feature_importances.csv')
                            if os.path.exists(sorted_feature_importance_path):
                                feature_importances_paths[aeid][preprocessing_model][model] = sorted_feature_importance_path

                            # Iterate over validation results (validation, massbank_validation_from_structure, massbank_validation_from_sirius)
                            for validation_type in os.listdir(model_path):
                                validation_result_path = os.path.join(model_path, validation_type)
                                if os.path.isdir(validation_result_path):
                                    validation_results[aeid][preprocessing_model][model][validation_type] = {}
                                    # Iterate over the chosen classification thresholds and get respective validation report metrics
                                    for threshold_report in os.listdir(validation_result_path):
                                        if 'report' in threshold_report:
                                            threshold = threshold_report.split("_")[1].split(".")[0]
                                            report_path = os.path.join(validation_result_path, threshold_report)
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
                                            
                                            validation_results[aeid][preprocessing_model][model][validation_type][threshold] = {
                                                'accuracy': accuracy,
                                                'precision': precision_macro_avg,
                                                'recall': recall_macro_avg,
                                                'f1': f1_macro_avg,
                                                'support': int(support_macro_avg),
                                                'support_true':  int(support_true),
                                                'support_false': int(support_false)
                                            }
            print(f"aeid={aeid}, Results successfully collected")
        else:
            print(f"FAILED: aeid={aeid}, Incomplete validation results")

# Create folder for target run
folder_output_path = os.path.join(OUTPUT_DIR_PATH, f"{algo}", f"{target_run_folder}")
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


print("All results successfully saved.")
# aeid_paths = pd.DataFrame(aeid_paths.items(), columns=['aeid', 'model_path'])
# path = os.path.join(folder_output_path, f"model_paths.csv")
# aeid_paths.to_csv(path)

# model_paths = pd.DataFrame(model_paths.items(), columns=['aeid', 'model_path'])
# path = os.path.join(folder_output_path, f"model_paths.csv")
# model_paths.to_csv(path)

# validation_results = pd.DataFrame(validation_results)
# path = os.path.join(folder_output_path, f"validation_results.csv")
# validation_results.to_csv(path)

# feature_importances_paths = pd.DataFrame(feature_importances_paths)
# path = os.path.join(folder_output_path, f"feature_importances.csv")
# feature_importances_paths.to_csv(path)

# print("Post-processing feature importances...")
# # Compare feature importances and visualize them
# for aeid in feature_importances_paths.keys():
#     for preprocessing_model in feature_importances_paths[aeid].keys():
#         for model in feature_importances_paths[aeid][preprocessing_model].keys():
#             if len(feature_importances_paths[aeid][preprocessing_model][model]) > 0:
#                 feature_importances_path = feature_importances_paths[aeid][preprocessing_model][model]
#                 feature_importances = pd.read_csv(feature_importances_path)
#                 feature_importances = feature_importances.rename(columns={'Unnamed: 0': 'feature'})
#                 feature_importances = feature_importances.set_index('feature')
#                 feature_importances = feature_importances.sort_values(by=['importance'], ascending=False)
#                 feature_importances = feature_importances.reset_index()
#                 feature_importances = feature_importances.rename(columns={'feature': 'index'})
#                 feature_importances = feature_importances.set_index('index')
#                 feature_importances = feature_importances.rename(columns={'importance': f"{aeid}_{preprocessing_model}_{model}"})
#                 feature_importances = feature_importances.head(20)
#                 # feature_importances.plot.barh()
#                 # plt.show()
#                 path = os.path.join(folder_output_path, f"feature_importances.csv")
#                 feature_importances.to_csv(path, mode='a', header=False)





