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
validation_results = {}
feature_importance = {}
aeid_paths = {}



target_run_folder_path = os.path.join(logs_folder, target_run_folder)
for aeid in os.listdir(target_run_folder_path):
    if aeid != ".log":
        aeid_path = os.path.join(target_run_folder_path, aeid)
        
        # Exclude aeids that did not finish successfully    
        if os.path.isfile(os.path.join(aeid_path, 'success.txt')):
            validation_results[aeid] = {}
            model_paths[aeid] = {}
            feature_importance[aeid] = {}
            aeid_paths[aeid] = aeid_path
            
            # Iterate over preprocessing models (feature selection) (SelectFromModel: XGBoost & RandomForest)
            for preprocessing_model in os.listdir(aeid_path):
                preprocessing_model_path = os.path.join(aeid_path, preprocessing_model)
                validation_results[aeid][preprocessing_model] = {}
                model_paths[aeid][preprocessing_model] = {}
                feature_importance[aeid][preprocessing_model] = {}
                processing_model_path = os.path.join(aeid_path, preprocessing_model)
                
                # Iterate over estimator models (LogisticRegression, SVC, MLPClassifier, RandomForest, XGBoost) on top of a preprocessing models
                for model in os.listdir(processing_model_path):
                    model_path = os.path.join(processing_model_path, model)
                    if os.path.isdir(model_path):
                        model_paths[aeid][preprocessing_model][model] = {}
                        validation_results[aeid][preprocessing_model][model]= {}
                        best_estimator_path = os.path.join(model_path, 'best_estimator_all.joblib')
                        optimal_threshold_path = os.path.join(model_path, 'optimal_threshold_validation.joblib')
                        model_paths[aeid][preprocessing_model][model] = (best_estimator_path, optimal_threshold_path)
                        feature_importance_path = os.path.join(model_path, 'sorted_feature_importances.csv')    
                        if os.path.exists(feature_importance_path):
                            feature_importance[aeid][preprocessing_model][model] = pd.read_csv(feature_importance_path)
                        
                        # Iterate over validation results (validation, massbank_validation_from_structure, massbank_validation_from_sirius)
                        for validation_type in os.listdir(model_path):
                            validation_result_path = os.path.join(model_path, validation_type)
                            if os.path.isdir(validation_result_path):
                                validation_results[aeid][preprocessing_model][model][validation_type] = {}
                                # Iterate over the chosen classification thresholds and get respective validation report metrics
                                for threshold_report in os.listdir(validation_result_path):
                                    if 'report' in threshold_report:
                                        threshold = threshold_report.split("_")[1]
                                        report_path = os.path.join(validation_result_path, threshold_report)
                                        report = pd.read_csv(report_path).reset_index(drop=True)
                                        accuracy = report['accuracy'].values[0]
                                        macro_avg_row = report[report['Unnamed: 0'] == 'macro avg']
                                        precision_macro_avg = macro_avg_row['precision'].values[0]
                                        recall_macro_avg = macro_avg_row['recall'].values[0]
                                        f1_macro_avg = macro_avg_row['f1-score'].values[0]
                                        validation_results[aeid][preprocessing_model][model][validation_type][threshold] = {
                                            'accuracy': accuracy,
                                            'precision': precision_macro_avg,
                                            'recall': recall_macro_avg,
                                            'f1': f1_macro_avg
                                        }


validation_results = pd.DataFrame(validation_results)
path = os.path.join(METADATA_SUBSET_DIR_PATH, f"validation_results_{algo}{FILE_FORMAT}")
validation_results.to_parquet(path, compression='gzip')
path = os.path.join(METADATA_SUBSET_DIR_PATH, f"validation_results_{algo}.csv")
validation_results.to_csv(path)


model_paths = pd.DataFrame(model_paths.items(), columns=['aeid', 'model_path'])
path = os.path.join(METADATA_SUBSET_DIR_PATH, f"model_paths_{algo}{FILE_FORMAT}")
model_paths.to_parquet(path, compression='gzip')

print("Done")

