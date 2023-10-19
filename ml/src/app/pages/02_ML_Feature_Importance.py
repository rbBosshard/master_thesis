import pandas as pd
import numpy as np
import os
import json
from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from sklearn.metrics import jaccard_score
import plotly.express as px


MOST_RECENT = 0
TARGET_RUN = "2023-10-18_22-36-10"

# st.set_page_config(
#     layout="wide",
# )


def blank_to_underscore(x):
    return x.replace(' ', '_')


rename = {
    'hitcall': 'hitcall',
    'hitcall_c': 'hitcall (c)',
    'True': 'positive',
    'False': 'negative',
    'macro avg': 'macro avg',
    'weighted avg': 'weighted avg',
    'val': 'Internal validation',
    'mb_val_structure': 'MB validation from structure',
    'mb_val_sirius': 'MB validation SIRIUS-predicted',
    'Internal validation': 'Internal',
    'MB validation from structure': 'MB structure',
    'MB validation SIRIUS-predicted': 'MB SIRIUS',
    'default': 'default=0.5',
    'tpr': 'TPR≈0.5',
    'tnr': 'TNR≈0.5',
    'optimal': 'cost(TPR,TNR)',
    'XGBClassifier': 'XGBoost',
    'XGBoost': 'XGB',
    'RF': 'RF',
    'RBF SVM': 'SVM',
    'MLP': 'MLP',
    'LR': 'LR',
    'RandomForestClassifier': 'RF',
    'LogisticRegression': 'LR',
    'SVC': 'RBF SVM',
    'MLPClassifier': 'MLP',
    'accuracy': 'accuracy',
}

reverse_rename = {v: k for k, v in rename.items()}


logs_folder = os.path.join(OUTPUT_DIR_PATH)
folder = os.path.join(logs_folder, TARGET_RUN)

# Load the JSON data from the output files
with open(os.path.join(folder, 'aeid_paths.json'), 'r') as fp:
    aeid_paths = json.load(fp)

with open(os.path.join(folder, 'model_paths.json'), 'r') as fp:
    model_paths = json.load(fp)

with open(os.path.join(folder, 'feature_importances_paths.json'), 'r') as fp:
    feature_importances_paths = json.load(fp)

with open(os.path.join(folder, 'validation_results.json'), 'r') as fp:
    validation_results = json.load(fp)

with open(os.path.join(folder, 'validation_results_scores.json'), 'r') as fp:
    validation_results_scores = json.load(fp)


all_features = pd.DataFrame()
unique_features = set()

target_variable = 'hitcall'
ml_algorithm = 'classification'
preprocessing_model = 'Feature_Selection_XGBClassifier'
estimator_model = 'XGBClassifier'

for aeid in validation_results[target_variable][ml_algorithm].keys():
    aeid_path = aeid_paths[target_variable][ml_algorithm][aeid]
    sorted_feature_importances_path = os.path.join(aeid_path, preprocessing_model, estimator_model, 'mb_val_sirius', 'sorted_feature_importances.csv')
    sorted_feature_importances = pd.read_csv(sorted_feature_importances_path).reset_index(drop=True).head(10)
    
    # Collect unique feature indexes
    unique_features.update(sorted_feature_importances['feature'])
        

    sorted_feature_importances['aeid'] = aeid
    all_features = pd.concat([all_features, sorted_feature_importances], ignore_index=True)

# Create a mapping of old feature indexes to new linear indexes
feature_index_mapping = {feature_index: linear_index for linear_index, feature_index in enumerate(unique_features)}

# Replace the original feature indexes with the new linear indexes
all_features['linearized_feature_index'] = all_features['feature'].map(feature_index_mapping)

# log transorm importances
all_features['importances'] = all_features['importances'] ** 0.2

nbinsx = all_features['aeid'].unique().shape[0]
nbinsy = all_features['linearized_feature_index'].unique().shape[0]


# generate heatmap
fig = px.density_heatmap(all_features,
                         x='aeid',
                         y="linearized_feature_index",
                         nbinsx=nbinsx,
                         nbinsy=nbinsy,
                         z="importances",
                        #  color_continuous_scale="gray_r",
                         title="Feature importance")

fig.show()