import pandas as pd
import os
import json
import numpy as np
from ml.src.pipeline.constants import OUTPUT_DIR_PATH
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go  # Import Plotly graph objects
from ml.src.utils.helper import render_svg
from sklearn.metrics import jaccard_score



MOST_RECENT = 0
TARGET_RUN = "2023-10-14_02-01-23"

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
run_folder = st.sidebar.selectbox('Select Run', [run_folder for run_folder in os.listdir(logs_folder)][::-1])
folder = os.path.join(logs_folder, run_folder)

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



reports = {}

for target_variable in ['hitcall']:
    reports[target_variable] = {}
    for ml_algorithm in ['classification']:
        reports[target_variable][ml_algorithm] = {}
        for aeid in validation_results[target_variable][ml_algorithm].keys():
            reports[target_variable][ml_algorithm][aeid] = {}
            similarity_matrix = []
            dfs = {}
            for preprocessing_model in ['Feature_Selection_XGBClassifier', 'Feature_Selection_RandomForestClassifier']:
                reports[target_variable][ml_algorithm][aeid][preprocessing_model] = {}

                for estimator_model in ['XGBClassifier', 'RandomForestClassifier']:
                    sorted_feature_importances_path = feature_importances_paths[target_variable][ml_algorithm][aeid][preprocessing_model][estimator_model]
                    sorted_feature_importances = pd.read_csv(sorted_feature_importances_path)

                    reports[target_variable][ml_algorithm][aeid][preprocessing_model][estimator_model] = sorted_feature_importances
                    dfs[preprocessing_model + '__Estimator_' + estimator_model] = sorted_feature_importances['features'].values[:100]

                
                # Calculate Jaccard Similarity between the top 100 features of each pair of dataframes
                similarity_matrix = []

                for features1 in [top_features_df1, top_features_df2, top_features_df3, top_features_df4]:
                    similarity_row = []
                    for features2 in [top_features_df1, top_features_df2, top_features_df3, top_features_df4]:
                        jaccard_similarity = jaccard_score(features1, features2)
                        similarity_row.append(jaccard_similarity)
                    similarity_matrix.append(similarity_row)