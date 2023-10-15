import pandas as pd
import os
import json

from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.utils.helper import render_svg

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

MOST_RECENT = 0
TARGET_RUN = "2023-10-14_16-38-47"


trendline_is_enabled = st.sidebar.checkbox("Enable trendline")

if trendline_is_enabled:
    trendline_type = st.sidebar.selectbox('Select Trendline Type', [None, 'ols', 'lowess'])
else:
    trendline_type = None

histogram_is_enabled = st.sidebar.checkbox("Enable histogram")
if histogram_is_enabled:
    marginal_type = st.sidebar.selectbox('Select Marginal Type', [None, 'histogram', 'rug', 'box', 'violin'])
else:
    marginal_type = None

st.sidebar.divider()

logs_folder = os.path.join(OUTPUT_DIR_PATH)
run_folder = st.sidebar.selectbox('Select Run', [run_folder for run_folder in os.listdir(logs_folder)])

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

# Add a dropdown to the sidebar: Select single target variable
selected_target_variable = st.sidebar.selectbox('Select Target Variable', list(validation_results.keys()))

# Add a dropdown to the sidebar: Select single ml algorithm
selected_ml_algorithm = st.sidebar.selectbox('Select ML Algorithm', list(validation_results[selected_target_variable].keys()))

dummy_aeid = str(list(aeid_paths[selected_target_variable][selected_ml_algorithm].keys())[0])

# Add a dropdown to the sidebar: Select preprocessing model
selected_preprocessing_model = st.sidebar.selectbox('Select Preprocessing Model', list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid].keys()))

dummy_estimator_model = list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model].keys())[0]

# Add a dropdown to the sidebar: Select validation type
selected_validation_type = st.sidebar.selectbox('Select Validation Type', list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model][dummy_estimator_model].keys()))

def process(trendline_is_enabled, trendline_type, histogram_is_enabled, marginal_type, reports, threshold_names, i):
    threshold = threshold_names[i]
    # st.header(f"Classification Threshold: {threshold}")
    filtered_reports = {}

    for estimator_model, aeid_dict in reports.items():
        filtered_reports[estimator_model] = {}
        for aeid, report in aeid_dict.items():
            if threshold in report:
                filtered_reports[estimator_model][aeid] = report[threshold]

            # Create lists to store data for plotting
    estimators = []
    aeids = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    support = []
    support_true = []
    support_false = []

    for estimator, aeid_data in filtered_reports.items():
        for aeid, metrics in aeid_data.items():
            estimators.append(estimator)
            aeids.append(aeid)
            accuracy.append(metrics['accuracy'][0])
            precision.append(metrics['precision'][0])
            recall.append(metrics['recall'][0])
            f1.append(metrics['f1'][0])
            support.append(metrics['support'][0])
            support_true.append(metrics['support_true'][0])
            support_false.append(metrics['support_false'][0])

    df = pd.DataFrame({'Estimator': estimators, 'aeid': aeids, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Support': support, 'Support True': support_true, 'Support False': support_false})

            # Create the scatter plot
    trendline_args = {'trendline': trendline_type, 'trendline_color_override':'black'}
    hisogram_args = {'marginal_x': marginal_type, 'marginal_y': marginal_type}
    if not trendline_is_enabled:
        trendline_args = {}
    if not histogram_is_enabled:
        hisogram_args = {}
            
    fig = px.scatter(df, x='Recall', y='Precision', color='Estimator', title=f'Precision vs Recall by Estimator for {threshold} Classification Threshold', hover_data=['aeid', 'Accuracy', 'F1', 'Support', 'Support True', 'Support False' ], opacity=0.6, **trendline_args, **hisogram_args)

            # show also gridlines
    fig.update_layout(yaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'),
                            xaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'))
            # make figure square shaped
    fig.update_layout(width=550, height=550)

    st.plotly_chart(fig)

if selected_ml_algorithm == 'classification':
    reports = {}
    threshold_names = []
    for estimator_model in validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model].keys():
        reports[estimator_model] = {}
        for aeid in validation_results[selected_target_variable][selected_ml_algorithm].keys():
            reports[estimator_model][aeid] = {}
            for classification_threshold in validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][selected_validation_type].keys():
                threshold_names.append(classification_threshold)
                suffix = os.path.join(selected_preprocessing_model, estimator_model, selected_validation_type)
                report_path = os.path.join(aeid_paths[selected_target_variable][selected_ml_algorithm][aeid], suffix, f'report_{classification_threshold}.csv')
                # report = pd.read_csv(report_path).reset_index(drop=True).rename(columns={'Unnamed: 0': 'class'})
                reports[estimator_model][aeid][classification_threshold] = validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][selected_validation_type][classification_threshold]
        

    
    
    
    # Plot for each each threshold (organised in 2x2 grid) the precision vs recall for all aeids and all estimators (the estimators are color-encoded)
    i = 0
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            i = 0
            
            process(trendline_is_enabled, trendline_type, histogram_is_enabled, marginal_type, reports, threshold_names, i)
            
        with col2:
            i = 1
            process(trendline_is_enabled, trendline_type, histogram_is_enabled, marginal_type, reports, threshold_names, i)



    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            i = 2
            process(trendline_is_enabled, trendline_type, histogram_is_enabled, marginal_type, reports, threshold_names, i)

        with col2:
            i = 3
            process(trendline_is_enabled, trendline_type, histogram_is_enabled, marginal_type, reports, threshold_names, i)



            
    
    


