import pandas as pd
import os
import json
import numpy as np

from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.utils.helper import render_svg

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

MOST_RECENT = 0
TARGET_RUN = "2023-10-14_16-38-47_all"

st.set_page_config(
    layout="wide",
)


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

# Create a mapping from user selection to Plotly color palette
color_palette_mapping = {
    'Light24': px.colors.qualitative.Light24,
    'Plotly': px.colors.qualitative.Plotly,
    'Bold': px.colors.qualitative.Bold,
    'D3': px.colors.qualitative.D3,
    'G10': px.colors.qualitative.G10,
    'T10': px.colors.qualitative.T10,
    'Alphabet': px.colors.qualitative.Alphabet,
    'Dark24': px.colors.qualitative.Dark24,
    'Dark2': px.colors.qualitative.Dark2,
    'Set1': px.colors.qualitative.Set1,
    'Pastel1': px.colors.qualitative.Pastel1,
    'Set2': px.colors.qualitative.Set2,
    'Pastel2': px.colors.qualitative.Pastel2,
    'Set3': px.colors.qualitative.Set3,
    'Antique': px.colors.qualitative.Antique,
    'Pastel': px.colors.qualitative.Pastel,
    'Prism': px.colors.qualitative.Prism,
    'Safe': px.colors.qualitative.Safe,
    'Vivid': px.colors.qualitative.Vivid,

}

# Select color palette from the user
selected_palette = st.sidebar.selectbox('Select Color Palette', list(color_palette_mapping.keys()))

# Get the corresponding Plotly color palette
colors = color_palette_mapping.get(selected_palette)

st.sidebar.divider()

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

# Add a dropdown to the sidebar: Select single target variable
selected_target_variable = st.sidebar.selectbox('Select Target Variable', list(validation_results.keys()))

# Add a dropdown to the sidebar: Select single ml algorithm
selected_ml_algorithm = st.sidebar.selectbox('Select ML Algorithm', list(validation_results[selected_target_variable].keys()))

dummy_aeid = str(list(aeid_paths[selected_target_variable][selected_ml_algorithm].keys())[0])

# Add a dropdown to the sidebar: Select preprocessing model
selected_preprocessing_model = st.sidebar.selectbox('Select Preprocessing Model', list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid].keys())[::-1])

dummy_estimator_model = list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model].keys())[0]

# Add a dropdown to the sidebar: Select validation type
selected_validation_type = st.sidebar.selectbox('Select Validation Type', list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model][dummy_estimator_model].keys())[::-1])

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

    # use marker symbols to encode different types of estimators
    for j, (estimator, aeid_data) in enumerate(filtered_reports.items()):
        for aeid, metrics in aeid_data.items():

            estimators.append(estimator)
            aeids.append(aeid)
            metric_accuracy = next(item for item in metrics if item['class'] == 'accuracy')
            accuracy.append(list(metric_accuracy.values())[1])

            metric = next(item for item in metrics if item['class'] == slice)
            precision.append(metric['precision'])
            recall.append(metric['recall'])
            f1.append(metric['f1-score'])
            support.append(metric['support'])

    # Add one more dummy datapoint for every estimator to make the legend work (otherwise the legend shows very small markers)
    for estimator in filtered_reports.keys():
        estimators.append(estimator)
        aeids.append('-1')
        accuracy.append(-1)
        precision.append(-1)
        recall.append(-1)
        f1.append(-1)
        support.append(1000000000)
    df = pd.DataFrame({'Estimator': estimators, 'aeid': aeids, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Support': support})

    trendline_args = {'trendline': trendline_type, 'trendline_color_override':'black'}
    hisogram_args = {'marginal_x': marginal_type, 'marginal_y': marginal_type}
    if not trendline_is_enabled:
        trendline_args = {}
    if not histogram_is_enabled:
        hisogram_args = {}

    args = {}
    
    
    # Define the marker size as support-encoded, size depends on the validation type
    scale = 4 if selected_validation_type == 'val' else 2
    # marker_size = df['Support'] / scale
    if slice == 'accuracy':
        marker_size = None
        opacity = 0.3
    else:
        marker_size = df['Support'].apply(lambda x: np.sqrt(x) / scale)
        opacity = 0.8
    
    # Colors: Light24, Alphabet, Dark24, Dark2, Set1, Pastel1, Set2, Pastel2, Set3, Antique, Bold, Pastel, Prism, Safe, Vivid
    fig = px.scatter(df,
        x='Recall', y='Precision', color='Estimator',
        hover_data=['aeid', 'Accuracy', 'F1', 'Support'], opacity=opacity,
        color_discrete_sequence=colors,
        size_max=30, 
         **args, **trendline_args, **hisogram_args)

    fig.update_traces(marker=dict(size=marker_size)) 

    fig.update_layout(title={'text': f'Precision/Recall ({slice}) with {threshold.upper()} classification threshold'}, title_font=dict(size=18))

    fig.update_traces(marker=dict(symbol="circle-dot",
            line=dict(
                color='black',
                width=0.7,
            )
        ))

    fig.update_layout(yaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'),
                            xaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'))
    
    # make figure square shaped
    # make figure square shaped fixed size
    fig.update_layout(width=550, height=550)
    if clip_axis:
        fig.update_layout(xaxis=dict(range=[0.45, 1.01]))
        fig.update_layout(yaxis=dict(range=[0.45, 1.01]))
    else:
        fig.update_layout(xaxis=dict(range=[0.0, 1.01]))
        fig.update_layout(yaxis=dict(range=[0.0, 1.01]))
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1)) 
    # make grid uniform
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showspikes=True, nticks=10, tickfont=dict(size=22, color="black"))  
    fig.update_yaxes(showspikes=True, nticks=10, tickfont=dict(size=22, color="black")) 
    fig.update_layout(
        xaxis_title_font=dict(size=25, color="black"),
        yaxis_title_font=dict(size=25, color="black"),
    )
    if i == 0: 
        fig.update_layout(legend=dict(orientation='v', yanchor='top', y=0.42, xanchor='left', x=0.5, font=dict(size=19, color='black'), title_font=dict(size=18, color='black')))
            # make legend box transparent
        fig.update_layout(legend=dict(bgcolor='rgba(255, 255, 255, 0.8)'))
    if not show_legend and i != 0:
        fig.update_layout(showlegend=False)
    


    st.plotly_chart(fig)

    # Create a summary table of average metrics, accuracy, precision, recall, f1 grouped by Estimator
    if show_summary:
        st.markdown(f"##### Assay endpoint averaged metrics on: {slice}")
        
        # drop the dummy trace
        df = df[df['aeid'] != '-1']
        grouped = df[['Estimator', 'Accuracy', 'Recall', 'Precision', 'F1']].groupby(['Estimator']).mean().reset_index()
        grouped['Accuracy'] = grouped['Accuracy'].apply(lambda x: f'{x:.2f}')
        grouped['Precision'] = grouped['Precision'].apply(lambda x: f'{x:.2f}')
        grouped['Recall'] = grouped['Recall'].apply(lambda x: f'{x:.2f}')
        grouped['F1'] = grouped['F1'].apply(lambda x: f'{x:.2f}')
        
        summary = grouped[['Estimator', 'Accuracy', 'Recall', 'Precision', 'F1']]

        custom_css = f"""
        <style>
            table {{
                border-collapse: collapse;
                # width: 100%;
                margin: 0;
            }}
            th, td {{
                padding: 5px;  /* Adjust the padding as needed */
                text-align: left;
            }}
            th {{
                font-size: 18;
                font-weight: bold;
                color: #6d6d6d;
                background-color: #f7ffff;
            }}
            td {{
                font-size: 18px;
            }}
        </style>
        """

        # Display the table using st.markdown with custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown(summary.to_html(escape=False), unsafe_allow_html=True)


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
    # Create a checkbox to show/hide the legend
    slice = st.selectbox('Select Class Slice', ['macro avg', 'True', 'False', 'weighted avg', 'accuracy'])
    show_summary = st.checkbox("Show Summary", value=True)
    show_legend = st.checkbox("Show Legend", value=False)
    clip_axis = st.checkbox("Clip Axis", value=True)
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



            
    
    


