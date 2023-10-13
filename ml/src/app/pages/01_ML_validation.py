import pandas as pd
import os
import json

from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.app.helper import render_svg

import streamlit as st

MOST_RECENT = 0
TARGET_RUN = "2023-10-13_02-07-39"

target_values = [target_value for target_value in os.listdir(OUTPUT_DIR_PATH) if os.path.isdir(os.path.join(OUTPUT_DIR_PATH, target_value))]
target_hitcall = st.sidebar.selectbox('Select Target Value', target_values)

algos = [algo for algo in os.listdir(os.path.join(OUTPUT_DIR_PATH, target_hitcall)) if os.path.isdir(os.path.join(OUTPUT_DIR_PATH, target_hitcall, algo))]
algo = st.sidebar.selectbox('Select ML supervised algorithm ', algos)

logs_folder = os.path.join(OUTPUT_DIR_PATH, f"{target_hitcall}", f"{algo}")
run_folder = st.sidebar.selectbox('Select Run', [run_folder for run_folder in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, run_folder))])

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


# Add a dropdown to the sidebar: Select single aeid
selected_aeid = st.sidebar.selectbox('Select AEID', list(aeid_paths.keys()))

# Add a dropdown to the sidebar: Select single preprocessing model
selected_preprocessing_model = st.sidebar.selectbox('Select Preprocessing Model', list(model_paths[selected_aeid].keys()))

# Add a dropdown to the sidebar: Select single estimator model
selected_estimator_model = st.sidebar.selectbox('Select Estimator Model', list(model_paths[selected_aeid][selected_preprocessing_model].keys()))

# Add a dropdown to the sidebar: Select single validation type
selected_validation_type = st.sidebar.selectbox('Select Validation Type', list(validation_results[selected_aeid][selected_preprocessing_model][selected_estimator_model].keys()))

# # Add a dropdown to the sidebar: Select single classification threshold
# selected_classification_threshold = st.sidebar.selectbox('Select Classification Threshold', list(validation_results[selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type].keys()))
# Load threshold_value from estimator model path    
# threshold_value_path = os.path.join(model_paths[selected_aeid][selected_preprocessing_model][selected_estimator_model], f'threshold_value.json')

st.title(f'Validation Results')
info_data = {
    "AEID": [selected_aeid],
    "Feature Selection": [selected_preprocessing_model.split('_')[2]],
    "Estimator": [selected_estimator_model],
    "Validation Set": [selected_validation_type],
    # "Classification Threshold": [selected_classification_threshold]
}
info_df = pd.DataFrame(info_data)
st.dataframe(info_df, hide_index=True, use_container_width=True)

# Plot the validation results
validation_result = validation_results[selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type]
validation_results_df = pd.DataFrame(validation_result)


# Show report as dataframe
# st.subheader('Classification Report')
# st.dataframe(report_df)

# Plot confusion matrices for all 4 classification thresholds
st.subheader('Confusion Matrices for 4 classification thresholds')

confusion_matrices_paths = {}
reports = {}
threshold_names = []
threshold_values = []
for classification_threshold in validation_results[selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type].keys():
    cm_path = os.path.join(aeid_paths[selected_aeid], selected_preprocessing_model, selected_estimator_model, selected_validation_type, f'confusion_matrix_{classification_threshold}.svg')
    confusion_matrices_paths[classification_threshold] = cm_path
    report_path = os.path.join(aeid_paths[selected_aeid], selected_preprocessing_model, selected_estimator_model, selected_validation_type, f'report_{classification_threshold}.csv')   
    reports[classification_threshold] = pd.read_csv(report_path).reset_index(drop=True).rename(columns={'Unnamed: 0': 'class'})
    threshold_names.append(classification_threshold)

# Create a 2x2 grid using columns
i = 0
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        i = 0
        cm_path = confusion_matrices_paths[threshold_names[i]]
        render_svg(open(cm_path).read())
        with st.expander("Show Report"):
            st.dataframe(reports[i])
    with col2:
        i = 1
        cm_path = confusion_matrices_paths[threshold_names[i]]
        render_svg(open(cm_path).read())
        with st.expander("Show Report"):
            st.dataframe(reports[i])
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        i = 2
        cm_path = confusion_matrices_paths[threshold_names[i]]
        render_svg(open(cm_path).read())
        with st.expander("Show Report"):
            st.dataframe(reports[i])
    with col2:
        i = 3
        cm_path = confusion_matrices_paths[threshold_names[i]]
        render_svg(open(cm_path).read())
        with st.expander("Show Report"):
            st.dataframe(reports[i])


# # Create a dropdown to select the validation type
# validation_types = validation_results[list(validation_results.keys())[0]][list(validation_results[list(validation_results.keys())[0]].keys())[0]].keys()
# print(validation_types)
# selected_validation_type = st.selectbox('Select Validation Type', precision_vs_recall_validation_results.columns.tolist())

# Filter the data for the selected validation type
# filtered_data = pd.DataFrame(precision_vs_recall_validation_results[selected_validation_type].tolist())

# Create a scatter plot with precision vs. recall
# fig = px.scatter(filtered_data, x='recall', y='precision', labels={'recall': 'Recall', 'precision': 'Precision'})

# Set plot title and axis labels
# fig.update_layout(
#     title=f'Precision vs. Recall for {selected_validation_type}',
#     xaxis_title='Recall',
#     yaxis_title='Precision'
# )

# Display the plot using Streamlit
# st.plotly_chart(fig)
