import streamlit as st
import plotly.express as px
import pandas as pd
import os
import json

from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.app.helper import folder_name_to_datetime


MOST_RECENT = 0
TARGET_RUN = "2023-10-13_02-07-39"
algo = "classification"
target_folder = "2023-10-13_02-07-39"
logs_folder = os.path.join(OUTPUT_DIR_PATH, f"{algo}")
subfolders = [f for f in os.listdir(logs_folder)]
sorted_subfolders = sorted(subfolders, key=folder_name_to_datetime, reverse=True)
date = sorted_subfolders[0] if MOST_RECENT else TARGET_RUN

folder = os.path.join(logs_folder, date)
print("Target run:", date)


path = os.path.join(folder, 'aeid_paths.json')

# Load the JSON data from the output files
with open(path, 'r') as fp:
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

# Add a dropdown to the sidebar: Select single classification threshold
selected_classification_threshold = st.sidebar.selectbox('Select Classification Threshold', list(validation_results[selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type].keys()))


# Create a Streamlit app
st.title('Precision vs. Recall for Different Validation Types')

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
