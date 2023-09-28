import streamlit as st
import plotly.express as px
import pandas as pd
import os

from ml.src.pipeline.constants import METADATA_SUBSET_DIR_PATH, FILE_FORMAT

algo = "binary_classification"

path = os.path.join(METADATA_SUBSET_DIR_PATH, f"precision_vs_recall_validation_results_{algo}{FILE_FORMAT}")

precision_vs_recall_validation_results = pd.read_parquet(path).T

# Create a Streamlit app
st.title('Precision vs. Recall for Different Validation Types')

# Create a dropdown to select the validation type
selected_validation_type = st.selectbox('Select Validation Type', precision_vs_recall_validation_results.columns.tolist())

# Filter the data for the selected validation type
filtered_data = pd.DataFrame(precision_vs_recall_validation_results[selected_validation_type].tolist())

# Create a scatter plot with precision vs. recall
fig = px.scatter(filtered_data, x='recall', y='precision', labels={'recall': 'Recall', 'precision': 'Precision'})

# Set plot title and axis labels
fig.update_layout(
    title=f'Precision vs. Recall for {selected_validation_type}',
    xaxis_title='Recall',
    yaxis_title='Precision'
)

# Display the plot using Streamlit
st.plotly_chart(fig)
