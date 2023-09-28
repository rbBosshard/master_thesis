import streamlit as st
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px

from ml.src.pipeline.constants import METADATA_SUBSET_DIR_PATH

# Create a larger sample DataFrame
path = os.path.join(METADATA_SUBSET_DIR_PATH, 'assay_info.parquet.gzip')

df = pd.read_parquet(path)

subset_assay_info_columns = ["aeid",
                             "assay_component_endpoint_name",
                             "assay_function_type",
                             "signal_direction",
                             "MechanisticTarget",
                             "ToxicityEndpoint",
                             "biological_process_target",
                             "intended_target_family",
                             "intended_target_family_sub",
                             "intended_target_type",
                             "intended_target_type_sub",
                             "burst_assay",
                             ]

df = df[subset_assay_info_columns]
df.insert(0, 'Select', True)


st.title("Interactive Predictive Model Grouping")

group_column = st.sidebar.selectbox("Select Column to Group On", df.columns[3:])

grouped_df = df.groupby(group_column)

selected_group_names = []

grouped_new = {}

# Display each group as a separate DataFrame
for group_name, group_data in grouped_df:
    # Display the filtered group DataFrame with checkbox column
    st.write(f"#### {group_name} Group:")

    # Display the filtered group DataFrame with checkbox column
    checkbox_config = st.data_editor(
        group_data,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select?",
                help="Select assay endpoint for inclusion in predictive model",
                default=True,
            )
        },
        disabled=subset_assay_info_columns,
        hide_index=True,
        use_container_width=True,

    )
    df_group = pd.DataFrame(checkbox_config)
    grouped_new[group_name] = df_group

grouped_selected = {}
for group_name, group_data in grouped_new.items():
    grouped_selected[group_name] = group_data[group_data['Select'] == True]

# Count the number of rows in the groups
counts = [(group_name, len(group_data)) for group_name, group_data in grouped_selected.items()]

avg_age = [(group_name, np.mean(group_data['aeid'])) for group_name, group_data in grouped_selected.items()]

# Create a DataFrame for the average age data
avg_age_df = pd.DataFrame(avg_age, columns=['Group', 'Average Age'])

# Create a polar plot using Plotly Express
fig = px.line_polar(avg_age_df, r='Average Age', theta='Group', line_close=True)

# Set the layout for the polar plot
fig.update_traces(fill='toself')
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        ),
    ),
)

# Streamlit app
st.title("Average Age Polar Plot")

# Display the polar plot in the Streamlit app
st.plotly_chart(fig)

total_row_count = sum(count for _, count in counts)

st.sidebar.write(f"Assay endpoint counts per groups: {total_row_count}")
st.sidebar.dataframe(pd.DataFrame(counts, columns=['Group', 'Count']), hide_index=True)
st.sidebar.dataframe(avg_age)

st.sidebar.write(f"Total Number of Rows in Groups: {total_row_count}")