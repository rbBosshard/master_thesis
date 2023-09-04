import pandas as pd
import streamlit as st
from plotly import graph_objs as go
import os
import plotly.graph_objects as go
import io


from ml.src.pipeline.constants import MASS_BANK_DIR_PATH, FILE_FORMAT, \
    METADATA_SUBSET_DIR_PATH, METADATA_DIR_PATH

BY_ASSAY = 0
EXPORT = 0


st.header('Assay endpoint coverage by compounds')
with st.spinner(f"Loading.."):
    aeid_chid_presence_matrix_path = os.path.join(METADATA_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
    df_all = pd.read_parquet(aeid_chid_presence_matrix_path)

    aeid_chid_presence_matrix_path = os.path.join(METADATA_SUBSET_DIR_PATH,
                                                  f"aeid_compound_presence_matrix{FILE_FORMAT}")
    df_subset = pd.read_parquet(aeid_chid_presence_matrix_path)

    dfs_wrapped = [("All assay endpoints", df_all), ("Subset of all assay endpoints", df_subset)]
    tabs = st.tabs([df_wrapped[0] for df_wrapped in dfs_wrapped])

    color_bottom = "#f7f7eb"
    color_1 = "#a50aff"
    color_top = "#6254f7"

    for i, (df_name, df) in enumerate(dfs_wrapped):
        print(i, df_name)
        with tabs[i]:
            shape = df.shape
            st.write(f"Number of assay endpoints: {shape[0]}")
            st.write(f"Number of compounds: {shape[1]}")
            total_ones = df.values.sum()
            st.write(f"Total number concentration-response series (samples): {total_ones}")
            total_values = df.size
            ratio_ones_to_total = 1 - total_ones / total_values

            column_ones_counts = df.sum()
            sorted_columns = column_ones_counts.sort_values(ascending=False).index
            df = df[sorted_columns]
            df['ones_count'] = df.sum(axis=1)
            df = df.sort_values(by='ones_count', ascending=False)
            df = df.drop(columns='ones_count')

            colorscale = [[0, color_bottom], [ratio_ones_to_total, color_bottom],
                          [ratio_ones_to_total, color_top], [1, color_top]]

            showscale = not EXPORT

            fig = go.Figure(go.Heatmap(z=df.values, colorscale=colorscale, showscale=showscale,
                                       colorbar=dict(tickvals=[0, 1], ticktext=["Absent", "Present"])))

            fig.update_layout(yaxis=dict(autorange="reversed"))

            if not EXPORT:
                fig.update_layout(
                    title=f"Presence matrix: {df_name}",
                    xaxis_title="Compound Index",
                    yaxis_title="Assay Endpoint Index",

                )

            st.plotly_chart(fig, use_container_width=False, dpi=1)



            print("ok")
            # # Display the exported PDF using Streamlit
            # st.markdown(f'<embed src="test.svg" width="100%" height="600px">', unsafe_allow_html=True)

            # df = df.T
            # num_compounds = df.shape[0]
            # num_assay_endpoints = df.shape[1]
            # df = df.sum(axis=1).sort_values(ascending=False) / num_assay_endpoints
            # fig = px.scatter(df, x=df.index, y=df.values, labels={'x': 'Compounds', 'y': 'Assay endpoints coverage'})
            # fig.update_layout(title_text=f'{df_name} coverage, 100% = {num_assay_endpoints} assay endpoints')
            # fig.update_xaxes(type='category')
            # fig.update_traces(marker=dict(size=3))
            # fig.update_traces(marker_symbol='circle-open')
            # fig.update_xaxes(title_text=f'Compounds (truncated, total: {num_compounds})')
            # fig.update_yaxes(title_text=f'Assay endpoint coverage')
            # fig.update_yaxes(range=[0, 1])
            # fig.update_layout(yaxis_tickformat='.0%')
            # # fig.update_xaxes(showticklabels=False)
            # st.plotly_chart(fig)
            # selected_points = plotly_events(fig)
            # st.write(selected_points)
