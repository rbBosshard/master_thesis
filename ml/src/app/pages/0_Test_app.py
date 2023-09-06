import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import streamlit as st

# Create a DataFrame with three columns
data = {
    'Group 1': np.random.randn(200) - 2,
    'Group 2': np.random.randn(200),
    'Group 3': np.random.randn(200) + 2
}
df = pd.DataFrame(data)

# Create distplot with custom bin_size
fig = ff.create_distplot([df[col] for col in df.columns], df.columns, bin_size=[0.1, 0.25, 0.5])


st.plotly_chart(fig, use_container_width=True, theme=None)
