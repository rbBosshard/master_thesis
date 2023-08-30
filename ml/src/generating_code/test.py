import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Sample presence_matrix (replace this with your actual presence_matrix data)
presence_matrix = np.array([[1, 0, 2, 1, 2],
                            [2, 1, 0, 1, 0],
                            [0, 0, 2, 2, 1]])

# Map values to colors
value_colors = {0: 'red', 1: 'yellow', 2: 'green'}

# Define tick values and tick text for the color bar
tickvals = list(value_colors.keys())
ticktext = list(value_colors.values())

# Create a custom colorscale
custom_colorscale = [[0, 'red'], [0.5, 'yellow'], [1, 'green']]

# Page setup
st.set_page_config(page_title="Presence Matrix App", layout="wide")

# Create a function for the main page
def main_page():
    st.title("Presence Matrix Visualization")
    
    # Display the presence_matrix
    st.write("Presence Matrix:")
    st.write(presence_matrix)
    
    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(z=presence_matrix, colorscale=custom_colorscale,
                                     colorbar=dict(tickvals=tickvals, ticktext=ticktext),
                                     hoverinfo='text', showscale=True))
    
    # Update the layout
    fig.update_layout(
        title="Presence Matrix Heatmap",
        xaxis_title="Compound Index",
        yaxis_title="Sample Index"
    )
    
    # Display the Plotly figure using Streamlit
    st.write("Presence Matrix Heatmap:")
    st.plotly_chart(fig)

# Create a function for the about page
def about_page():
    st.title("About")
    st.write("This is a Streamlit app for visualizing presence matrices.")
    st.write("Feel free to add more content here!")

# Create navigation
nav_options = ["Main", "About"]
page = st.sidebar.radio("Navigation", nav_options)

# Display selected page
if page == "Main":
    main_page()
elif page == "About":
    about_page()
