import streamlit as st

st.set_page_config(
    page_title="Demo App",
    page_icon="👋",
    layout="wide",
)

st.write("# Welcome to the result demo of my Master's Thesis")

st.write("by Robin Bosshard")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    **👈 Select a demo from the side bar to visualize some (intermediate) results.**
    """
)
