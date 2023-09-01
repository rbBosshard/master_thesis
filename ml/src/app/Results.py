import streamlit as st

st.set_page_config(
    page_title="Results",
    page_icon="👋",
)

st.write("# Welcome!")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    **👈 Select a demo from the side bar to visualize some (intermediate) results.**
    """
)
