import base64

from datetime import datetime


import streamlit as st


def folder_name_to_datetime(folder_name):
    return datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')


def get_verbose_name(t, default_threshold):
    threshold_mapping = {
                    'default': 'default_threshold',
                    'optimal': 'optimal_threshold',
                    'tpr': 'fixed_threshold_tpr',
                    'tnr': 'fixed_threshold_tnr'
    }
    return threshold_mapping.get(t)


def render_svg(svg):
    """
    Renders the given svg string.
    https://gist.github.com/treuille/8b9cbfec270f7cda44c5fc398361b3b1
    """
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
