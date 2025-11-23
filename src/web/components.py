"""
Reusable components for the Streamlit app
===========================================

This module contains custom widgets and layout functions for the Munich Traffic Optimization web application.
"""

import streamlit as st

def header(title):
    """Display the main header of the app."""
    st.markdown(f'<h1 style="text-align: center;">{title}</h1>', unsafe_allow_html=True)

def subheader(text):
    """Display a subheader in the app."""
    st.markdown(f'<h2 style="text-align: center;">{text}</h2>', unsafe_allow_html=True)

def metric_card(label, value, delta=None, delta_color="normal"):
    """Display a metric card with label, value, and optional delta."""
    col = st.columns(1)[0]
    with col:
        st.metric(label=label, value=value, delta=delta, delta_color=delta_color)

def info_message(message):
    """Display an informational message."""
    st.info(message)

def success_message(message):
    """Display a success message."""
    st.success(message)

def warning_message(message):
    """Display a warning message."""
    st.warning(message)

def image_display(image, caption=None):
    """Display an image with an optional caption."""
    st.image(image, caption=caption, use_container_width=True)