# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk

import ollama

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -2x;
        margin-bottom: -2px;
        margin-left: -2px;
        margin-right: -2px;
    }
    img[data-testid="stLogo"] {
                height: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("How to Create Your Own Chatbot w/ Ollama")
st.divider()

with st.sidebar:
    st.logo(path_cda + '\\logo.png', size='large')
    st.empty()

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.subheader(':one: Import libraries')
st.code('''
import streamlit as st
import ollama
''')

st.subheader(':two: Create chat history')
st.code('''
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
''')

st.subheader(':three: Create chat input')
st.code('''
if prompt := st.chat_input("What can I help with?"):

    # User
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Model
    with st.chat_message("assistant"):
        response = ollama.chat(model='gemma2',
                               messages=[
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                ],
                                stream=True)
        
        response_content = ''
        def catch_response(response):
            global response_content
            for chunk in response:
                response_content += chunk['message']['content']
                yield chunk['message']['content']

        stream = catch_response(response)
        st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response_content})
''')

st.subheader(':four: Chatbot')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What can I help with?"):

    # User
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Model
    with st.chat_message("assistant"):
        response = ollama.chat(model='gemma2',
                               messages=[
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                ],
                                stream=True)
        
        stream_content = ''
        def catch_stream(response):
            global stream_content
            for chunk in response:
                stream_content += chunk['message']['content']
                yield chunk['message']['content']

        stream = catch_stream(response)
        st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": stream_content})