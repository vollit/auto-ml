import streamlit as st
from operator import index
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup, compare_models, pull, save_model, load_model

st.set_page_config(
    page_title="Explainable Auto ML",
    page_icon="✅",
    layout="wide",
)


with st.sidebar: 
    st.image("./assets/info-support-logo.png")
    st.title("Auto-XAI")
    choice = st.radio("Navigation", ["Upload dataset","Setup Data","Data Analyse","Modelling", "Visualise"])

 
if choice == "Upload dataset":
    st.title("Upload Dataset")
    file = st.file_uploader("Upload Your Dataset")

    if file: 
        df = pd.read_csv(file, index_col=None, delimiter=',')
        df.to_csv('dataset.csv', index=None)
        st.session_state.df = df

if choice == "Setup Data":
    df = st.session_state.df 
    st.title("Choose your target variable")
    
    targetVariable = st.selectbox('Choose your target variable"',df.columns.tolist())
    st.write(pd.factorize(df[targetVariable]))
    
    if targetVariable:
        st.write(df[targetVariable].unique())
    st.session_state.target_variable = targetVariable
    st.title("Rename columns")
    form = st.form(key=f"form_1")
    for x,i in enumerate(df.columns.tolist()):
        form.text_input('Feature ' + i,i, key=f"input_{x}")
    submit = form.form_submit_button("Update columns")
    if submit:
        new_columns = []
        for i in range(len(df.columns.tolist())):
            new_columns.append(st.session_state[f'input_{i}'])
        df.set_axis(new_columns, axis=1, inplace=True)
        df[targetVariable] = pd.factorize(df[targetVariable])[0]
        st.session_state.df = df


if choice == "Data Analyse":
    st.title("Data Analyse")
    df = st.session_state.df

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("NaN", df.isna().sum().sum())
    col4.metric("NaN", df.isna().sum().sum())

    with st.expander("Head"):
        st.dataframe(df.head(5))

    with st.expander("Tail"):
        st.dataframe(df.tail(5))

    with st.expander("Summery Non-categorial"):
        st.dataframe(df.describe())

    with st.expander("Target Variable"):
        st.bar_chart(df[st.session_state.target_variable].value_counts())
        st.dataframe(df[st.session_state.target_variable].value_counts())

    with st.expander("Correlation"):
        st.dataframe(df.corr())

    # Customizable Plot

    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

        # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)


        elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)

        elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)

        # Custom Plot 
        elif type_of_plot:
            cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

if choice == "Modelling":
    st.title("Model wordt gekozen.")
    df = st.session_state.df

    if st.button('Run Modelling'): 
        setup(df, target=st.session_state.target_variable)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)

    

if choice == "Visualise":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)