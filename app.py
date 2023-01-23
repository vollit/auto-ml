import streamlit as st
from operator import index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
import plotly.express as px
from interpret import show
from pycaret.classification import *
from interpret.provider import InlineProvider
from interpret import set_visualize_provider
set_visualize_provider(InlineProvider())

st.set_page_config(
    page_title="Interpretable AutoML",
    page_icon="✅",
    layout="wide",
)

with st.sidebar: 
    st.image("./assets/info-support-logo.png")
    st.title("Interpretable AI")
    st.info('This demo application is made to show the possibilities for interpretable AutoML.', icon="ℹ️")
    choice = st.radio("Steps", ["Upload dataset","Setup data","Data analysis","Modelling", "Visualise"])
    st.write('> Made by [Jonathan Vollmuller](https://www.linkedin.com/in/jonathanvollmuller/)  Interpretability project for **Info Support**   **Human Centered AI**')

 
if choice == "Upload dataset":
    st.title("Upload dataset")
    file = st.file_uploader("Upload Your Dataset")
    
    with open('./example-data/Breast_cancer_data.csv', 'rb') as f:
        st.download_button('Download example file', f, file_name='breast-cancer-data.csv')
    
    if file: 
        df = pd.read_csv(file, index_col=None, delimiter=',')
        df.to_csv('dataset.csv', index=None)
        st.session_state.df = df

if choice == "Setup data":

    try: 
        df = st.session_state.df 
    except AttributeError:
        st.error('Upload a dataset before going to the next step')
        st.stop()
        
    st.warning("Choose your target variable")
    
    targetVariable = st.selectbox('Choose your target variable"',df.columns.tolist())

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


if choice == "Data analysis":
    st.title("Data analysis")
    try: 
        df = st.session_state.df 
    except AttributeError:
        st.error('Upload a dataset before going to the next step')
        st.stop()

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
    st.title("Train multiple interpretable models")
    try: 
        df = st.session_state.df 
    except AttributeError:
        st.error('Upload a dataset before going to the next step')
        st.stop()
    
    with st.expander("Train settings"):
        trainSize = st.slider('Test / Train ratio ', 0, 100, 70, step=5)
        pcaCheck = st.checkbox('Use PCA')
        normaliseCheck = st.checkbox('Normalise data')
    

    if st.button('Run Modelling'): 
        with st.spinner('Training models...'):
            setup(df, target='diagnosis',pca = pcaCheck,train_size = (trainSize/100),normalize=normaliseCheck)
            setup_df = pull()
            
            ml = ExplainableBoostingClassifier()
            
            ebm = create_model(ml)
            st.session_state.ebm = ebm
            ebmmetric = pull()
            dt = create_model('dt')
            dtmetric = pull()
            lr = create_model('lr')
            lrmetric = pull()
            
            st.dataframe(pd.concat([ebmmetric.iloc[-2:-1].rename(index={'Mean': 'Explainable Boosting Classifier'}),dtmetric.iloc[-2:-1].rename(index={'Mean': 'Decision Tree'}),lrmetric.iloc[-2:-1].rename(index={'Mean': 'Linear Regression'})]),use_container_width=1)
            
            
            with st.expander("Explainable Boosting Classifier"):
                st.dataframe(ebmmetric,use_container_width=1)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    plot_model(ebm, plot='confusion_matrix',display_format='streamlit')
                with col2:
                    plot_model(ebm,display_format='streamlit')
                with col3:
                    plot_model(ebm,plot='learning',display_format='streamlit')
            
            with st.expander("Decision tree"):
                st.dataframe(dtmetric,use_container_width=1)
                col1, col2, col3 = st.columns(3)
                with col1:
                    plot_model(dt, plot='confusion_matrix',display_format='streamlit')
                with col2:
                    plot_model(dt,display_format='streamlit')
                with col3:
                    plot_model(dt,plot='learning',display_format='streamlit')
            
            with st.expander("Linear regression"):
                st.dataframe(lrmetric,use_container_width=1)
                col1, col2, col3 = st.columns(3)
                with col1:
                    plot_model(lr, plot='confusion_matrix',display_format='streamlit')
                with col2:
                    plot_model(lr,display_format='streamlit')
                with col3:
                    plot_model(lr, plot='learning',display_format='streamlit')
                
            st.success('Three interpretable models trained!', icon="✅")
        
if choice == "Visualise":
    st.title("Visualise")
    st.header('Global explanation')
    
    try: 
        ebm = st.session_state.ebm 
    except AttributeError:
        st.error('Train a model before going to the last step')
        st.stop()
    try: 
        df = st.session_state.df 
    except AttributeError:
        st.error('Upload a dataset before going to the next step')
        st.stop()
    
    ebm_global = ebm.explain_global()
    st.plotly_chart(ebm_global.visualize())
     
    features = st.multiselect('Choose features to visualise', df.columns)
    
    for x in range(len(features)):
        st.plotly_chart(ebm_global.visualize(x))
   
    st.header('Local explanation')
      
    y = df[st.session_state.target_variable]
    X = df.loc[:, df.columns != st.session_state.target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    ebm_local = ebm.explain_local(X_test, y_test)
 
    dfRow = st.number_input('Enter a row number to see the Local explanation of the test set', max_value=len(df.index),step=1)
    col1, col2 = st.columns(2)
    
    if dfRow  <= len(df.index):
        with col1:
            st.dataframe(df.iloc[dfRow].transpose(),use_container_width=1)
        with col2:
            st.plotly_chart(ebm_local.visualize(dfRow))
    else:
        st.warning('Row does not exist, please try another row number.')

   
        
        

