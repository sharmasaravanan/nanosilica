import os
import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import streamlit as st
from PIL import Image
import joblib


st.set_page_config(page_title='Nano Silica Analysis', layout='wide', initial_sidebar_state='auto')
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

scaler = joblib.load(os.path.join('model', 'scaler.pkl'))
model = joblib.load(os.path.join('model', 'adaModel.pkl'))

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.write(' ')

with col2:
    st.write(' ')

with col3:
    image = Image.open('templates//logo.jpeg')
    st.image(image, use_column_width='auto')

with col4:
    st.write(' ')

with col5:
    st.write(' ')


st.markdown("<h1 style='text-align: center; color: white;'>Nano Silica Structural Analysis</h1>",
            unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: white;'>Using Artificial Neural Network!</h6>",
            unsafe_allow_html=True)

st.markdown("### File Upload")
data_file = st.file_uploader("Upload structural analysis", type=["xlsx", 'csv'])
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.write(' ')

with col2:
    st.write(' ')

with col3:
    st.write(' ')

with col4:
    st.write(' ')

with col5:
    st.write(' ')

with col6:
    st.write(' ')

with col7:
    with open('Example.csv') as f:
        st.download_button('Download sample', f, file_name='Example.csv', mime='text/csv')

if data_file is not None:
    file_details = {"filename": data_file.name, "filetype": data_file.type, "filesize": data_file.size}

    with open(os.path.join("./testData/", data_file.name), "wb") as f:
        f.write(data_file.getbuffer())

    if st.button('Analyze'):
        if data_file.name.split(".") == "xlsx":
            testData = pd.read_excel("./testData/" + data_file.name)
        else:
            testData = pd.read_csv("./testData/" + data_file.name)

        x = testData.iloc[:, :6]
        st.write("You have successfully uploaded the data!")
        st.dataframe(x)

        with st.spinner('Wait for it...'):
            time.sleep(3)

        predicted = list(model.predict(scaler.transform(x)))
        testData['CompressiveStrength'] = predicted

        col1, col2 = st.columns(2)
        with col1:
            sns.scatterplot(y="CompressiveStrength", x="Cement", hue="nanoSilica", size="CoarseAggregates",
                            data=testData)
            plt.title("Predicted CC Strength vs (Cement, nanoSilica, CoarseAggregates)")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            st.pyplot(plt)

        with col2:
            sns.scatterplot(y="CompressiveStrength", x="CoarseAggregates", hue="FineAggregates", size="nanoSilica",
                            data=testData)
            plt.title("Predicted CC Strength vs (Coarse aggregate, Fine Aggregates, nanoSilica)")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            st.pyplot(plt)

        st.markdown("---")

        st.markdown("<h6 style='text-align: center; color: white;'>Compressive Strength for the given data is</h6>",
                    unsafe_allow_html=True)
        st.dataframe(testData)
        st.markdown("---")
