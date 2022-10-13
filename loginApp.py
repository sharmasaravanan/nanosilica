import streamlit as st
import streamlit_authenticator as stauth
import yaml
import os
import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from PIL import Image
import joblib

# hashed_passwords = stauth.Hasher(['123', '456']).generate()
st.set_page_config(page_title='Compressive Strength prediction', layout='wide', initial_sidebar_state='auto')
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

scaler = joblib.load(os.path.join('model', 'scaler.pkl'))
model = joblib.load(os.path.join('model', 'adaModel.pkl'))

with open('./config.yml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    st.markdown("<h1 style='text-align: center; color: white;'>Compressive Strength prediction of Nano-Silica based"
                " concrete</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Using Artificial Neural Network!</h3>",
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
            testData['Compressive Strength after 28 days - MPa - Newton per Square millimeters'] = predicted

            df = testData.copy()
            df.columns = ["Cement", "Fine Aggregates", "Coarse Aggregates", "w/c", "sp", "Nano Silica",
                          'Compressive Strength']
            col1, col2 = st.columns(2)
            with col1:
                sns.scatterplot(y="Compressive Strength", x="Cement", hue="Nano Silica", size="Coarse Aggregates",
                                data=df)
                plt.title("Compressive Strength vs Cement-Nano Silica-Coarse Aggregates")
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                st.pyplot(plt)

            with col2:
                sns.scatterplot(y="Compressive Strength", x="Coarse Aggregates", hue="Fine Aggregates",
                                size="Nano Silica", data=df)
                plt.title("Compressive Strength vs Coarse aggregate-Fine Aggregates-Nano Silica")
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                st.pyplot(plt)

            col1, col2 = st.columns(2)
            with col1:
                plt.figure(figsize=(10, 10))
                sns.pairplot(df)
                st.pyplot(plt)

            with col2:
                plt.figure(figsize=(10, 10))
                sns.heatmap(df.corr(), annot=True, cmap='Blues')
                plt.title("Feature Correlation Heatmap")
                st.pyplot(plt)

            st.markdown("---")
            st.markdown("<h6 style='text-align: center; color: white;'>Compressive Strength for the given data is</h6>",
                        unsafe_allow_html=True)
            st.dataframe(testData)
            st.markdown("---")


elif not authentication_status:
    st.error('Username/password is incorrect')

elif authentication_status is None:
    st.warning('Please enter your username and password')

footer = """
<style> 
    a:link , a:visited{ color: blue; background-color: transparent; text-decoration: underline; }

    a:hover,  a:active { color: red; background-color: transparent; text-decoration: underline; }

    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #ffffffc4; color: black;
    text-align: center; }
</style>

<div class="footer">
    <p>Developed by <strong>Dr.N.K.Dhapekar </strong><a style='display: block; text-align: center;'>
    BE(Civil); M.Tech(Structural Engg.); Ph.D(Civil Engg.); AIE; INAAR; IBC. Ph No:- +91-9009051878</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
