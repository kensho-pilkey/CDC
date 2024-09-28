import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_excel('Health_Science_Dataset.xlsx')
    return df

st.title("COVID-19, Pneumonia, and Influenza Mortality Dashboard")
st.write("This application visualizes mortality trends for COVID-19, pneumonia, and influenza using historical data.")

df = load_data()

if st.checkbox("Show raw data"):
    st.write(df.head())
