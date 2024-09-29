import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import matplotlib.dates as mdates

@st.cache_data
def load_data():
    disease_df = pd.read_excel('Health_Science_Dataset.xlsx', header=1)
    flight_df = pd.read_csv("flights_sample_3m.csv")
    return disease_df, flight_df

def flight_graphs(start_date, end_date, show_flu, show_covid, show_pneumonia, show_total):
    # Convert FL_DATE to datetime
    flight_df['FL_DATE'] = pd.to_datetime(flight_df['FL_DATE'])

    # Convert start_date and end_date to pd.Timestamp
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter Dates based on user input
    df_filtered = flight_df[(flight_df['FL_DATE'] >= start_date) & (flight_df['FL_DATE'] <= end_date)]

    # Convert Week Ending Date to datetime in disease_df
    disease_df['Week Ending Date'] = pd.to_datetime(disease_df['Week Ending Date'])

    # Deaths data for each category
    deaths_by_week = disease_df.groupby('Week Ending Date')['Pneumonia, Influenza, or COVID-19 Deaths'].sum().reset_index()
    flu_deaths_by_week = disease_df.groupby('Week Ending Date')['Influenza Deaths'].sum().reset_index()
    covid_deaths_by_week = disease_df.groupby('Week Ending Date')['COVID-19 Deaths'].sum().reset_index()
    pneumonia_deaths_by_week = disease_df.groupby('Week Ending Date')['Pneumonia Deaths'].sum().reset_index()

    # Merged data for each category
    cancellations_by_week = df_filtered.groupby('FL_DATE')['CANCELLED'].sum().reset_index()
    flu_merged = pd.merge(flu_deaths_by_week, cancellations_by_week, left_on='Week Ending Date', right_on='FL_DATE', how='inner')
    covid_merged = pd.merge(covid_deaths_by_week, cancellations_by_week, left_on='Week Ending Date', right_on='FL_DATE', how='inner')
    pneumonia_merged = pd.merge(pneumonia_deaths_by_week, cancellations_by_week, left_on='Week Ending Date', right_on='FL_DATE', how='inner')
    total_merged = pd.merge(deaths_by_week, cancellations_by_week, left_on='Week Ending Date', right_on='FL_DATE', how='inner')

    fig, ax1 = plt.subplots()

    # Plot selected death types based on checkboxes
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Deaths', color='black')

    if show_flu:
        ax1.plot(flu_merged['Week Ending Date'], flu_merged['Influenza Deaths'], color='tab:green', label='Flu Deaths')
    if show_covid:
        ax1.plot(covid_merged['Week Ending Date'], covid_merged['COVID-19 Deaths'], color='tab:red', label='COVID-19 Deaths')
    if show_pneumonia:
        ax1.plot(pneumonia_merged['Week Ending Date'], pneumonia_merged['Pneumonia Deaths'], color='tab:orange', label='Pneumonia Deaths')
    if show_total:
        ax1.plot(total_merged['Week Ending Date'], total_merged['Pneumonia, Influenza, or COVID-19 Deaths'], color='tab:purple', label='Total Deaths')
    
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for flight cancellations
    ax2 = ax1.twinx()
    ax2.set_ylabel('Flight Cancellations', color='tab:blue')
    ax2.plot(total_merged['FL_DATE'], total_merged['CANCELLED'], color='tab:blue', label='Flight Cancellations')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Dynamically set x-axis major locator based on the date range
    date_range = (end_date - start_date).days
    if date_range > 365 * 2:
        ax1.xaxis.set_major_locator(mdates.YearLocator())  # Show one tick per year if range is more than 2 years
    elif date_range > 30 * 6:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show tick every 6 months if range is more than 6 months
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator())  # Show one tick per month if range is smaller

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Title and legends
    plt.title('Flu, COVID-19, Pneumonia Deaths vs Flight Cancellations')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Display the plot
    fig.tight_layout()
    st.pyplot(fig)

st.title("COVID-19, Pneumonia, and Influenza Mortality Dashboard")
st.write("This application visualizes mortality trends for COVID-19, pneumonia, and influenza using historical data.")

disease_df, flight_df = load_data()

with st.sidebar:
    st.title('📅 Select Time Range')
    
    start_date = st.date_input(
        'Start Date',
        min_value=datetime(2019, 12, 29),
        max_value=datetime(2023, 10, 28),
        value=datetime(2019, 12, 29)
    )
    
    end_date = st.date_input(
        'End Date',
        min_value=start_date,
        max_value=datetime(2023, 10, 28),
        value=datetime(2023, 10, 28)
    )
    
    show_flu = st.checkbox("Show Flu Deaths", value=True)
    show_covid = st.checkbox("Show COVID-19 Deaths", value=True)
    show_pneumonia = st.checkbox("Show Pneumonia Deaths", value=True)
    show_total = st.checkbox("Show Total Deaths", value=True)

flight_graphs(start_date, end_date, show_flu, show_covid, show_pneumonia, show_total)
