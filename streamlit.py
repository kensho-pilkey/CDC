import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import us
from datetime import datetime

import sys
print("Python executable:", sys.executable)
import numpy as np 
print("NumPy version:", np.__version__)
print("NumPy location:", np.__file__)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import sqlite3
import us
from ipywidgets import widgets, interact
import plotly.graph_objects as go
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import seaborn as sns
from prophet import Prophet
import plotly.express as px
import us
from kneed import KneeLocator
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from scipy.stats import f_oneway

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    disease_df = pd.read_excel('Health_Science_Dataset.xlsx', header=1)
    flight_df = pd.read_csv("flights_sample_3m.csv")
    return disease_df, flight_df

@st.cache_data
def load_health_data():
    df = pd.read_excel("Health_Science_Dataset.xlsx", header=1)

    for col_idx in [6, 7, 8, 9]:
        col_name = df.columns[col_idx]
        df[col_name] = df[col_name].astype(str)

    for col_idx in [0, 1, 2, 5]:
        col_name = df.columns[col_idx]
        df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    df['MMWRyear'] = df['MMWRyear'].astype(int)

    for col_idx in [4] + list(range(10, 16)):
        col_name = df.columns[col_idx]
        df[col_name] = df[col_name].apply(lambda x: int(x) if pd.notna(x) else x)

    # Map 'Jurisdiction' to state abbreviations
    state_abbrevs = {state.name: state.abbr for state in us.states.STATES}
    df['State_Abbr'] = df['Jurisdiction'].apply(lambda x: state_abbrevs.get(x, None))
    df = df[df['State_Abbr'].notna()]

    # Map 'State_Abbr' to FIPS codes
    abbr_to_fips = {state.abbr: state.fips for state in us.states.STATES}
    df['FIPS'] = df['State_Abbr'].map(abbr_to_fips)

    return df

# Function to plot deaths vs flight cancellations graph
def flight_graphs(flight_df, disease_df, start_date, end_date, show_flu, show_covid, show_pneumonia, show_total):
    # Convert FL_DATE to datetime
    flight_df['FL_DATE'] = pd.to_datetime(flight_df['FL_DATE'])

    # Convert start_date and end_date to pd.Timestamp
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter Dates based on user input
    df_filtered = flight_df[(flight_df['FL_DATE'] >= start_date) & (flight_df['FL_DATE'] <= end_date)]

    # Convert Week Ending Date to datetime in disease_df
    disease_df['Week Ending Date'] = pd.to_datetime(disease_df['Week Ending Date'])
    disease_df = disease_df[(disease_df['Jurisdiction'] == 'United States') & (disease_df['Age Group'] == 'All Ages')]
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
        ax1.xaxis.set_major_locator(mdates.YearLocator())
    elif date_range > 30 * 6:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator())

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

# Function to create the pie chart comparing total deaths by age group
def create_pie_chart_by_age_group(disease_df, start_date, end_date):
    # Filter data for the selected date range
    filtered_df = disease_df[
        (disease_df['Week Ending Date'] >= pd.to_datetime(start_date)) &
        (disease_df['Week Ending Date'] <= pd.to_datetime(end_date)) &
        (disease_df['Jurisdiction'] == 'United States')
    ]

    # Group by age group and sum total deaths
    age_group_df = filtered_df.groupby('Age Group')['Pneumonia, Influenza, or COVID-19 Deaths'].sum().reset_index()

    # Filter out "All Ages" for the pie chart
    age_group_df = age_group_df[age_group_df['Age Group'] != 'All Ages']

    red_shades = ['#FFCCCC', '#FF9999', '#FF6666']
    # Create pie chart
    fig = px.pie(
        age_group_df, 
        values='Pneumonia, Influenza, or COVID-19 Deaths', 
        names='Age Group',
        title="Total Deaths by Age Group",
        hole=0.3,
        color_discrete_sequence=red_shades
    )
    st.plotly_chart(fig)

def cluster_graphs(df):
    for col_idx in [6, 7, 8, 9]:
        col_name = df.columns[col_idx]
        df[col_name] = df[col_name].astype(str)

    # Dates
    for col_idx in [0, 1, 2, 5]:
        col_name = df.columns[col_idx]
        df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Year
    df['MMWRyear'] = df['MMWRyear'].astype(int)

    # Integer
    for col_idx in [4] + list(range(10, 16)):
        col_name = df.columns[col_idx]
        df[col_name] = df[col_name].apply(lambda x: int(x) if pd.notna(x) else x)

    # Construct 'Year-Week' string
    df['Year-Week'] = df['MMWRyear'].astype(str) + '-W' + df['MMWRweek'].astype(str).str.zfill(2)
    # Convert 'Year-Week' to datetime using ISO week date format
    df['Date'] = pd.to_datetime(df['Year-Week'] + '-1', format='%G-W%V-%u', errors='coerce')
    # Convert 'Date' to string
    df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')

    state_names = [state.name for state in us.states.STATES]
    state_abbrevs = [state.abbr for state in us.states.STATES]
    name_to_abbr = {state.name: state.abbr for state in us.states.STATES}

    def map_jurisdiction(jurisdiction):
        if jurisdiction in state_abbrevs:
            return jurisdiction
        elif jurisdiction in name_to_abbr:
            return name_to_abbr[jurisdiction]
        else:
            return None  # For unrecognized jurisdictions

    df['State_Abbr'] = df['Jurisdiction'].apply(map_jurisdiction)

    # Filter out unrecognized states
    df = df[df['State_Abbr'].notna()].copy()

    # Map 'State_Abbr' to FIPS codes
    abbr_to_fips = {state.abbr: state.fips for state in us.states.STATES}
    df['FIPS'] = df['State_Abbr'].map(abbr_to_fips)
    
    optimal_k = 2  # Adjust based on the Elbow curve
    metric = 'Total Deaths'

    # Pivot the dataframe to have states as rows and dates as columns
    df_pivot = df.pivot_table(
        index='State_Abbr',
        columns='Date',
        values=metric,
        aggfunc='sum'
    )

    # Fill missing values with zeros (or you can use interpolation)
    scaler = TimeSeriesScalerMeanVariance()
    df_pivot.fillna(0, inplace=True)
    X_scaled = scaler.fit_transform(df_pivot.values)
    
    # Convert to a time series dataset for clustering
    X_ts = to_time_series_dataset(X_scaled)
    
    # Apply TimeSeriesKMeans clustering
    km_dtw = TimeSeriesKMeans(n_clusters=optimal_k, metric="dtw", random_state=42)
    clusters = km_dtw.fit_predict(X_ts)
    df_pivot['Cluster'] = clusters

    # Plotting the cluster centroids
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(optimal_k):
        ax.plot(km_dtw.cluster_centers_[i].ravel(), label=f'Cluster {i}')
    ax.legend()
    ax.set_title(f'Cluster Centroids for {metric}')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Standardized Deaths')
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def create_choropleth_with_year_annotation(df, death_metric):
    if 'Date' not in df.columns:
        df['Year-Week'] = df['MMWRyear'].astype(str) + '-W' + df['MMWRweek'].astype(str).str.zfill(2)
        df['Date'] = pd.to_datetime(df['Year-Week'] + '-1', format='%G-W%V-%u', errors='coerce')

    if 'Date_Str' not in df.columns:
        df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')

    dates = sorted(df['Date'].unique())

    date = dates[0]
    df_date = df[df['Date'] == date]

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=df_date['State_Abbr'],
        z=df_date[death_metric],
        locationmode='USA-states',
        colorscale='Reds',
        colorbar_title='Number of Deaths',
        hovertext=df_date['Jurisdiction'],
        hoverinfo='text+z',
    ))

    # Set transparent background
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='albers usa',
            bgcolor='rgba(0,0,0,0)'  # Transparent map background
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        sliders=[dict(
            active=0,
            currentvalue={'prefix': '', 'font': {'size': 16, 'color': '#666'}, 'visible': False, 'xanchor': 'right'},
            steps=[dict(
                method='animate',
                args=[
                    [str(date)],
                    {'mode': 'immediate', 'frame': {'duration': 500, 'redraw': True}, 'transition': {'duration': 0}}
                ],
                label=''  # Hide labels
            ) for date in dates]
        )],
        annotations=[dict(
            x=0.5,
            y=1.05,
            xref='paper',
            yref='paper',
            text=f"Year: {pd.to_datetime(date).year}",
            showarrow=False,
            font=dict(size=16),
            align='center',
        )],
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True}],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}],
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                x=0.1,
                xanchor="right",
                y=0.1,
                yanchor="top"
            )
        ]
    )

    frames = []
    for date in dates:
        df_date = df[df['Date'] == date]
        year_str = pd.to_datetime(date).year
        frame = go.Frame(
            data=[go.Choropleth(
                locations=df_date['State_Abbr'],
                z=df_date[death_metric],
                locationmode='USA-states',
                colorscale='Reds',
                hovertext=df_date['Jurisdiction'],
                hoverinfo='text+z',
            )],
            name=str(date),
            layout=go.Layout(
                annotations=[dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text=f"Year: {year_str}",
                    showarrow=False,
                    font=dict(size=16),
                    align='center',
                )]
            )
        )
        frames.append(frame)

    fig.frames = frames

    return fig

# Function to create a bar chart comparing total deaths per state
def create_total_deaths_per_state_chart(disease_df, start_date, end_date, show_flu, show_covid, show_pneumonia):
    # Filter data for the selected date range and 'All Ages'
    filtered_df = disease_df[
        (disease_df['Week Ending Date'] >= pd.to_datetime(start_date)) &
        (disease_df['Week Ending Date'] <= pd.to_datetime(end_date)) &
        (disease_df['Age Group'] == 'All Ages')
    ]

    # Exclude non-state jurisdictions
    excluded_jurisdictions = ['United States'] + [f'HHS Region {i}' for i in range(1, 11)]
    filtered_df = filtered_df[~filtered_df['Jurisdiction'].isin(excluded_jurisdictions)]

    # Select relevant columns for the selected diseases
    columns_to_sum = []
    if show_flu:
        columns_to_sum.append('Influenza Deaths')
    if show_covid:
        columns_to_sum.append('COVID-19 Deaths')
    if show_pneumonia:
        columns_to_sum.append('Pneumonia Deaths')

    # Group by state and sum the deaths
    state_deaths_df = filtered_df.groupby('Jurisdiction')[columns_to_sum].sum().reset_index()

    # Create a bar chart
    state_deaths_df['Total Deaths'] = state_deaths_df[columns_to_sum].sum(axis=1)
    fig = px.bar(
        state_deaths_df,
        x='Jurisdiction',
        y='Total Deaths',
        title="Total Deaths per State",
        labels={'Jurisdiction': 'State', 'Total Deaths': 'Number of Deaths'},
        text_auto=True,
        color_discrete_sequence=['#FFCCCC']
    )
    st.plotly_chart(fig)


st.title("COVID-19, Pneumonia, and Influenza Mortality Dashboard with Choropleth")
st.write("")
disease_df, flight_df = load_data()
df = load_health_data()


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

    st.title('🗺️ Map Settings')

    death_metric = st.selectbox(
        "Map Death Metric:",
        ['Total Deaths', 'Pneumonia Deaths', 'Influenza Deaths', 'Pneumonia or Influenza', 'Pneumonia, Influenza, or COVID-19 Deaths']
    )

# Display the flight graphs
col1, col2 = st.columns(2)

with col1:
    flight_graphs(flight_df, disease_df, start_date, end_date, show_flu, show_covid, show_pneumonia, show_total)

with col2:
    create_pie_chart_by_age_group(disease_df, start_date, end_date)

# Display the total deaths chart
st.subheader("Total Deaths by State (for Selected Diseases)")
create_total_deaths_per_state_chart(disease_df, start_date, end_date, show_flu, show_covid, show_pneumonia)

# Create new columns for the choropleth map and cluster graph
col1, col2 = st.columns(2)

with col1:
    st.subheader("Choropleth Map of Deaths by State")
    choropleth_fig = create_choropleth_with_year_annotation(df, death_metric)
    st.plotly_chart(choropleth_fig)

with col2:
    st.subheader("Cluster Graph")
    st.write("")
    cluster_graphs(disease_df)
