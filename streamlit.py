import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import us
from datetime import datetime

# Caching data loading for better performance
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

# Function to create the choropleth map with year annotation
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

    initial_year = pd.to_datetime(date).year
    fig.update_layout(
        annotations=[dict(
            x=0.5,
            y=1.05,
            xref='paper',
            yref='paper',
            text=f"Year: {initial_year}",
            showarrow=False,
            font=dict(size=16),
            align='center',
        )]
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

    steps = []
    for date in dates:
        date_str = str(date)
        step = dict(
            method='animate',
            args=[
                [date_str],
                {'mode': 'immediate', 'frame': {'duration': 500, 'redraw': True}, 'transition': {'duration': 0}}
            ],
            label=''  # Remove labels to avoid clutter
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': '', 'font': {'size': 16, 'color': '#666'}, 'visible': False, 'xanchor': 'right'},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title_text=f'{death_metric} Over Time',
        title_x=0.5,
        geo_scope='usa',
        geo_projection_type='albers usa',
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
            )],
            x=0.1,
            y=0,
            xanchor='right',
            yanchor='top'
        )]
    )

    return fig

# Streamlit app
st.title("COVID-19, Pneumonia, and Influenza Mortality Dashboard with Choropleth")

# Load data
disease_df, flight_df = load_data()
df = load_health_data()

# Sidebar for selecting death metric
death_metric = st.sidebar.selectbox(
    "Select Death Metric:",
    ['Total Deaths', 'Pneumonia Deaths', 'Influenza Deaths', 'Pneumonia or Influenza', 'Pneumonia, Influenza, or COVID-19 Deaths']
)

# Sidebar for selecting date range and which death types to show
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

# Display the flight graphs
st.subheader("Flu, COVID-19, Pneumonia Deaths vs Flight Cancellations")
flight_graphs(flight_df, disease_df, start_date, end_date, show_flu, show_covid, show_pneumonia, show_total)

# Display the choropleth map
st.subheader("Choropleth Map of Deaths by State")
choropleth_fig = create_choropleth_with_year_annotation(df, death_metric)
st.plotly_chart(choropleth_fig)
