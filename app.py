import os
import argparse
import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
from collections import defaultdict
import csv 
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

import requests
import tempfile
import shutil
import xarray as xr

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False

import tropycal.tracks as tracks

# -----------------------------
# Configuration and Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more details
    format='%(asctime)s - %(levelname)s - %(message)s'
)

parser = argparse.ArgumentParser(description='Typhoon Analysis Dashboard')
parser.add_argument('--data_path', type=str, default=os.getcwd(), help='Path to the data directory')
args = parser.parse_args()
DATA_PATH = args.data_path

# Data paths
ONI_DATA_PATH = os.path.join(DATA_PATH, 'oni_data.csv')
TYPHOON_DATA_PATH = os.path.join(DATA_PATH, 'processed_typhoon_data.csv')
MERGED_DATA_CSV = os.path.join(DATA_PATH, 'merged_typhoon_era5_data.csv')  # used in other tabs

# IBTrACS settings (only used for updating typhoon options)
BASIN_FILES = {
    'EP': 'ibtracs.EP.list.v04r01.csv',
    'NA': 'ibtracs.NA.list.v04r01.csv',
    'WP': 'ibtracs.WP.list.v04r01.csv'
}
IBTRACS_BASE_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/'
LOCAL_IBTRACS_PATH = os.path.join(DATA_PATH, 'ibtracs.WP.list.v04r01.csv')
CACHE_FILE = os.path.join(DATA_PATH, 'ibtracs_cache.pkl')
CACHE_EXPIRY_DAYS = 1

# -----------------------------
# Color Maps and Standards
# -----------------------------
color_map = {
    'C5 Super Typhoon': 'rgb(255, 0, 0)',
    'C4 Very Strong Typhoon': 'rgb(255, 165, 0)',
    'C3 Strong Typhoon': 'rgb(255, 255, 0)',
    'C2 Typhoon': 'rgb(0, 255, 0)',
    'C1 Typhoon': 'rgb(0, 255, 255)',
    'Tropical Storm': 'rgb(0, 0, 255)',
    'Tropical Depression': 'rgb(128, 128, 128)'
}
atlantic_standard = {
    'C5 Super Typhoon': {'wind_speed': 137, 'color': 'Red', 'hex': '#FF0000'},
    'C4 Very Strong Typhoon': {'wind_speed': 113, 'color': 'Orange', 'hex': '#FFA500'},
    'C3 Strong Typhoon': {'wind_speed': 96, 'color': 'Yellow', 'hex': '#FFFF00'},
    'C2 Typhoon': {'wind_speed': 83, 'color': 'Green', 'hex': '#00FF00'},
    'C1 Typhoon': {'wind_speed': 64, 'color': 'Cyan', 'hex': '#00FFFF'},
    'Tropical Storm': {'wind_speed': 34, 'color': 'Blue', 'hex': '#0000FF'},
    'Tropical Depression': {'wind_speed': 0, 'color': 'Gray', 'hex': '#808080'}
}
taiwan_standard = {
    'Strong Typhoon': {'wind_speed': 51.0, 'color': 'Red', 'hex': '#FF0000'},
    'Medium Typhoon': {'wind_speed': 33.7, 'color': 'Orange', 'hex': '#FFA500'},
    'Mild Typhoon': {'wind_speed': 17.2, 'color': 'Yellow', 'hex': '#FFFF00'},
    'Tropical Depression': {'wind_speed': 0, 'color': 'Gray', 'hex': '#808080'}
}

# -----------------------------
# Season and Regions
# -----------------------------
season_months = {
    'all': list(range(1, 13)),
    'summer': [6, 7, 8],
    'winter': [12, 1, 2]
}
regions = {
    "Taiwan Land": {"lat_min": 21.8, "lat_max": 25.3, "lon_min": 119.5, "lon_max": 122.1},
    "Taiwan Sea": {"lat_min": 19, "lat_max": 28, "lon_min": 117, "lon_max": 125},
    "Japan": {"lat_min": 20, "lat_max": 45, "lon_min": 120, "lon_max": 150},
    "China": {"lat_min": 18, "lat_max": 53, "lon_min": 73, "lon_max": 135},
    "Hong Kong": {"lat_min": 21.5, "lat_max": 23, "lon_min": 113, "lon_max": 115},
    "Philippines": {"lat_min": 5, "lat_max": 21, "lon_min": 115, "lon_max": 130}
}

# -----------------------------
# ONI and Typhoon Data Functions
# -----------------------------
def download_oni_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)
    return True

def convert_oni_ascii_to_csv(input_file, output_file):
    data = defaultdict(lambda: [''] * 12)
    season_to_month = {'DJF':12, 'JFM':1, 'FMA':2, 'MAM':3, 'AMJ':4, 'MJJ':5,
                       'JJA':6, 'JAS':7, 'ASO':8, 'SON':9, 'OND':10, 'NDJ':11}
    with open(input_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                season, year, anom = parts[0], parts[1], parts[-1]
                if season in season_to_month:
                    month = season_to_month[season]
                    if season == 'DJF':
                        year = str(int(year)-1)
                    data[year][month-1] = anom
    with open(output_file, 'w', newline='') as f:
        writer = pd.ExcelWriter(f)
        writer = csv.writer(f)
        writer.writerow(['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        for year in sorted(data.keys()):
            writer.writerow([year] + data[year])

def update_oni_data():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    temp_file = os.path.join(DATA_PATH, "temp_oni.ascii.txt")
    input_file = os.path.join(DATA_PATH, "oni.ascii.txt")
    output_file = ONI_DATA_PATH
    if download_oni_file(url, temp_file):
        if not os.path.exists(input_file) or not os.path.exists(output_file):
            os.replace(temp_file, input_file)
            convert_oni_ascii_to_csv(input_file, output_file)
        else:
            os.remove(temp_file)

def load_data(oni_path, typhoon_path):
    if not os.path.exists(typhoon_path):
        logging.error(f"Typhoon data file not found: {typhoon_path}")
        return pd.DataFrame(), pd.DataFrame()
    try:
        oni_data = pd.read_csv(oni_path)
        typhoon_data = pd.read_csv(typhoon_path, low_memory=False)
        typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
        typhoon_data = typhoon_data.dropna(subset=['ISO_TIME'])
        return oni_data, typhoon_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_oni_data(oni_data):
    oni_long = oni_data.melt(id_vars=['Year'], var_name='Month', value_name='ONI')
    month_map = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                 'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    oni_long['Month'] = oni_long['Month'].map(month_map)
    oni_long['Date'] = pd.to_datetime(oni_long['Year'].astype(str)+'-'+oni_long['Month']+'-01')
    oni_long['ONI'] = pd.to_numeric(oni_long['ONI'], errors='coerce')
    return oni_long

def process_typhoon_data(typhoon_data):
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    typhoon_data['USA_WIND'] = pd.to_numeric(typhoon_data['USA_WIND'], errors='coerce')
    typhoon_data['USA_PRES'] = pd.to_numeric(typhoon_data['USA_PRES'], errors='coerce')
    typhoon_data['LON'] = pd.to_numeric(typhoon_data['LON'], errors='coerce')
    logging.info(f"Unique basins in typhoon_data: {typhoon_data['SID'].str[:2].unique()}")
    typhoon_max = typhoon_data.groupby('SID').agg({
        'USA_WIND':'max','USA_PRES':'min','ISO_TIME':'first','SEASON':'first','NAME':'first',
        'LAT':'first','LON':'first'
    }).reset_index()
    typhoon_max['Month'] = typhoon_max['ISO_TIME'].dt.strftime('%m')
    typhoon_max['Year'] = typhoon_max['ISO_TIME'].dt.year
    typhoon_max['Category'] = typhoon_max['USA_WIND'].apply(categorize_typhoon)
    return typhoon_max

def merge_data(oni_long, typhoon_max):
    return pd.merge(typhoon_max, oni_long, on=['Year','Month'])

def categorize_typhoon(wind_speed):
    if wind_speed >= 137:
        return 'C5 Super Typhoon'
    elif wind_speed >= 113:
        return 'C4 Very Strong Typhoon'
    elif wind_speed >= 96:
        return 'C3 Strong Typhoon'
    elif wind_speed >= 83:
        return 'C2 Typhoon'
    elif wind_speed >= 64:
        return 'C1 Typhoon'
    elif wind_speed >= 34:
        return 'Tropical Storm'
    else:
        return 'Tropical Depression'

def classify_enso_phases(oni_value):
    if isinstance(oni_value, pd.Series):
        oni_value = oni_value.iloc[0]
    if oni_value >= 0.5:
        return 'El Nino'
    elif oni_value <= -0.5:
        return 'La Nina'
    else:
        return 'Neutral'

# ------------- Regression Functions -------------
def perform_wind_regression(start_year, start_month, end_year, end_month):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].dropna(subset=['USA_WIND','ONI'])
    data['severe_typhoon'] = (data['USA_WIND']>=64).astype(int)
    X = sm.add_constant(data['ONI'])
    y = data['severe_typhoon']
    model = sm.Logit(y, X).fit(disp=0)
    beta_1 = model.params['ONI']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['ONI']
    return f"Wind Regression: β1={beta_1:.4f}, Odds Ratio={exp_beta_1:.4f}, P-value={p_value:.4f}"

def perform_pressure_regression(start_year, start_month, end_year, end_month):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].dropna(subset=['USA_PRES','ONI'])
    data['intense_typhoon'] = (data['USA_PRES']<=950).astype(int)
    X = sm.add_constant(data['ONI'])
    y = data['intense_typhoon']
    model = sm.Logit(y, X).fit(disp=0)
    beta_1 = model.params['ONI']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['ONI']
    return f"Pressure Regression: β1={beta_1:.4f}, Odds Ratio={exp_beta_1:.4f}, P-value={p_value:.4f}"

def perform_longitude_regression(start_year, start_month, end_year, end_month):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].dropna(subset=['LON','ONI'])
    data['western_typhoon'] = (data['LON']<=140).astype(int)
    X = sm.add_constant(data['ONI'])
    y = data['western_typhoon']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    beta_1 = model.params['ONI']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['ONI']
    return f"Longitude Regression: β1={beta_1:.4f}, Odds Ratio={exp_beta_1:.4f}, P-value={p_value:.4f}"

# ------------- IBTrACS Data Loading -------------
def load_ibtracs_data():
    ibtracs_data = {}
    for basin, filename in BASIN_FILES.items():
        local_path = os.path.join(DATA_PATH, filename)
        if not os.path.exists(local_path):
            logging.info(f"Downloading {basin} basin file...")
            response = requests.get(IBTRACS_BASE_URL+filename)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded {basin} basin file.")
        try:
            logging.info(f"--> Starting to read in IBTrACS data for basin {basin}")
            ds = tracks.TrackDataset(source='ibtracs', ibtracs_url=local_path)
            logging.info(f"--> Completed reading in IBTrACS data for basin {basin}")
            ibtracs_data[basin] = ds
        except ValueError as e:
            logging.warning(f"Skipping basin {basin} due to error: {e}")
            ibtracs_data[basin] = None
    return ibtracs_data

ibtracs = load_ibtracs_data()

# ------------- Load & Process Data -------------
update_oni_data()
oni_data, typhoon_data = load_data(ONI_DATA_PATH, TYPHOON_DATA_PATH)
oni_long = process_oni_data(oni_data)
typhoon_max = process_typhoon_data(typhoon_data)
merged_data = merge_data(oni_long, typhoon_max)

# ------------- Visualization Functions -------------
def generate_typhoon_tracks(filtered_data, typhoon_search):
    fig = go.Figure()
    for sid in filtered_data['SID'].unique():
        storm_data = filtered_data[filtered_data['SID'] == sid]
        phase = storm_data['ENSO_Phase'].iloc[0]
        color = {'El Nino':'red','La Nina':'blue','Neutral':'green'}.get(phase, 'black')
        fig.add_trace(go.Scattergeo(
            lon=storm_data['LON'], lat=storm_data['LAT'], mode='lines',
            name=storm_data['NAME'].iloc[0], line=dict(width=2, color=color)
        ))
    if typhoon_search:
        mask = filtered_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if mask.any():
            storm_data = filtered_data[mask]
            fig.add_trace(go.Scattergeo(
                lon=storm_data['LON'], lat=storm_data['LAT'], mode='lines',
                name=f'Matched: {typhoon_search}', line=dict(width=5, color='yellow')
            ))
    fig.update_layout(
        title='Typhoon Tracks',
        geo=dict(projection_type='natural earth', showland=True),
        height=700
    )
    return fig

def generate_wind_oni_scatter(filtered_data, typhoon_search):
    fig = px.scatter(filtered_data, x='ONI', y='USA_WIND', color='Category',
                     hover_data=['NAME','Year','Category'],
                     title='Wind Speed vs ONI',
                     labels={'ONI':'ONI Value','USA_WIND':'Max Wind Speed (knots)'},
                     color_discrete_map=color_map)
    if typhoon_search:
        mask = filtered_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=filtered_data.loc[mask,'ONI'], y=filtered_data.loc[mask,'USA_WIND'],
                mode='markers', marker=dict(size=10, color='red', symbol='star'),
                name=f'Matched: {typhoon_search}',
                text=filtered_data.loc[mask,'NAME']+' ('+filtered_data.loc[mask,'Year'].astype(str)+')'
            ))
    return fig

def generate_pressure_oni_scatter(filtered_data, typhoon_search):
    fig = px.scatter(filtered_data, x='ONI', y='USA_PRES', color='Category',
                     hover_data=['NAME','Year','Category'],
                     title='Pressure vs ONI',
                     labels={'ONI':'ONI Value','USA_PRES':'Min Pressure (hPa)'},
                     color_discrete_map=color_map)
    if typhoon_search:
        mask = filtered_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=filtered_data.loc[mask,'ONI'], y=filtered_data.loc[mask,'USA_PRES'],
                mode='markers', marker=dict(size=10, color='red', symbol='star'),
                name=f'Matched: {typhoon_search}',
                text=filtered_data.loc[mask,'NAME']+' ('+filtered_data.loc[mask,'Year'].astype(str)+')'
            ))
    return fig

def generate_regression_analysis(filtered_data):
    fig = px.scatter(filtered_data, x='LON', y='ONI', hover_data=['NAME'],
                     title='Typhoon Generation Longitude vs ONI (All Years)')
    if len(filtered_data) > 1:
        X = np.array(filtered_data['LON']).reshape(-1,1)
        y = filtered_data['ONI']
        model = sm.OLS(y, sm.add_constant(X)).fit()
        y_pred = model.predict(sm.add_constant(X))
        fig.add_trace(go.Scatter(x=filtered_data['LON'], y=y_pred, mode='lines', name='Regression Line'))
        slope = model.params[1]
        slopes_text = f"All Years Slope: {slope:.4f}"
    else:
        slopes_text = "Insufficient data for regression"
    return fig, slopes_text

def generate_main_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    filtered_data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].copy()
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    if enso_phase != 'all':
        filtered_data = filtered_data[filtered_data['ENSO_Phase'] == enso_phase.capitalize()]
    tracks_fig = generate_typhoon_tracks(filtered_data, typhoon_search)
    wind_scatter = generate_wind_oni_scatter(filtered_data, typhoon_search)
    pressure_scatter = generate_pressure_oni_scatter(filtered_data, typhoon_search)
    regression_fig, slopes_text = generate_regression_analysis(filtered_data)
    return tracks_fig, wind_scatter, pressure_scatter, regression_fig, slopes_text

def get_full_tracks(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    filtered_data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].copy()
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    if enso_phase != 'all':
        filtered_data = filtered_data[filtered_data['ENSO_Phase'] == enso_phase.capitalize()]
    unique_storms = filtered_data['SID'].unique()
    count = len(unique_storms)
    fig = go.Figure()
    for sid in unique_storms:
        storm_data = typhoon_data[typhoon_data['SID']==sid]
        name = storm_data['NAME'].iloc[0] if pd.notnull(storm_data['NAME'].iloc[0]) else "Unnamed"
        storm_oni = filtered_data[filtered_data['SID']==sid]['ONI'].iloc[0]
        color = 'red' if storm_oni>=0.5 else ('blue' if storm_oni<=-0.5 else 'green')
        fig.add_trace(go.Scattergeo(
            lon=storm_data['LON'], lat=storm_data['LAT'], mode='lines',
            name=f"{name} ({storm_data['SEASON'].iloc[0]})",
            line=dict(width=1.5, color=color), hoverinfo="name"
        ))
    if typhoon_search:
        search_mask = typhoon_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if search_mask.any():
            for sid in typhoon_data[search_mask]['SID'].unique():
                storm_data = typhoon_data[typhoon_data['SID']==sid]
                fig.add_trace(go.Scattergeo(
                    lon=storm_data['LON'], lat=storm_data['LAT'], mode='lines+markers',
                    name=f"MATCHED: {storm_data['NAME'].iloc[0]} ({storm_data['SEASON'].iloc[0]})",
                    line=dict(width=3, color='yellow'),
                    marker=dict(size=5), hoverinfo="name"
                ))
    fig.update_layout(
        title=f"Typhoon Tracks ({start_year}-{start_month} to {end_year}-{end_month})",
        geo=dict(
            projection_type='natural earth',
            showland=True,
            showcoastlines=True,
            landcolor='rgb(243,243,243)',
            countrycolor='rgb(204,204,204)',
            coastlinecolor='rgb(204,204,204)',
            center=dict(lon=140, lat=20),
            projection_scale=3
        ),
        legend_title="Typhoons by ENSO Phase",
        showlegend=True,
        height=700
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text="Red: El Niño, Blue: La Niña, Green: Neutral",
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)"
    )
    return fig, f"Total typhoons displayed: {count}"

def get_wind_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    results = generate_main_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search)
    regression = perform_wind_regression(start_year, start_month, end_year, end_month)
    return results[1], regression

def get_pressure_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    results = generate_main_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search)
    regression = perform_pressure_regression(start_year, start_month, end_year, end_month)
    return results[2], regression

def get_longitude_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    results = generate_main_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search)
    regression = perform_longitude_regression(start_year, start_month, end_year, end_month)
    return results[3], results[4], regression

def categorize_typhoon_by_standard(wind_speed, standard='atlantic'):
    if standard=='taiwan':
        wind_speed_ms = wind_speed * 0.514444
        if wind_speed_ms >= 51.0:
            return 'Strong Typhoon', taiwan_standard['Strong Typhoon']['hex']
        elif wind_speed_ms >= 33.7:
            return 'Medium Typhoon', taiwan_standard['Medium Typhoon']['hex']
        elif wind_speed_ms >= 17.2:
            return 'Mild Typhoon', taiwan_standard['Mild Typhoon']['hex']
        return 'Tropical Depression', taiwan_standard['Tropical Depression']['hex']
    else:
        if wind_speed >= 137:
            return 'C5 Super Typhoon', atlantic_standard['C5 Super Typhoon']['hex']
        elif wind_speed >= 113:
            return 'C4 Very Strong Typhoon', atlantic_standard['C4 Very Strong Typhoon']['hex']
        elif wind_speed >= 96:
            return 'C3 Strong Typhoon', atlantic_standard['C3 Strong Typhoon']['hex']
        elif wind_speed >= 83:
            return 'C2 Typhoon', atlantic_standard['C2 Typhoon']['hex']
        elif wind_speed >= 64:
            return 'C1 Typhoon', atlantic_standard['C1 Typhoon']['hex']
        elif wind_speed >= 34:
            return 'Tropical Storm', atlantic_standard['Tropical Storm']['hex']
        return 'Tropical Depression', atlantic_standard['Tropical Depression']['hex']

# ------------- Updated TSNE Cluster Function with Mean Curves -------------
def update_route_clusters(start_year, start_month, end_year, end_month, enso_value, season):
    try:
        # Merge raw typhoon data with ONI so that each storm has multiple points.
        raw_data = typhoon_data.copy()
        raw_data['Year'] = raw_data['ISO_TIME'].dt.year
        raw_data['Month'] = raw_data['ISO_TIME'].dt.strftime('%m')
        merged_raw = pd.merge(raw_data, process_oni_data(oni_data), on=['Year','Month'], how='left')
        
        # Filter by date
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 28)
        merged_raw = merged_raw[(merged_raw['ISO_TIME'] >= start_date) & (merged_raw['ISO_TIME'] <= end_date)]
        logging.info(f"Total points after date filtering: {merged_raw.shape[0]}")
        
        # Filter by ENSO phase if specified
        merged_raw['ENSO_Phase'] = merged_raw['ONI'].apply(classify_enso_phases)
        if enso_value != 'all':
            merged_raw = merged_raw[merged_raw['ENSO_Phase'] == enso_value.capitalize()]
        logging.info(f"Total points after ENSO filtering: {merged_raw.shape[0]}")
        
        # Apply regional filter for Western Pacific (adjust boundaries as needed)
        wp_data = merged_raw[(merged_raw['LON'] >= 100) & (merged_raw['LON'] <= 180) &
                             (merged_raw['LAT'] >= 0) & (merged_raw['LAT'] <= 40)]
        logging.info(f"Total points after WP regional filtering: {wp_data.shape[0]}")
        if wp_data.empty:
            logging.info("WP regional filter returned no data; using all filtered data.")
            wp_data = merged_raw
        
        # Group by storm ID (SID); each group must have at least 2 observations
        all_storms_data = []
        for sid, group in wp_data.groupby('SID'):
            group = group.sort_values('ISO_TIME')
            times = pd.to_datetime(group['ISO_TIME']).values
            lats = group['LAT'].astype(float).values
            lons = group['LON'].astype(float).values
            if len(lons) < 2:
                continue
            # Also store wind and pressure for interpolation
            wind = group['USA_WIND'].astype(float).values if 'USA_WIND' in group.columns else None
            pres = group['USA_PRES'].astype(float).values if 'USA_PRES' in group.columns else None
            all_storms_data.append((sid, lons, lats, times, wind, pres))
        logging.info(f"Storms available for TSNE after grouping: {len(all_storms_data)}")
        if not all_storms_data:
            return go.Figure(), go.Figure(), make_subplots(rows=2, cols=1), "No valid storms for clustering."
        
        # Interpolate each storm's route (and wind/pressure) to a common length
        max_length = max(len(item[1]) for item in all_storms_data)
        route_vectors = []
        wind_curves = []
        pres_curves = []
        storm_ids = []
        for sid, lons, lats, times, wind, pres in all_storms_data:
            t = np.linspace(0, 1, len(lons))
            t_new = np.linspace(0, 1, max_length)
            try:
                lon_interp = interp1d(t, lons, kind='linear', fill_value='extrapolate')(t_new)
                lat_interp = interp1d(t, lats, kind='linear', fill_value='extrapolate')(t_new)
            except Exception as ex:
                logging.error(f"Interpolation error for storm {sid}: {ex}")
                continue
            route_vector = np.column_stack((lon_interp, lat_interp)).flatten()
            if np.isnan(route_vector).any():
                continue
            route_vectors.append(route_vector)
            storm_ids.append(sid)
            # Interpolate wind and pressure if available; otherwise, fill with NaN
            if wind is not None and len(wind) >= 2:
                try:
                    wind_interp = interp1d(t, wind, kind='linear', fill_value='extrapolate')(t_new)
                except Exception as ex:
                    logging.error(f"Wind interpolation error for storm {sid}: {ex}")
                    wind_interp = np.full(max_length, np.nan)
            else:
                wind_interp = np.full(max_length, np.nan)
            if pres is not None and len(pres) >= 2:
                try:
                    pres_interp = interp1d(t, pres, kind='linear', fill_value='extrapolate')(t_new)
                except Exception as ex:
                    logging.error(f"Pressure interpolation error for storm {sid}: {ex}")
                    pres_interp = np.full(max_length, np.nan)
            else:
                pres_interp = np.full(max_length, np.nan)
            wind_curves.append(wind_interp)
            pres_curves.append(pres_interp)
        logging.info(f"Storms with valid route vectors: {len(route_vectors)}")
        if len(route_vectors) == 0:
            return go.Figure(), go.Figure(), make_subplots(rows=2, cols=1), "No valid storms after interpolation."
        
        route_vectors = np.array(route_vectors)
        wind_curves = np.array(wind_curves)
        pres_curves = np.array(pres_curves)
        
        # Run TSNE on route vectors
        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        tsne_results = tsne.fit_transform(route_vectors)
        
        # Dynamic DBSCAN: choose eps so that we have roughly 5 to 20 clusters if possible
        selected_labels = None
        selected_eps = None
        for eps in np.linspace(1.0, 10.0, 91):
            dbscan = DBSCAN(eps=eps, min_samples=3)
            labels = dbscan.fit_predict(tsne_results)
            clusters = set(labels) - {-1}
            num_clusters = len(clusters)
            if 5 <= num_clusters <= 20:
                selected_labels = labels
                selected_eps = eps
                break
        if selected_labels is None:
            selected_eps = 5.0
            dbscan = DBSCAN(eps=selected_eps, min_samples=3)
            selected_labels = dbscan.fit_predict(tsne_results)
        logging.info(f"Selected DBSCAN eps: {selected_eps:.2f} yielding {len(set(selected_labels) - {-1})} clusters.")
        
        # TSNE scatter plot
        fig_tsne = go.Figure()
        colors = px.colors.qualitative.Safe
        unique_labels = sorted(set(selected_labels) - {-1})
        for i, label in enumerate(unique_labels):
            indices = np.where(selected_labels == label)[0]
            fig_tsne.add_trace(go.Scatter(
                x=tsne_results[indices, 0],
                y=tsne_results[indices, 1],
                mode='markers',
                marker=dict(color=colors[i % len(colors)]),
                name=f"Cluster {label}"
            ))
        noise_indices = np.where(selected_labels == -1)[0]
        if len(noise_indices) > 0:
            fig_tsne.add_trace(go.Scatter(
                x=tsne_results[noise_indices, 0],
                y=tsne_results[noise_indices, 1],
                mode='markers',
                marker=dict(color='grey'),
                name='Noise'
            ))
        fig_tsne.update_layout(
            title="t-SNE of Storm Routes",
            xaxis_title="t-SNE Dim 1",
            yaxis_title="t-SNE Dim 2"
        )
        
        # For each cluster, compute mean route, mean wind curve, and mean pressure curve.
        fig_routes = go.Figure()
        cluster_stats = []  # To hold mean curves for wind and pressure
        for i, label in enumerate(unique_labels):
            indices = np.where(selected_labels == label)[0]
            cluster_ids = [storm_ids[j] for j in indices]
            cluster_vectors = route_vectors[indices, :]
            mean_vector = np.mean(cluster_vectors, axis=0)
            mean_route = mean_vector.reshape((max_length, 2))
            mean_lon = mean_route[:, 0]
            mean_lat = mean_route[:, 1]
            fig_routes.add_trace(go.Scattergeo(
                lon=mean_lon,
                lat=mean_lat,
                mode='lines',
                line=dict(width=4, color=colors[i % len(colors)]),
                name=f"Cluster {label} Mean Route"
            ))
            # Get storms in this cluster from wp_data by SID
            cluster_raw = wp_data[wp_data['SID'].isin(cluster_ids)]
            # For each storm in the cluster, we already interpolated wind_curves and pres_curves.
            cluster_winds = wind_curves[indices, :]  # shape: (#storms, max_length)
            cluster_pres = pres_curves[indices, :]    # shape: (#storms, max_length)
            # Compute mean curves (if available)
            if cluster_winds.size > 0:
                mean_wind_curve = np.nanmean(cluster_winds, axis=0)
            else:
                mean_wind_curve = np.full(max_length, np.nan)
            if cluster_pres.size > 0:
                mean_pres_curve = np.nanmean(cluster_pres, axis=0)
            else:
                mean_pres_curve = np.full(max_length, np.nan)
            cluster_stats.append((label, mean_wind_curve, mean_pres_curve))
        
        # Create cluster stats plot with curves vs normalized route index (0 to 1)
        x_axis = np.linspace(0, 1, max_length)
        fig_stats = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Mean Wind Speed (knots)", "Mean MSLP (hPa)"))
        for i, (label, wind_curve, pres_curve) in enumerate(cluster_stats):
            fig_stats.add_trace(go.Scatter(
                x=x_axis,
                y=wind_curve,
                mode='lines',
                line=dict(width=2, color=colors[i % len(colors)]),
                name=f"Cluster {label} Mean Wind"
            ), row=1, col=1)
            fig_stats.add_trace(go.Scatter(
                x=x_axis,
                y=pres_curve,
                mode='lines',
                line=dict(width=2, color=colors[i % len(colors)]),
                name=f"Cluster {label} Mean MSLP"
            ), row=2, col=1)
        fig_stats.update_layout(
            title="Cluster Mean Curves",
            xaxis_title="Normalized Route Index",
            yaxis_title="Mean Wind Speed (knots)",
            xaxis2_title="Normalized Route Index",
            yaxis2_title="Mean MSLP (hPa)",
            showlegend=True
        )
        
        info = f"TSNE clustering complete. Selected eps: {selected_eps:.2f}. Clusters: {len(unique_labels)}."
        return fig_tsne, fig_routes, fig_stats, info
    except Exception as e:
        logging.error(f"Error in TSNE clustering: {e}")
        return go.Figure(), go.Figure(), make_subplots(rows=2, cols=1), f"Error in TSNE clustering: {e}"

# ------------- Animation Functions Using Processed CSV & Stock Map -------------
def generate_track_video_from_csv(year, storm_id, standard):
    storm_df = typhoon_data[typhoon_data['SID'] == storm_id].copy()
    if storm_df.empty:
        logging.error(f"No data found for storm: {storm_id}")
        return None
    storm_df = storm_df.sort_values('ISO_TIME')
    lats = storm_df['LAT'].astype(float).values
    lons = storm_df['LON'].astype(float).values
    times = pd.to_datetime(storm_df['ISO_TIME']).values
    if 'USA_WIND' in storm_df.columns:
        winds = pd.to_numeric(storm_df['USA_WIND'], errors='coerce').values
    else:
        winds = np.full(len(lats), np.nan)
    storm_name = storm_df['NAME'].iloc[0]
    season = storm_df['SEASON'].iloc[0]
    
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)
    lat_padding = max((max_lat - min_lat)*0.3, 5)
    lon_padding = max((max_lon - min_lon)*0.3, 5)
    
    fig = plt.figure(figsize=(12,6), dpi=100)
    ax = plt.axes([0.05, 0.05, 0.60, 0.85],
                  projection=ccrs.PlateCarree(central_longitude=180))
    ax.stock_img()
    ax.set_extent([min_lon - lon_padding, max_lon + lon_padding, min_lat - lat_padding, max_lat + lat_padding],
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"{year} {storm_name} - {season}", fontsize=14)
    
    line, = ax.plot([], [], transform=ccrs.PlateCarree(), color='blue', linewidth=2)
    point, = ax.plot([], [], 'o', markersize=8, transform=ccrs.PlateCarree())
    date_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))
    storm_info_text = fig.text(0.70, 0.60, '', fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    from matplotlib.lines import Line2D
    standard_dict = atlantic_standard if standard=='atlantic' else taiwan_standard
    legend_elements = [Line2D([0],[0], marker='o', color='w', label=cat,
                              markerfacecolor=details['hex'], markersize=8)
                       for cat, details in standard_dict.items()]
    ax.legend(handles=legend_elements, title="Storm Categories",
              loc='upper right', fontsize=9)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        date_text.set_text('')
        storm_info_text.set_text('')
        return line, point, date_text, storm_info_text

    def update(frame):
        line.set_data(lons[:frame+1], lats[:frame+1])
        point.set_data([lons[frame]], [lats[frame]])
        wind_speed = winds[frame] if frame < len(winds) else np.nan
        category, color = categorize_typhoon_by_standard(wind_speed, standard)
        point.set_color(color)
        dt_str = pd.to_datetime(times[frame]).strftime('%Y-%m-%d %H:%M')
        date_text.set_text(dt_str)
        info_str = (f"Name: {storm_name}\n"
                    f"Date: {dt_str}\n"
                    f"Wind: {wind_speed:.1f} kt\n"
                    f"Category: {category}")
        storm_info_text.set_text(info_str)
        return line, point, date_text, storm_info_text

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(times),
                                  interval=200, blit=True, repeat=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    writer = animation.FFMpegWriter(fps=5, bitrate=1800)
    ani.save(temp_file.name, writer=writer)
    plt.close(fig)
    return temp_file.name

def simplified_track_video(year, basin, typhoon, standard):
    if not typhoon:
        return None
    storm_id = typhoon.split('(')[-1].strip(')')
    return generate_track_video_from_csv(year, storm_id, standard)

# ------------- Typhoon Options Update Functions -------------
basin_to_prefix = {
    "All Basins": "all",
    "NA - North Atlantic": "NA",
    "EP - Eastern North Pacific": "EP",
    "WP - Western North Pacific": "WP"
}

def update_typhoon_options(year, basin):
    try:
        if basin == "All Basins":
            summaries = []
            for data in ibtracs.values():
                if data is not None:
                    season_data = data.get_season(int(year))
                    if season_data.summary().empty:
                        continue
                    summaries.append(season_data.summary())
            if len(summaries) == 0:
                logging.error("No storms found for given year and basin.")
                return gr.update(choices=[], value=None)
            combined_summary = pd.concat(summaries, ignore_index=True)
        else:
            prefix = basin_to_prefix.get(basin)
            ds = ibtracs.get(prefix)
            if ds is None:
                logging.error(f"Dataset not found for basin {basin}")
                return gr.update(choices=[], value=None)
            season_data = ds.get_season(int(year))
            if season_data.summary().empty:
                logging.error("No storms found for given year and basin.")
                return gr.update(choices=[], value=None)
            combined_summary = season_data.summary()
        options = []
        for i in range(len(combined_summary)):
            try:
                name = combined_summary['name'][i] if pd.notnull(combined_summary['name'][i]) else "Unnamed"
                storm_id = combined_summary['id'][i]
                options.append(f"{name} ({storm_id})")
            except Exception:
                continue
        return gr.update(choices=options, value=options[0] if options else None)
    except Exception as e:
        logging.error(f"Error in update_typhoon_options: {e}")
        return gr.update(choices=[], value=None)

def update_typhoon_options_anim(year, basin):
    try:
        data = typhoon_data.copy()
        data['Year'] = data['ISO_TIME'].dt.year
        season_data = data[data['Year'] == int(year)]
        if season_data.empty:
            logging.error(f"No storms found for year {year} in animation update.")
            return gr.update(choices=[], value=None)
        summary = season_data.groupby('SID').first().reset_index()
        options = []
        for idx, row in summary.iterrows():
            name = row['NAME'] if pd.notnull(row['NAME']) else "Unnamed"
            options.append(f"{name} ({row['SID']})")
        return gr.update(choices=options, value=options[0] if options else None)
    except Exception as e:
        logging.error(f"Error in update_typhoon_options_anim: {e}")
        return gr.update(choices=[], value=None)

# ------------- Gradio Interface -------------
with gr.Blocks(title="Typhoon Analysis Dashboard") as demo:
    gr.Markdown("# Typhoon Analysis Dashboard")
    
    with gr.Tab("Overview"):
        gr.Markdown("""
        ## Welcome to the Typhoon Analysis Dashboard

        This dashboard allows you to analyze typhoon data in relation to ENSO phases.

        ### Features:
        - **Track Visualization**: View typhoon tracks by time period and ENSO phase.
        - **Wind Analysis**: Examine wind speed vs ONI relationships.
        - **Pressure Analysis**: Analyze pressure vs ONI relationships.
        - **Longitude Analysis**: Study typhoon generation longitude vs ONI.
        - **Path Animation**: View animated storm tracks on a free stock world map (centered at 180°) with a dynamic sidebar and persistent legend.
        - **TSNE Cluster**: Perform t-SNE clustering on WP storm routes using raw merged typhoon+ONI data.
          For each cluster, a mean route is computed and, importantly, mean wind and MSLP curves (plotted versus normalized route index)
          are computed from start to end.
        """)
    
    with gr.Tab("Track Visualization"):
        with gr.Row():
            start_year = gr.Number(label="Start Year", value=2000, minimum=1900, maximum=2024, step=1)
            start_month = gr.Dropdown(label="Start Month", choices=list(range(1,13)), value=1)
            end_year = gr.Number(label="End Year", value=2024, minimum=1900, maximum=2024, step=1)
            end_month = gr.Dropdown(label="End Month", choices=list(range(1,13)), value=6)
            enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all','El Nino','La Nina','Neutral'], value='all')
            typhoon_search = gr.Textbox(label="Typhoon Search")
        analyze_btn = gr.Button("Generate Tracks")
        tracks_plot = gr.Plot(label="Typhoon Tracks", elem_id="tracks_plot")
        typhoon_count = gr.Textbox(label="Number of Typhoons Displayed")
        analyze_btn.click(fn=get_full_tracks,
                          inputs=[start_year, start_month, end_year, end_month, enso_phase, typhoon_search],
                          outputs=[tracks_plot, typhoon_count])
    
    with gr.Tab("Wind Analysis"):
        with gr.Row():
            wind_start_year = gr.Number(label="Start Year", value=2000, minimum=1900, maximum=2024, step=1)
            wind_start_month = gr.Dropdown(label="Start Month", choices=list(range(1,13)), value=1)
            wind_end_year = gr.Number(label="End Year", value=2024, minimum=1900, maximum=2024, step=1)
            wind_end_month = gr.Dropdown(label="End Month", choices=list(range(1,13)), value=6)
            wind_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all','El Nino','La Nina','Neutral'], value='all')
            wind_typhoon_search = gr.Textbox(label="Typhoon Search")
        wind_analyze_btn = gr.Button("Generate Wind Analysis")
        wind_scatter = gr.Plot(label="Wind Speed vs ONI")
        wind_regression_results = gr.Textbox(label="Wind Regression Results")
        wind_analyze_btn.click(fn=get_wind_analysis,
                               inputs=[wind_start_year, wind_start_month, wind_end_year, wind_end_month, wind_enso_phase, wind_typhoon_search],
                               outputs=[wind_scatter, wind_regression_results])
    
    with gr.Tab("Pressure Analysis"):
        with gr.Row():
            pressure_start_year = gr.Number(label="Start Year", value=2000, minimum=1900, maximum=2024, step=1)
            pressure_start_month = gr.Dropdown(label="Start Month", choices=list(range(1,13)), value=1)
            pressure_end_year = gr.Number(label="End Year", value=2024, minimum=1900, maximum=2024, step=1)
            pressure_end_month = gr.Dropdown(label="End Month", choices=list(range(1,13)), value=6)
            pressure_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all','El Nino','La Nina','Neutral'], value='all')
            pressure_typhoon_search = gr.Textbox(label="Typhoon Search")
        pressure_analyze_btn = gr.Button("Generate Pressure Analysis")
        pressure_scatter = gr.Plot(label="Pressure vs ONI")
        pressure_regression_results = gr.Textbox(label="Pressure Regression Results")
        pressure_analyze_btn.click(fn=get_pressure_analysis,
                                   inputs=[pressure_start_year, pressure_start_month, pressure_end_year, pressure_end_month, pressure_enso_phase, pressure_typhoon_search],
                                   outputs=[pressure_scatter, pressure_regression_results])
    
    with gr.Tab("Longitude Analysis"):
        with gr.Row():
            lon_start_year = gr.Number(label="Start Year", value=2000, minimum=1900, maximum=2024, step=1)
            lon_start_month = gr.Dropdown(label="Start Month", choices=list(range(1,13)), value=1)
            lon_end_year = gr.Number(label="End Year", value=2000, minimum=1900, maximum=2024, step=1)
            lon_end_month = gr.Dropdown(label="End Month", choices=list(range(1,13)), value=6)
            lon_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all','El Nino','La Nina','Neutral'], value='all')
            lon_typhoon_search = gr.Textbox(label="Typhoon Search (Optional)")
        lon_analyze_btn = gr.Button("Generate Longitude Analysis")
        regression_plot = gr.Plot(label="Longitude vs ONI")
        slopes_text = gr.Textbox(label="Regression Slopes")
        lon_regression_results = gr.Textbox(label="Longitude Regression Results")
        lon_analyze_btn.click(fn=get_longitude_analysis,
                              inputs=[lon_start_year, lon_start_month, lon_end_year, lon_end_month, lon_enso_phase, lon_typhoon_search],
                              outputs=[regression_plot, slopes_text, lon_regression_results])
    
    with gr.Tab("Tropical Cyclone Path Animation"):
        with gr.Row():
            year_dropdown = gr.Dropdown(label="Year", choices=[str(y) for y in range(1950,2025)], value="2000")
            basin_dropdown = gr.Dropdown(label="Basin", choices=["NA - North Atlantic","EP - Eastern North Pacific","WP - Western North Pacific","All Basins"], value="NA - North Atlantic")
        with gr.Row():
            typhoon_dropdown = gr.Dropdown(label="Tropical Cyclone")
            standard_dropdown = gr.Dropdown(label="Classification Standard", choices=['atlantic','taiwan'], value='atlantic')
        animate_btn = gr.Button("Generate Animation")
        path_video = gr.Video(label="Tropical Cyclone Path Animation", format="mp4", interactive=False, elem_id="path_video")
        animation_info = gr.Markdown("""
        ### Animation Instructions
        1. Select a year and basin (data is from your processed CSV).
        2. Choose a tropical cyclone from the populated list.
        3. Select a classification standard (Atlantic or Taiwan).
        4. Click "Generate Animation".
        5. The animation displays the storm track on a free stock world map (centered at 180°) with a dynamic sidebar and persistent legend.
        """)
        year_dropdown.change(fn=update_typhoon_options_anim, inputs=[year_dropdown, basin_dropdown], outputs=typhoon_dropdown)
        basin_dropdown.change(fn=update_typhoon_options_anim, inputs=[year_dropdown, basin_dropdown], outputs=typhoon_dropdown)
        animate_btn.click(fn=simplified_track_video,
                          inputs=[year_dropdown, basin_dropdown, typhoon_dropdown, standard_dropdown],
                          outputs=path_video)
    
    with gr.Tab("TSNE Cluster"):
        with gr.Row():
            tsne_start_year = gr.Number(label="Start Year", value=2000, minimum=1900, maximum=2024, step=1)
            tsne_start_month = gr.Dropdown(label="Start Month", choices=list(range(1,13)), value=1)
            tsne_end_year = gr.Number(label="End Year", value=2024, minimum=1900, maximum=2024, step=1)
            tsne_end_month = gr.Dropdown(label="End Month", choices=list(range(1,13)), value=12)
            tsne_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all','El Nino','La Nina','Neutral'], value='all')
            tsne_season = gr.Dropdown(label="Season", choices=['all','summer','winter'], value='all')
        tsne_analyze_btn = gr.Button("Analyze")
        tsne_plot = gr.Plot(label="t-SNE Clusters")
        routes_plot = gr.Plot(label="Typhoon Routes with Mean Routes")
        stats_plot = gr.Plot(label="Cluster Statistics")
        cluster_info = gr.Textbox(label="Cluster Information", lines=10)
        tsne_analyze_btn.click(fn=update_route_clusters,
                               inputs=[tsne_start_year, tsne_start_month, tsne_end_year, tsne_end_month, tsne_enso_phase, tsne_season],
                               outputs=[tsne_plot, routes_plot, stats_plot, cluster_info])

demo.launch(share=True)
