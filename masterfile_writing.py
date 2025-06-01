"""
Created on Mon Apr 21 15:31:10 2025
Title: Omni-Channel Retailer Sales Forecasting & Consumer Trends
Author: @dsarrge
"""
#%%
# === Load Packages ===
import pandas as pd
import numpy as np
from datetime import timedelta

#%%
# === Load File Paths ===
file_paths = {
    'William Sonoma':   r"wsi_all.xlsx",
    'Sur La Table':     r"slt_all.xlsx",
    'Macys':            r"mc_all.xlsx",
    'Crate&Barrel':     r"cb_all.xlsx",
}

# === Clean Load Function ===
def load_and_clean(filepath, customer):
    df = pd.read_excel(filepath, engine='openpyxl')
    df = df.dropna(subset=['GEA SKU'])

    # Set Week End for WSI
    if customer == 'Macys':
        df['Week End'] = pd.to_datetime(df['Week Start']) + pd.Timedelta(days=7)
    else:
        df['Week End'] = pd.to_datetime(df['Week End'])

    # Sales Logic for WSI
    if customer == 'William Sonoma':
        df['WTD Net Sales U'] = (
            pd.to_numeric(df.get('RTL Net Sales U', 0), errors='coerce').fillna(0) +
            pd.to_numeric(df.get('DTC Net Sales U', 0), errors='coerce').fillna(0)
        )
    else:
        df['WTD Net Sales U'] = pd.to_numeric(df.get('WTD Net Sales U'), errors='coerce').fillna(0)

    df.loc[df['WTD Net Sales U'] < 0, 'WTD Net Sales U'] = 0

    # Final Clean Columns
    df_clean = df[['GEA SKU', 'Week End', 'WTD Net Sales U']].copy()
    df_clean['Customer'] = customer

    # Time Features
    df_clean['FW'] = df_clean['Week End'].dt.isocalendar().week
    df_clean['Month'] = df_clean['Week End'].dt.month
    df_clean['Quarter'] = df_clean['Week End'].dt.quarter
    df_clean['Year'] = df_clean['Week End'].dt.year
    
    # Data Types
    df_clean['GEA SKU'] = df_clean['GEA SKU'].astype(str)
    df_clean['WTD Net Sales U'] = df_clean['WTD Net Sales U'].astype(int)
    df_clean['Customer'] = df_clean['Customer'].astype(str)

    return df_clean

# === Load All Data ===
df_all = pd.concat(
    [load_and_clean(path, name) for name, path in file_paths.items()],
    ignore_index=True
)

# Check for NA values
df_all = df_all.dropna (subset=['GEA SKU'])
na_counts = df_all.isna().sum()
print("NA counts per column:")
print(na_counts)

#%%
# Drop empty cells
df_all = df_all[df_all['GEA SKU'].str.strip() != '']

# === Final Export ===
df_all.to_excel("cleaned_selltrhu.xlsx", index=False)
