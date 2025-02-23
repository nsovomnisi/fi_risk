#%%

# import packages
import pandas as pd
import numpy as np
import os
import re
import datetime
from sqlalchemy import create_engine
from skimpy import clean_columns
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('qtagg')


#%%

# navigate to data folder
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'data')
files = os.listdir(data_directory)
if not files:
    raise FileNotFoundError("No files found in the data directory.")
file = files[0]
file_path = os.path.join(data_directory, file)

# %%

# get list of sheets in the excel file
list_sheets = pd.ExcelFile(file_path).sheet_names

#%%

# read excel file
df_data = pd.read_excel(os.path.join(data_directory,file),sheet_name=None)

# inititalize empty dataframe
df_holdings = pd.DataFrame()

# loop through the first two sheets, add a datestamp column to each dataframe, and concatenate them
for i in range(len(list_sheets) - 1):
    df = df_data[list_sheets[i]]
    match = re.search(r'\((\d{1,2} \w+ \d{4})\)', list_sheets[i])
    if match:
        date_str = match.group(1)
        df = df.assign(datestamp=pd.to_datetime(date_str, format='%d %B %Y'))
    df_holdings = pd.concat([df_holdings, df], ignore_index=True)

# assign the last sheet to df_risk_metrics
df_risk_metrics = df_data[list_sheets[-1]]

#%%

# clean column names
df_holdings = clean_columns(df_holdings)
df_risk_metrics = clean_columns(df_risk_metrics)

# %%

# remove rows where portfolio_code/asset_id is null
df_holdings.dropna(subset=['asset_id'], inplace=True)
df_risk_metrics.dropna(subset=['portfolio_code'], inplace=True)

#%%

df_june = df_holdings[df_holdings['datestamp'] == pd.Timestamp('2024-06-30')]
df_march = df_holdings[df_holdings['datestamp'] == pd.Timestamp('2024-03-31')]

#%%

plt.figure(figsize=(12, 5))
sns.kdeplot(df_june["weight_%"], label="June 2024", fill=False, alpha=0.5)
sns.kdeplot(df_march["weight_%"], label="March 2024", fill=False, alpha=0.5)
plt.xlabel("Portfolio weight_%")
plt.ylabel("Density")
plt.title("Portfolio Weight Distribution: March vs. June 2024")
plt.legend()
# Save plot to file if needed
plt.savefig("portfolio_weight_distribution.png")
# Show the plot (in interactive environments)
plt.show()

# %%


# Compute weight change
df_march.set_index("asset_id", inplace=True)
df_june.set_index("asset_id", inplace=True)

#%%

# Compute weight change
weight_change = df_june["weight_%"] - df_march["weight_%"]
top_weight_changes = weight_change.abs().nlargest(10)  # Largest absolute changes

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=top_weight_changes.index, y=top_weight_changes.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.xlabel("Asset ID")
plt.ylabel("Weight Change (%)")
plt.title("Top 10 Largest Weight Changes (March vs. June)")
plt.savefig("top_weight_changes.png")
plt.show()

# %%

# Compute active risk change
risk_change = df_june["active_total_risk"] - df_march["active_total_risk"]
top_risk_changes = risk_change.abs().nlargest(10)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=top_risk_changes.index, y=top_risk_changes.values, palette="magma")
plt.xticks(rotation=45)
plt.xlabel("Asset ID")
plt.ylabel("Active Risk Change")
plt.title("Top 10 Largest Active Risk Changes (March vs. June)")
plt.savefig("top_risk_changes.png")
plt.show()

#%%

# Compute duration change
duration_change = df_june["active_effective_duration_mac"] - df_march["active_effective_duration_mac"]
top_duration_changes = duration_change.abs().nlargest(10)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=top_duration_changes.index, y=top_duration_changes.values, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Asset ID")
plt.ylabel("Duration Change")
plt.title("Top 10 Largest Duration Changes (March vs. June)")
plt.savefig("top_duration_changes.png")
plt.show()


#%%


plt.figure(figsize=(10, 5))
sns.lineplot(x=df_risk_metrics["reference_date"], y=df_risk_metrics["tracking_error_ex_ante"], marker="o", color="blue")
plt.xlabel("Date")
plt.ylabel("Tracking Error")
plt.title("Tracking Error Over Time")
plt.xticks(rotation=45)
plt.savefig("tracking_error.png")
plt.show()

#%%

plt.figure(figsize=(10, 5))
sns.lineplot(x=df_risk_metrics["reference_date"], y=df_risk_metrics["credit_spread_dur_active"], marker="o", color="red")
plt.xlabel("Date")
plt.ylabel("Credit Spread Duration")
plt.title("Credit Spread Duration Over Time")
plt.xticks(rotation=45)
plt.savefig("credit_spread_duration.png")
plt.show()

#%%
