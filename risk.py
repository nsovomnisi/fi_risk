#%%

# Import packages
import pandas as pd
import os
import re
from skimpy import clean_columns
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%%

# Navigate to data folder
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'data')
files = os.listdir(data_directory)
if not files:
    raise FileNotFoundError("No files found in the data directory.")
file = files[0]
file_path = os.path.join(data_directory, file)

# Get list of sheets in the excel file
list_sheets = pd.ExcelFile(file_path).sheet_names

# Read excel file
df_data = pd.read_excel(os.path.join(data_directory, file), sheet_name=None)

# Initialize empty dataframe
df_holdings = pd.DataFrame()

# Loop through the first two sheets, add a datestamp column to each dataframe, and concatenate them
for i in range(len(list_sheets) - 1):
    df = df_data[list_sheets[i]]
    match = re.search(r'\((\d{1,2} \w+ \d{4})\)', list_sheets[i])
    if match:
        date_str = match.group(1)
        df = df.assign(datestamp=pd.to_datetime(date_str, format='%d %B %Y'))
    df_holdings = pd.concat([df_holdings, df], ignore_index=True)

# Assign the last sheet to df_risk_metrics
df_risk_metrics = df_data[list_sheets[-1]]

# Clean column names
df_holdings = clean_columns(df_holdings)
df_risk_metrics = clean_columns(df_risk_metrics)

# Remove rows where asset_id is null
df_holdings.dropna(subset=['asset_id'], inplace=True)
df_holdings['weight_%'] = df_holdings['weight_%']*100
df_holdings['bmk_weight_%'] = df_holdings['bmk_weight_%']*100
df_holdings['active_weight_%'] = df_holdings['active_weight_%']*100

# Create separate dataframes for each period
df_june = df_holdings[df_holdings['datestamp'] == pd.Timestamp('2024-06-30')]
df_march = df_holdings[df_holdings['datestamp'] == pd.Timestamp('2024-03-31')]

# Select the top 20 holdings for each period
df_top_holdings = df_holdings.groupby('datestamp').apply(lambda x: x.nlargest(20, 'weight_%')).reset_index(drop=True)

#%%

# ENHANCED ANALYSIS - PORTFOLIO COMPOSITION SHIFTS

# 1. Calculate portfolio statistics summary for each period
def portfolio_summary(df):
    summary = {
        'Total Assets': len(df),
        'Max Weight (%)': df['weight_%'].max(),
        'Top 5 Concentration (%)': df.nlargest(5, 'weight_%')['weight_%'].sum(),
        'Top 10 Concentration (%)': df.nlargest(10, 'weight_%')['weight_%'].sum(),
        'Avg. Active Weight (%)': df['active_weight_%'].mean(),
        'Avg. Active Total Risk': df['active_total_risk'].mean(),
        'Max Active Total Risk': df['active_total_risk'].max(),
        'Avg. Active Effective Duration': df['active_effective_duration_mac'].mean(),
        'Avg. Active Spread Duration': df['active_spread_duration'].mean()
    }
    return pd.Series(summary)

march_summary = portfolio_summary(df_march)
june_summary = portfolio_summary(df_june)
portfolio_evolution = pd.DataFrame({'March 2024': march_summary, 'June 2024': june_summary})
portfolio_evolution['Change'] = portfolio_evolution['June 2024'] - portfolio_evolution['March 2024']
portfolio_evolution['% Change'] = (portfolio_evolution['Change'] / portfolio_evolution['March 2024'] * 100).round(2)

# Print the summary statistics
print("\nPORTFOLIO EVOLUTION SUMMARY:")
print(portfolio_evolution)

# 2. Identify new additions and removals between periods
march_assets = set(df_march['asset_id'])
june_assets = set(df_june['asset_id'])

new_assets = june_assets - march_assets
removed_assets = march_assets - june_assets

# Get details of new additions
if new_assets:
    new_assets_df = df_june[df_june['asset_id'].isin(new_assets)]
    print(f"\nNEW ASSETS ADDED ({len(new_assets)}):")
    print(new_assets_df[['asset_id', 'asset_name', 'weight_%', 'active_weight_%']].sort_values('weight_%', ascending=False).head(10))

# Get details of removed assets
if removed_assets:
    removed_assets_df = df_march[df_march['asset_id'].isin(removed_assets)]
    print(f"\nASSETS REMOVED ({len(removed_assets)}):")
    print(removed_assets_df[['asset_id', 'asset_name', 'weight_%', 'active_weight_%']].sort_values('weight_%', ascending=False).head(10))

# 3. Identify biggest weight changes in common assets
common_assets = march_assets.intersection(june_assets)
if common_assets:
    march_common = df_march[df_march['asset_id'].isin(common_assets)].set_index('asset_id')
    june_common = df_june[df_june['asset_id'].isin(common_assets)].set_index('asset_id')
    
    # Calculate weight changes
    weight_changes = pd.DataFrame({
        'March Weight (%)': march_common['weight_%'].round(2),
        'June Weight (%)': june_common['weight_%'].round(2)
    })
    weight_changes['Absolute Change (%)'] = (weight_changes['June Weight (%)'] - weight_changes['March Weight (%)']).abs()
    weight_changes['Relative Change (%)'] = ((weight_changes['June Weight (%)'] - weight_changes['March Weight (%)']) / weight_changes['March Weight (%)'] * 100).round(2)
    weight_changes = weight_changes.join(march_common[['asset_name']])
    
    print("\nBIGGEST WEIGHT CHANGES IN EXISTING HOLDINGS:")
    print(weight_changes.sort_values('Absolute Change (%)', ascending=False).head(10))

#%%

# TABLE VISUALIZATIONS

# Portfolio Evolution Summary - Adjusting for rounding percentages and column widths
portfolio_evolution['% Change'] = portfolio_evolution['% Change'].round(2)  # Round to 2 decimals
portfolio_evolution['Change'] = portfolio_evolution['Change'].round(2)  # Round to 2 decimals

portfolio_evolution['Metric'] = portfolio_evolution.index
portfolio_evolution = portfolio_evolution[['Metric', 'March 2024', 'June 2024', 'Change', '% Change']]

# Create the table with custom column width and prevent overlap
fig = ff.create_table(portfolio_evolution)
fig.update_layout(
    title_text="Portfolio Evolution Summary (March 2024 vs June 2024)",
    margin=dict(t=50, b=50),
)

fig.show()

#%%

# New Assets Added - Adjust rounding and prevent overlap
if new_assets:
    new_assets_table = new_assets_df[['asset_id', 'asset_name', 'weight_%', 'active_weight_%']].sort_values('weight_%', ascending=False).head(10)
    new_assets_table['weight_%'] = new_assets_table['weight_%'].round(2)
    new_assets_table['active_weight_%'] = new_assets_table['active_weight_%'].round(2)

    # Create the table with custom column width and prevent overlap
    fig = ff.create_table(new_assets_table)
    fig.update_layout(
        title_text="New Assets Added (March vs June 2024)",
        margin=dict(t=50, b=50),
    )
    fig.show()

#%%

# Removed Assets - Adjust rounding and prevent overlap
if removed_assets:
    removed_assets_table = removed_assets_df[['asset_id', 'asset_name', 'weight_%', 'active_weight_%']].sort_values('weight_%', ascending=False).head(10)
    removed_assets_table['weight_%'] = removed_assets_table['weight_%'].round(2)
    removed_assets_table['active_weight_%'] = removed_assets_table['active_weight_%'].round(2)

    # Create the table with custom column width and prevent overlap
    fig = ff.create_table(removed_assets_table)
    fig.update_layout(
        title_text="Assets Removed (March vs June 2024)",
        margin=dict(t=50, b=50),
    )
    fig.show()

#%%

# Biggest Weight Changes in Existing Holdings - Adjust rounding and prevent overlap
top_weight_changes_table = weight_changes.sort_values('Absolute Change (%)', ascending=False).head(10)
top_weight_changes_table['Absolute Change (%)'] = top_weight_changes_table['Absolute Change (%)'].round(2)
top_weight_changes_table['Relative Change (%)'] = top_weight_changes_table['Relative Change (%)'].round(2)

# Create the table with custom column width and prevent overlap
fig = ff.create_table(top_weight_changes_table)
fig.update_layout(
    title_text="Biggest Weight Changes in Existing Holdings (March vs June 2024)",
    margin=dict(t=50, b=50),
)
fig.show()


#%%
# ENHANCED VISUALIZATIONS

# 1. Top Holdings Comparison - Enhanced with hover info
fig = px.bar(df_top_holdings, 
             x='asset_name', 
             y='weight_%', 
             color='datestamp', 
             barmode='group', 
             title='Top 20 Holdings - March vs June 2024', 
             labels={'weight_%': 'Portfolio Weight (%)'},
             hover_data=['asset_id', 'active_weight_%', 'active_total_risk'],
             text='weight_%')

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(xaxis_tickangle=-45, height=800)
fig.show()

#%%

# 2. Portfolio Composition Comparison (Pie charts in subplots)
fig = make_subplots(rows=2, cols=1,  # Change to 2 rows and 1 column
                    specs=[[{'type':'domain'}], [{'type':'domain'}]],  # Pie charts in both rows
                    subplot_titles=['March 2024 - Top 10 Holdings', 'June 2024 - Top 10 Holdings'],
                    vertical_spacing=0.1)

# Add first pie chart for March 2024
fig.add_trace(go.Pie(labels=df_march.nlargest(10, 'weight_%')['asset_name'], 
                    values=df_march.nlargest(10, 'weight_%')['weight_%'],
                    textinfo='label+percent',
                    hole=.3,
                    pull=[0.1 if i == 0 else 0 for i in range(10)]), 
             row=1, col=1)  # First pie chart in first row, first column

# Add second pie chart for June 2024
fig.add_trace(go.Pie(labels=df_june.nlargest(10, 'weight_%')['asset_name'], 
                    values=df_june.nlargest(10, 'weight_%')['weight_%'],
                    textinfo='label+percent',
                    hole=.3,
                    pull=[0.1 if i == 0 else 0 for i in range(10)]), 
             row=2, col=1)  # Second pie chart in second row, first column

# Update layout
fig.update_layout(title_text="Portfolio Concentration - Top 10 Holdings", height=800)

# Show the figure
fig.show()


#%%

# 3. Risk Metrics Time Series - Combined plot with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=df_risk_metrics['reference_date'], y=df_risk_metrics['tracking_error_ex_ante'], 
               mode='lines+markers', name='Tracking Error',
               line=dict(color='blue', width=2)),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df_risk_metrics['reference_date'], y=df_risk_metrics['beta_p'], 
               mode='lines+markers', name='Beta',
               line=dict(color='red', width=2, dash='dot')),
    secondary_y=True,
)

fig.update_layout(
    title_text="Portfolio Risk Metrics Over Time",
    height=500,
    hovermode="x unified"
)

fig.update_layout(
    xaxis=dict(
        type='date',  # This ensures the x-axis is using dates, not numeric values
        range=[df_risk_metrics['reference_date'].min(), df_risk_metrics['reference_date'].max()]
    )
)


fig.update_xaxes(title_text="Date", type='date' )
fig.update_yaxes(title_text="Tracking Error", secondary_y=False)
fig.update_yaxes(title_text="Beta", secondary_y=True)
fig.show()

#%%

# 4. Active Risk Decomposition - Bubble Chart (size represents contribution to risk)
fig = px.scatter(df_june.nlargest(20, '%_cr_to_active_total_risk'), 
                x='active_effective_duration_mac', y='active_spread_duration',
                size='%_cr_to_active_total_risk', color='weight_%',
                hover_name='asset_name', hover_data=['asset_id', 'weight_%', 'active_total_risk'],
                size_max=60, color_continuous_scale=px.colors.sequential.Viridis,
                title="Active Risk Decomposition - June 2024 (Top 20 Risk Contributors)")

fig.update_layout(height=600)
fig.show()

#%%

# 5. Benchmark vs Portfolio Weight Comparison - More readable with limited assets
# Only show assets with significant benchmark or portfolio weights
significant_assets = df_june[
    (df_june['weight_%'] > 0.05) | (df_june['bmk_weight_%'] > 1)
].sort_values('active_weight_%', key=abs, ascending=False).head(15)

fig = px.bar(significant_assets, 
             x='asset_name', 
             y=['weight_%', 'bmk_weight_%'], 
             barmode='group',
             title='Portfolio vs Benchmark Weights - Key Positions',
             labels={'value': 'Weight (%)', 'variable': 'Weight Type'},
             color_discrete_map={'weight_%': 'blue', 'bmk_weight_%': 'lightgrey'})

fig.add_trace(
    go.Scatter(x=significant_assets['asset_name'], y=significant_assets['active_weight_%'],
              mode='lines+markers', name='Active Weight',
              line=dict(color='red', width=2))
)

fig.update_layout(xaxis_tickangle=-45, height=800)
fig.show()

#%%

# 6. Duration Exposures Over Time
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_risk_metrics['reference_date'], y=df_risk_metrics['spread_duration_active'], 
                        mode='lines+markers', name='Active Spread Duration',
                        line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=df_risk_metrics['reference_date'], y=df_risk_metrics['credit_spread_dur_active'], 
                        mode='lines+markers', name='Credit Spread Duration',
                        line=dict(color='red', width=2)))

# Add a reference line at zero
fig.add_hline(y=0, line_dash="dash", line_color="grey")

fig.update_layout(
    title='Duration Exposures Over Time',
    xaxis_title='Date',
    yaxis_title='Duration (Years)',
    height=500,
    hovermode="x unified"
)
fig.show()

#%%

# 7. Risk vs Return Analysis (if price data is available)
if all(col in df_june.columns for col in ['dirty_price', 'price']):
    # Calculate return metrics if pricing data is available
    common_assets = march_assets.intersection(june_assets)
    if common_assets:
        march_common = df_march[df_march['asset_id'].isin(common_assets)].set_index('asset_id')
        june_common = df_june[df_june['asset_id'].isin(common_assets)].set_index('asset_id')
        
        # Calculate price change
        price_analysis = pd.DataFrame({
            'March Price': march_common['dirty_price'],
            'June Price': june_common['dirty_price'],
            'Risk (June)': june_common['active_total_risk']
        })
        
        price_analysis['Return (%)'] = ((price_analysis['June Price'] - price_analysis['March Price']) / 
                                    price_analysis['March Price'] * 100).round(2)
        
        price_analysis = price_analysis.join(june_common[['asset_name', 'weight_%']])
        
        # Create Risk vs Return bubble chart
        fig = px.scatter(price_analysis.dropna(), 
                        x='Risk (June)', y='Return (%)',
                        size='weight_%', color='Return (%)',
                        hover_name='asset_name',
                        size_max=50, 
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        title="Risk vs Return Analysis - March to June 2024")
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        
        # Add quadrant labels
        fig.update_layout(
            yaxis=dict(range=[-15, 15]),  # Cap the y-axis at 50%
            annotations=[
                dict(x=price_analysis['Risk (June)'].max()*0.75, y=price_analysis['Return (%)'].max()*0.75, 
                     text="High Risk / High Return", showarrow=False, xanchor="center"),
                dict(x=price_analysis['Risk (June)'].min()*0.75, y=price_analysis['Return (%)'].max()*0.75, 
                     text="Low Risk / High Return", showarrow=False, xanchor="center"),
                dict(x=price_analysis['Risk (June)'].max()*0.75, y=price_analysis['Return (%)'].min()*0.75, 
                     text="High Risk / Low Return", showarrow=False, xanchor="center"),
                dict(x=price_analysis['Risk (June)'].min()*0.75, y=price_analysis['Return (%)'].min()*0.75, 
                     text="Low Risk / Low Return", showarrow=False, xanchor="center")
            ],
            height=600
        )
        fig.show()

#%%