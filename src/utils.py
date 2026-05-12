import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_marketing_spend_trend(df):
    """
    Plot marketing spend over time with trend line.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['Date'], df['Marketing_Spend'], marker='o', linewidth=2, markersize=4, label='Marketing Spend')
    
    # Add linear trend
    z = np.polyfit(range(len(df)), df['Marketing_Spend'], 1)
    p = np.poly1d(z)
    ax.plot(df['Date'], p(range(len(df))), 'r--', alpha=0.8, linewidth=2, label='Trend')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Marketing Spend ($)', fontsize=12)
    ax.set_title('Marketing Spend Over Time (36 Months)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_actual_vs_predicted(df_train, y_actual, y_pred):
    """
    Plot actual vs predicted sales (scatter + perfect prediction line).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_actual, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Sales ($)', fontsize=12)
    ax.set_ylabel('Predicted Sales ($)', fontsize=12)
    ax.set_title('Actual vs Predicted Sales (Training Data)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_residuals(y_actual, y_pred):
    """
    Plot residuals (errors) over time.
    """
    residuals = y_actual - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals over time
    ax1.scatter(range(len(residuals)), residuals, alpha=0.6, s=50)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Residual ($)', fontsize=12)
    ax1.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residual ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_forecast(df_train, forecast_df, y_pred_train, y_pred_forecast, ci_lower, ci_upper):
    """
    Plot historical data, trained predictions, and forecast with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historical actual sales
    ax.plot(df_train['Date'], df_train['Actual_Sales'], 'o-', linewidth=2, markersize=5, label='Historical Sales', color='blue')
    
    # Model fit on training data
    ax.plot(df_train['Date'], y_pred_train, '--', linewidth=2, label='Model Fit', color='blue', alpha=0.6)
    
    # Forecast
    all_dates = pd.concat([df_train['Date'], forecast_df['Date']], ignore_index=True)
    all_pred = np.concatenate([y_pred_train, y_pred_forecast])
    
    ax.plot(forecast_df['Date'], y_pred_forecast, 'o-', linewidth=2, markersize=6, label='Forecast', color='green')
    
    # Confidence interval
    ax.fill_between(forecast_df['Date'], ci_lower, ci_upper, alpha=0.2, color='green', label='95% Confidence Interval')
    ax.plot(forecast_df['Date'], ci_lower, '--', linewidth=1, color='green', alpha=0.5)
    ax.plot(forecast_df['Date'], ci_upper, '--', linewidth=1, color='green', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.set_title('Sales Forecast: Historical Data + 3-Month Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def format_currency(value):
    """
    Format a number as currency.
    """
    return f"${value:,.2f}"
