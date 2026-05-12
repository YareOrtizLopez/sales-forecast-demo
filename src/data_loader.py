import pandas as pd
import numpy as np
from pathlib import Path

def load_marketing_spend(filepath: str) -> pd.DataFrame:
    """
    Load marketing spend CSV and perform basic validation.
    
    Args:
        filepath: Path to marketing_spend.csv
    
    Returns:
        DataFrame with Date (datetime), Month_Index, Marketing_Spend
    """
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = ['Date', 'Month_Index', 'Marketing_Spend']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Basic sanity checks
    if (df['Marketing_Spend'] <= 0).any():
        raise ValueError("Marketing spend must be positive")
    
    if df['Month_Index'].max() != len(df):
        print("Warning: Month_Index may not be sequential")
    
    return df

def get_next_forecast_months(df: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
    """
    Generate forecast period (next n months after the last training data point).
    
    Args:
        df: Training data
        n_months: Number of months to forecast (default 3)
    
    Returns:
        DataFrame with forecast period dates and Month_Index
    """
    last_date = df['Date'].max()
    last_month_idx = df['Month_Index'].max()
    last_spend = df['Marketing_Spend'].iloc[-1]
    
    # Assume spend grows at recent average rate
    recent_growth = (df['Marketing_Spend'].iloc[-1] - df['Marketing_Spend'].iloc[-6]) / 6
    
    forecast_dates = []
    forecast_months = []
    forecast_spends = []
    
    for i in range(1, n_months + 1):
        next_date = last_date + pd.DateOffset(months=i)
        next_month_idx = last_month_idx + i
        next_spend = last_spend + (recent_growth * i)
        
        forecast_dates.append(next_date)
        forecast_months.append(next_month_idx)
        forecast_spends.append(next_spend)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Month_Index': forecast_months,
        'Marketing_Spend': forecast_spends
    })
    
    return forecast_df
