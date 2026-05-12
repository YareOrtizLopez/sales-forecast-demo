import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for linear regression model.
    
    Features:
    - Month_Index: 1-36 (captures linear time trend)
    - Marketing_Spend: Raw spend
    - Is_December: Binary flag for December months (12, 24, 36, ...)
    - Spend_MA3: 3-month moving average (smooths noise)
    - Spend_Lag1: Previous month's spend (captures momentum)
    
    Args:
        df: DataFrame with Date, Month_Index, Marketing_Spend
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Extract month from date
    df['Month'] = df['Date'].dt.month
    
    # Is_December flag
    df['Is_December'] = (df['Month'] == 12).astype(int)
    
    # 3-month moving average of spend (handles noise)
    df['Spend_MA3'] = df['Marketing_Spend'].rolling(window=3, center=True).mean()
    # Forward-fill and back-fill for missing values (pandas 3.0+ syntax)
    df['Spend_MA3'] = df['Spend_MA3'].bfill().ffill()
    
    # Lagged spend (previous month)
    df['Spend_Lag1'] = df['Marketing_Spend'].shift(1)
    df['Spend_Lag1'] = df['Spend_Lag1'].fillna(df['Marketing_Spend'].iloc[0])
    
    return df

def prepare_training_data(df: pd.DataFrame):
    """
    Prepare X (features) and y (target) for model training.
    
    Args:
        df: DataFrame with engineered features and 'Actual_Sales' column
    
    Returns:
        X (features), y (target sales), feature names list
    """
    feature_cols = ['Month_Index', 'Marketing_Spend', 'Is_December', 'Spend_MA3']
    
    X = df[feature_cols].values
    y = df['Actual_Sales'].values
    
    return X, y, feature_cols
