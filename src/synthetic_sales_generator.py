import pandas as pd
import numpy as np

def generate_synthetic_sales(
    df: pd.DataFrame,
    base_sales: float = 20000,
    trend_coef: float = 400,
    elasticity: float = 2.5,
    december_premium_pct: float = 0.50,
    noise_std: float = 1500
) -> pd.DataFrame:
    """
    Generate realistic synthetic sales data based on marketing spend.
    
    Formula:
        Sales = Base + (Trend × Month_Index) + (Elasticity × Spend) 
                + (December_Premium × Is_December) + Noise
    
    The December_Premium is a multiplicative 50% boost on top of the elasticity effect.
    This reflects real business dynamics:
    - Holiday demand surge (consumers buy more in Dec)
    - Budget flush (companies spend remaining Q4 budgets)
    - Tax incentives (year-end purchase incentives)
    - Gift-giving season (retail peak)
    
    Args:
        df: DataFrame with Month_Index, Marketing_Spend, Is_December
        base_sales: Baseline monthly sales (no marketing effect)
        trend_coef: Organic growth per month (dollars)
        elasticity: Sales per $1 of marketing spend (dollars)
        december_premium_pct: Multiplicative boost for December (e.g., 0.50 = 50% increase)
        noise_std: Standard deviation of random noise (dollars)
    
    Returns:
        DataFrame with added 'Actual_Sales' column
    """
    df = df.copy()
    
    # Ensure features exist
    if 'Is_December' not in df.columns:
        df['Is_December'] = (df['Date'].dt.month == 12).astype(int)
    
    # Base formula: trend + elasticity
    sales = (
        base_sales +
        (trend_coef * df['Month_Index']) +
        (elasticity * df['Marketing_Spend'])
    )
    
    # December premium: 50% multiplicative boost
    # For December, multiply the elasticity component by 1.5
    december_boost = elasticity * df['Marketing_Spend'] * december_premium_pct * df['Is_December']
    sales += december_boost
    
    # Add realistic noise (normal distribution)
    noise = np.random.normal(0, noise_std, len(df))
    sales += noise
    
    # Ensure sales are positive
    sales = np.maximum(sales, 1000)
    
    df['Actual_Sales'] = sales
    
    return df

def explain_december_premium():
    """
    Return explanation of why December gets a 50% premium.
    """
    return """
    Why December Sales Get a 50% Premium:
    
    1. HOLIDAY DEMAND SURGE
       Consumers and businesses increase purchases before year-end holidays.
       E-commerce, retail, and B2B all see peak seasonal demand.
    
    2. BUDGET FLUSH / USE-IT-OR-LOSE-IT
       Companies have annual budgets that expire Dec 31.
       Finance departments encourage managers to spend remaining budget
       to preserve next year's allocation.
    
    3. TAX & ACCOUNTING INCENTIVES
       Year-end purchases trigger tax deductions and accounting benefits.
       Equipment, software, and inventory purchases accelerate in Q4.
    
    4. GIFT-GIVING SEASON
       Retail peaks for corporate gifting, consumer gifting, and self-purchase.
       Average basket size increases.
    
    MODELING APPROACH:
    Instead of just capturing elasticity (Sales = $2.50 × Spend),
    December gets an ADDITIONAL 50% boost on top of that elasticity.
    
    Example: 
    - Regular month: Spend $10,000 → Sales = Baseline + (2.5 × $10,000) = +$25,000
    - December:     Spend $11,000 → Sales = Baseline + (2.5 × $11,000 × 1.5) = +$41,250
    
    This is the REALISTIC behavior you'll see in actual P&L data.
    """
