import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_marketing_spend, get_next_forecast_months
from feature_engineering import engineer_features, prepare_training_data
from synthetic_sales_generator import generate_synthetic_sales, explain_december_premium
from model import SalesForecaster
from utils import (
    plot_marketing_spend_trend,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_forecast,
    format_currency
)

# Page config
st.set_page_config(
    page_title="Sales Forecast Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Sales Forecast Dashboard")
st.markdown("*An FP&A teaching demo: Forecasting monthly sales using marketing spend with explainable linear regression*")

# ============================================================================
# SIDEBAR: Configuration & Data Generation Parameters
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load data
    csv_path = Path('data/marketing_spend.csv')
    if csv_path.exists():
        df_raw = load_marketing_spend(str(csv_path))
        st.success(f"✅ Loaded {len(df_raw)} months of marketing spend data")
    else:
        st.error(f"❌ Could not find {csv_path}")
        st.stop()
    
    st.subheader("Synthetic Sales Parameters")
    st.markdown("Adjust these to control how sales respond to marketing spend:")
    
    base_sales = st.slider(
        "Base Sales (no marketing effect)",
        min_value=10000,
        max_value=30000,
        value=20000,
        step=1000,
        help="Baseline monthly sales"
    )
    
    trend_coef = st.slider(
        "Monthly Growth Trend",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="Organic growth per month (dollars)"
    )
    
    elasticity = st.slider(
        "Marketing Elasticity",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.5,
        help="Sales increase per $1 of marketing spend"
    )
    
    december_premium = st.slider(
        "December Premium %",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Additional multiplicative boost for December (e.g., 0.50 = 50%)"
    )
    
    noise_std = st.slider(
        "Noise (Realistic Variation)",
        min_value=500,
        max_value=5000,
        value=1500,
        step=250,
        help="Standard deviation of random monthly variation"
    )
    
    st.markdown("---")
    st.subheader("About the December Premium")
    if st.button("📖 Learn why December gets +50%"):
        st.info(explain_december_premium())

# ============================================================================
# MAIN CONTENT: Tabs
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Data Exploration", "🔧 Feature Engineering", "📊 Model", "🔮 Forecast", "📋 Summary"]
)

# ============================================================================
# TAB 1: Data Exploration
# ============================================================================
with tab1:
    st.header("Data Exploration: Marketing Spend")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Time Period", f"{len(df_raw)} months")
    with col2:
        st.metric("Min Spend", format_currency(df_raw['Marketing_Spend'].min()))
    with col3:
        st.metric("Max Spend", format_currency(df_raw['Marketing_Spend'].max()))
    with col4:
        st.metric("Avg Spend", format_currency(df_raw['Marketing_Spend'].mean()))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spend Trend")
        fig = plot_marketing_spend_trend(df_raw)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Spend Statistics")
        st.dataframe(df_raw[['Date', 'Month_Index', 'Marketing_Spend']].describe(), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Raw Data (First 12 months)")
    st.dataframe(df_raw.head(12), use_container_width=True)

# ============================================================================
# TAB 2: Feature Engineering
# ============================================================================
with tab2:
    st.header("Feature Engineering")
    
    st.markdown("""
    We create features from the raw marketing spend data to help the model learn:
    
    1. **Month_Index**: Numeric time trend (1-36) → captures linear growth
    2. **Marketing_Spend**: Raw monthly spend → elasticity driver
    3. **Is_December**: Binary flag (0 or 1) → captures seasonal December boost
    4. **Spend_MA3**: 3-month moving average → smooths noise
    5. **Spend_Lag1**: Previous month's spend → momentum indicator
    """)
    
    # Engineer features
    df_features = engineer_features(df_raw)
    
    st.subheader("Engineered Features (First 12 months)")
    st.dataframe(
        df_features[['Date', 'Month_Index', 'Marketing_Spend', 'Is_December', 'Spend_MA3']].head(12),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("December Flag Examples")
    december_rows = df_features[df_features['Is_December'] == 1][['Date', 'Month', 'Is_December', 'Marketing_Spend']]
    st.dataframe(december_rows, use_container_width=True)
    st.info(f"✅ Found {len(december_rows)} December months in the data")

# ============================================================================
# TAB 3: Model Training & Interpretation
# ============================================================================
with tab3:
    st.header("Linear Regression Model")
    
    # Generate synthetic sales
    df_with_sales = generate_synthetic_sales(
        df_features,
        base_sales=base_sales,
        trend_coef=trend_coef,
        elasticity=elasticity,
        december_premium_pct=december_premium,
        noise_std=noise_std
    )
    
    # Prepare training data
    X, y, feature_names = prepare_training_data(df_with_sales)
    
    # Train model
    forecaster = SalesForecaster()
    forecaster.fit(X, y, feature_names)
    
    # Display model summary
    st.markdown("### Model Equation")
    st.code(forecaster.get_formula(), language="text")
    
    st.markdown("### Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{forecaster.r2:.4f}", help="% of variance explained (higher is better)")
    with col2:
        st.metric("RMSE", format_currency(forecaster.rmse), help="Root mean squared error")
    with col3:
        st.metric("MAE", format_currency(forecaster.mae), help="Mean absolute error")
    
    st.markdown("---")
    st.markdown("### Coefficient Interpretation")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': forecaster.coefficients,
        'Interpretation': [
            'Growth per month (dollars)',
            'Sales per $1 marketing spend',
            'Additional boost for December (dollars)',
            'Smoothed spend elasticity'
        ]
    })
    
    st.dataframe(coef_df, use_container_width=True)
    
    st.markdown(f"**Intercept (Base Sales):** {format_currency(forecaster.intercept)}")
    
    st.markdown("---")
    st.markdown("### Model Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted")
        fig_av_pred = plot_actual_vs_predicted(df_with_sales, y, forecaster.y_pred_train)
        st.pyplot(fig_av_pred)
    
    with col2:
        st.subheader("Residuals Analysis")
        fig_resid = plot_residuals(y, forecaster.y_pred_train)
        st.pyplot(fig_resid)

# ============================================================================
# TAB 4: Forecast
# ============================================================================
with tab4:
    st.header("3-Month Sales Forecast")
    
    # Generate forecast period
    forecast_df = get_next_forecast_months(df_with_sales, n_months=3)
    
    # Engineer features for forecast
    forecast_df = engineer_features(forecast_df)
    
    # Prepare forecast features (same as training)
    X_forecast = forecast_df[feature_names].values
    
    # Make predictions with confidence intervals
    y_pred_forecast, ci_lower, ci_upper = forecaster.predict_with_ci(X_forecast, confidence=0.95)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': forecast_df['Date'],
        'Month': forecast_df['Month_Index'],
        'Marketing_Spend': forecast_df['Marketing_Spend'],
        'Forecast_Sales': y_pred_forecast,
        'CI_Lower_95%': ci_lower,
        'CI_Upper_95%': ci_upper
    })
    
    # Format for display
    results_display = results_df.copy()
    for col in ['Marketing_Spend', 'Forecast_Sales', 'CI_Lower_95%', 'CI_Upper_95%']:
        results_display[col] = results_display[col].apply(format_currency)
    
    st.subheader("Forecast Results")
    st.dataframe(results_display, use_container_width=True)
    
    # Download forecast as CSV
    csv_download = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_download,
        file_name="sales_forecast_jan_mar_2026.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.subheader("Forecast Visualization")
    fig_forecast = plot_forecast(df_with_sales, forecast_df, forecaster.y_pred_train, y_pred_forecast, ci_lower, ci_upper)
    st.pyplot(fig_forecast)
    
    st.markdown("---")
    st.subheader("Forecast Details")
    
    for idx, row in results_df.iterrows():
        month_name = row['Date'].strftime('%B %Y')
        st.markdown(f"**{month_name}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spend", format_currency(row['Marketing_Spend']))
        with col2:
            st.metric("Forecast", format_currency(row['Forecast_Sales']))
        with col3:
            st.metric("Lower CI", format_currency(row['CI_Lower_95%']))
        with col4:
            st.metric("Upper CI", format_currency(row['CI_Upper_95%']))
        st.divider()

# ============================================================================
# TAB 5: Summary & Interpretation
# ============================================================================
with tab5:
    st.header("Executive Summary")
    
    st.markdown("""
    ## Forecast Summary
    
    This dashboard demonstrates a realistic sales forecasting workflow using **explainable linear regression**.
    
    ### Key Findings
    """)
    
    # Summary stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Historical Data**")
        st.markdown(f"- Period: {df_raw['Date'].min().strftime('%B %Y')} to {df_raw['Date'].max().strftime('%B %Y')}")
        st.markdown(f"- Data points: {len(df_raw)}")
        st.markdown(f"- Spend range: {format_currency(df_raw['Marketing_Spend'].min())} – {format_currency(df_raw['Marketing_Spend'].max())}")
    
    with col2:
        st.markdown("**Model Performance**")
        st.markdown(f"- R² Score: {forecaster.r2:.4f}")
        st.markdown(f"- RMSE: {format_currency(forecaster.rmse)}")
        st.markdown(f"- Avg. Error: {format_currency(forecaster.mae)}")
    
    st.markdown("---")
    st.markdown("### Why Linear Regression?")
    st.markdown("""
    ✅ **Explainability:** Each coefficient translates to a business metric (elasticity = sales per $1 spend)  
    ✅ **Simplicity:** Easy to explain to CFOs and non-technical stakeholders  
    ✅ **Data Efficiency:** Works well with only 36 months of data  
    ✅ **Interpretability:** Can identify which drivers matter most  
    ✅ **Stability:** Robust, no overfitting on small datasets  
    """)
    
    st.markdown("---")
    st.markdown("### Why the December 50% Premium?")
    st.markdown("""
    Real business data shows December sales are 50% higher than linear elasticity alone predicts:
    
    1. **Holiday demand surge:** Consumers/businesses buy more before year-end
    2. **Budget flush:** Companies spend remaining annual budgets in Q4
    3. **Tax incentives:** Year-end purchase incentives drive incremental sales
    4. **Gift-giving season:** Retail peak for corporate and consumer gifting
    
    This model captures that structural reality, making forecasts more realistic for planning.
    """)
    
    st.markdown("---")
    st.markdown("### Next Steps")
    st.markdown("""
    1. **Validate** forecast accuracy as actual sales data arrives in Jan–Mar 2026
    2. **Retrain** model quarterly with new data to improve coefficients
    3. **Compare** forecast vs. actuals to refine assumptions
    4. **Expand** to segment-level forecasts (by product, channel, geography)
    5. **Integrate** with P&L planning and budget setting
    """)
    
    st.markdown("---")
    st.info("**Created for FP&A teaching demo | May 2026**")
