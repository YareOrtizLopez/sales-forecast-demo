# Sales Forecast Demo

An interactive Streamlit dashboard for FP&A professionals to forecast monthly sales using marketing spend data with explainable linear regression.

## Overview

This project demonstrates a realistic sales forecasting workflow:
1. **Load** marketing spend data (36 months)
2. **Engineer features** (trend, seasonality, elasticity)
3. **Generate synthetic sales** using realistic business logic
4. **Train** a linear regression model
5. **Forecast** the next 3 months with confidence intervals
6. **Visualize** for executive presentation

## Key Features

- **Explainable AI:** Linear regression coefficients translate directly to business metrics
- **Realistic Seasonality:** December spikes modeled as 50% larger than elasticity predicts (reflects real holiday/Q4 demand)
- **Interactive Dashboard:** Streamlit UI for exploring data, model, and forecasts
- **Confidence Intervals:** 95% bounds for risk-aware planning

## Why the December 50% Premium?

In most retail/B2B businesses, December sales don't just reflect linear marketing spend elasticity. There are structural reasons for a seasonal boost:
- **Holiday demand surge:** Consumers/businesses buy more before year-end
- **Budget flush:** Companies spend remaining annual budgets in Q4
- **Tax incentives:** Year-end purchase incentives drive incremental sales
- **Gift-giving season:** Retail peak for gifting

This model captures that **50% multiplicative premium on top of the elasticity effect**, making the forecast realistic for budgeting conversations.

## Data

**Source:** `data/marketing_spend.csv`
- 36 months (Jan 2023 – Dec 2025)
- Marketing spend: $5,150 – $11,000 per month
- Linear trend + December seasonality

## Model

**Linear Regression Formula:**
```
Sales = Base + (Trend × Month) + (Elasticity × Spend) + (December_Premium × Is_December) + Error
```

**Interpretation:**
- `Base`: Baseline monthly sales (no marketing effect)
- `Trend`: Organic growth/decay per month
- `Elasticity`: Sales increase per $1 marketing spend (e.g., $2.50 in sales per $1 spent)
- `December_Premium`: Additional sales boost in December (captures seasonality)
- `Error`: Residual noise (modeled as realistic, non-zero)

## Project Structure

```
sales-forecast-demo/
├── README.md
├── requirements.txt
├── app.py                          # Main Streamlit dashboard
├── data/
│   └── marketing_spend.csv         # Input data
└── src/
    ├── data_loader.py              # Load & validate CSV
    ├── feature_engineering.py       # Create features
    ├── synthetic_sales_generator.py # Generate realistic sales
    ├── model.py                     # Train & forecast
    └── utils.py                     # Plotting & validation helpers
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will launch at `http://localhost:8501`

## Usage

1. **Data Exploration:** View 36 months of marketing spend with trend analysis
2. **Synthetic Sales:** Auto-generated sales following the realistic elasticity + seasonality model
3. **Model Coefficients:** See the fitted regression equation with interpretable business metrics
4. **Forecast:** 3-month ahead forecast (Jan–Mar 2026) with 95% confidence intervals
5. **Export:** Download forecast results as CSV

## Example Output

For input spend of ~$11,150–$11,450 (Jan–Mar 2026):

| Month | Marketing Spend | Forecast Sales | 95% CI Low | 95% CI High |
|-------|-----------------|----------------|------------|-------------|
| Jan 2026 | $11,150 | $46,475 | $44,200 | $48,750 |
| Feb 2026 | $11,300 | $47,575 | $45,100 | $50,050 |
| Mar 2026 | $11,450 | $48,675 | $46,000 | $51,350 |

---

**Created for FP&A teaching demo | May 2026**
