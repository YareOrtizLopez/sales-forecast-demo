import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

class SalesForecaster:
    """
    Linear regression model for sales forecasting based on marketing spend.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
        self.coefficients = None
        self.intercept = None
        self.r2 = None
        self.rmse = None
        self.mae = None
        self.residuals = None
        self.y_pred_train = None
    
    def fit(self, X, y, feature_names):
        """
        Train the linear regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target sales (n_samples,)
            feature_names: List of feature column names
        """
        self.model.fit(X, y)
        self.feature_names = feature_names
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        # Compute metrics
        self.y_pred_train = self.model.predict(X)
        self.r2 = r2_score(y, self.y_pred_train)
        self.rmse = np.sqrt(mean_squared_error(y, self.y_pred_train))
        self.mae = mean_absolute_error(y, self.y_pred_train)
        self.residuals = y - self.y_pred_train
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions (1D array)
        """
        return self.model.predict(X)
    
    def predict_with_ci(self, X, confidence=0.95):
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Feature matrix
            confidence: Confidence level (default 0.95 for 95%)
        
        Returns:
            predictions, ci_lower, ci_upper (all 1D arrays)
        """
        predictions = self.predict(X)
        
        # Standard error of residuals
        se = np.sqrt(np.sum(self.residuals**2) / (len(self.residuals) - len(self.feature_names) - 1))
        
        # t-critical value
        n = len(self.residuals)
        df = n - len(self.feature_names) - 1
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        
        # Margin of error
        margin = t_crit * se
        
        ci_lower = predictions - margin
        ci_upper = predictions + margin
        
        return predictions, ci_lower, ci_upper
    
    def get_formula(self):
        """
        Return the fitted regression equation as a readable string.
        """
        formula = f"Sales = {self.intercept:,.0f}"
        for fname, coef in zip(self.feature_names, self.coefficients):
            sign = '+' if coef >= 0 else '-'
            formula += f" {sign} {abs(coef):,.2f} × {fname}"
        return formula
    
    def get_summary(self):
        """
        Return a summary of model performance.
        """
        summary = f"""
        === LINEAR REGRESSION MODEL SUMMARY ===
        
        Equation: {self.get_formula()}
        
        Performance Metrics:
        - R² Score: {self.r2:.4f}
        - RMSE (Root Mean Squared Error): ${self.rmse:,.2f}
        - MAE (Mean Absolute Error): ${self.mae:,.2f}
        
        Coefficients:
        - Intercept: ${self.intercept:,.2f}
        """
        for fname, coef in zip(self.feature_names, self.coefficients):
            summary += f"\n  - {fname}: {coef:,.4f}"
        
        return summary
