"""
Machine Learning Stock Price Predictor
=======================================
Uses a Random Forest Regressor (ensemble learning) to predict next-day
stock prices based on technical indicators engineered from historical data.

Concepts demonstrated:
- Feature engineering (moving averages, RSI, Bollinger Bands)
- Train/test split for model evaluation
- scikit-learn RandomForestRegressor
- Model metrics: MSE, R² score, RMSE
- Visualization of actual vs. predicted prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class StockPredictor:
    """
    Predicts next-day stock closing prices using a Random Forest model.

    Random Forest is an ensemble of decision trees - each tree votes on
    the prediction, reducing overfitting compared to a single tree.
    """

    def __init__(self, ticker="AAPL", period="2y"):
        """
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG', 'TSLA')
            period (str): Data period ('1y', '2y', '5y')
        """
        self.ticker = ticker.upper()
        self.period = period
        self.model = RandomForestRegressor(
            n_estimators=200,       # 200 decision trees in the forest
            max_depth=10,           # Limit tree depth to avoid overfitting
            min_samples_split=5,
            random_state=42         # Reproducible results
        )
        self.scaler = StandardScaler()  # Normalize features (zero mean, unit variance)
        self.data = None
        self.features = [
            'Close', 'Volume', 'High', 'Low',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'Signal_Line',
            'BB_Upper', 'BB_Lower', 'BB_Width',
            'Volume_MA', 'Price_Change', 'Volume_Change',
            'Daily_Return', 'Volatility'
        ]
        self.is_trained = False
        self.train_results = None

    def fetch_data(self):
        """
        Download historical stock data using yfinance.

        Returns:
            pandas.DataFrame: Raw OHLCV data
        """
        print(f"[ML] Fetching {self.period} of data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)

        if self.data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker}")

        print(f"[ML] Downloaded {len(self.data)} trading days of data.")
        return self.data

    def engineer_features(self):
        """
        Create technical indicator features from raw price data.

        This is the 'feature engineering' step in ML - transforming raw
        data into meaningful signals the model can learn from.

        Returns:
            pandas.DataFrame: Data with engineered features
        """
        df = self.data.copy()

        # --- Moving Averages (trend following) ---
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # --- RSI (Relative Strength Index: 0-100, overbought >70, oversold <30) ---
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- MACD (Moving Average Convergence Divergence) ---
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # --- Bollinger Bands (volatility measure) ---
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

        # --- Volume indicators ---
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Change'] = df['Volume'].pct_change()

        # --- Price momentum ---
        df['Price_Change'] = df['Close'].pct_change()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

        # Target: predict NEXT day's closing price
        df['Target'] = df['Close'].shift(-1)

        df.dropna(inplace=True)
        return df

    def train(self):
        """
        Train the Random Forest model on engineered features.

        Uses an 80/20 train/test split (no shuffle - time series order matters).

        Returns:
            dict: Model performance metrics and predictions
        """
        if self.data is None:
            self.fetch_data()

        print("[ML] Engineering features...")
        df = self.engineer_features()

        X = df[self.features].values
        y = df['Target'].values
        dates = df.index

        # Split - keep chronological order (no shuffle for time series!)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_test = dates[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"[ML] Training Random Forest on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        self.is_trained = True
        self.train_results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'dates_test': dates_test
        }

        print("\n[ML] === Model Performance ===")
        print(f"  R² Score:  {r2:.4f}  (1.0 = perfect)")
        print(f"  RMSE:      ${rmse:.2f}")
        print(f"  MAE:       ${mae:.2f}")

        return self.train_results

    def predict_next_day(self):
        """
        Predict the next trading day's closing price.

        Returns:
            tuple: (predicted_price, current_price, expected_change_%)
        """
        if not self.is_trained:
            self.train()

        df = self.engineer_features()
        last_row = df[self.features].iloc[-1:].values
        last_scaled = self.scaler.transform(last_row)
        prediction = self.model.predict(last_scaled)[0]
        current = df['Close'].iloc[-1]
        change_pct = ((prediction - current) / current) * 100

        return prediction, current, change_pct

    def get_feature_importance(self):
        """
        Return which features the model found most predictive.

        Returns:
            pandas.Series: Feature importance scores sorted descending
        """
        if not self.is_trained:
            return None

        importance = pd.Series(
            self.model.feature_importances_,
            index=self.features
        ).sort_values(ascending=False)

        return importance

    def plot_results(self, save_path=None):
        """
        Plot actual vs. predicted stock prices.

        Args:
            save_path (str): Optional path to save the chart as PNG
        """
        if not self.is_trained:
            print("[ML] Model not trained yet. Call train() first.")
            return

        r = self.train_results
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'{self.ticker} - ML Stock Price Prediction (Random Forest)', fontsize=14, fontweight='bold')

        # --- Price prediction chart ---
        ax1 = axes[0]
        ax1.plot(r['dates_test'], r['y_test'], label='Actual Price', color='#2196F3', linewidth=2)
        ax1.plot(r['dates_test'], r['y_pred'], label='Predicted Price', color='#F44336',
                 linewidth=1.5, linestyle='--', alpha=0.8)
        ax1.fill_between(r['dates_test'], r['y_test'], r['y_pred'],
                          alpha=0.1, color='orange', label='Prediction Error')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'Actual vs. Predicted Closing Price | R² = {r["r2"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # --- Feature importance chart ---
        ax2 = axes[1]
        importance = self.get_feature_importance().head(10)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
        bars = ax2.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top 10 Most Important Features')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, importance.values[::-1]):
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[ML] Chart saved to: {save_path}")

        plt.show()

    def summary(self):
        """
        Return a text summary of the prediction results for use in LLM context.

        Returns:
            str: Human-readable summary
        """
        if not self.is_trained:
            return "Model has not been trained yet."

        pred, current, change_pct = self.predict_next_day()
        r = self.train_results
        direction = "UP" if change_pct > 0 else "DOWN"

        return (
            f"Stock: {self.ticker}\n"
            f"Current Price: ${current:.2f}\n"
            f"Predicted Next Day: ${pred:.2f} ({direction} {abs(change_pct):.2f}%)\n"
            f"Model Accuracy (R²): {r['r2']:.4f}\n"
            f"Prediction Error (RMSE): ${r['rmse']:.2f}\n"
            f"Training period: {self.period}\n"
            f"Model: Random Forest (200 trees, max_depth=10)\n"
            f"Features used: {len(self.features)} technical indicators"
        )
