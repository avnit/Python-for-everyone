"""
Financial Data Module
=====================
Extends the yfinance work from Class-6 with more comprehensive data retrieval,
portfolio analysis, Sharpe ratio, correlation matrices, and risk metrics.

Builds directly on top of the Class-6 financial_functions.py foundation.
"""

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


class FinancialDataFetcher:
    """
    Fetch and analyze financial data for stocks and portfolios.

    Extends Class-6 work with:
    - Portfolio-level analysis
    - Risk metrics (Sharpe ratio, volatility)
    - Correlation analysis
    - Dividend tracking
    """

    RISK_FREE_RATE = 0.05  # 5% annual (approx. current T-bill rate)

    def __init__(self, tickers=None):
        """
        Args:
            tickers (list): List of stock ticker symbols
        """
        if tickers is None:
            tickers = ["AAPL"]
        self.tickers = [t.upper() for t in tickers]
        self.data = {}
        self.portfolio_data = None

    def get_stock_info(self, ticker):
        """
        Get company information for a single ticker.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Company info (name, sector, market cap, P/E ratio, etc.)
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 'N/A'),
            '52w_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52w_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }

    def fetch_history(self, ticker, start=None, end=None, period="1y"):
        """
        Fetch historical OHLCV data for a single ticker.

        Args:
            ticker (str): Stock symbol
            start (datetime): Start date
            end (datetime): End date
            period (str): Period string if no start/end ('1y', '2y', etc.)

        Returns:
            pandas.DataFrame: Historical price data
        """
        if start and end:
            df = yf.download(ticker, start=start, end=end, progress=False)
        else:
            df = yf.Ticker(ticker).history(period=period)

        self.data[ticker] = df
        return df

    def fetch_portfolio(self, start=None, end=None, period="1y"):
        """
        Fetch data for all tickers in the portfolio.

        Args:
            start (datetime): Start date
            end (datetime): End date
            period (str): Period string

        Returns:
            pandas.DataFrame: Multi-level DataFrame with all tickers
        """
        print(f"[Finance] Fetching portfolio data for: {', '.join(self.tickers)}")
        if start and end:
            self.portfolio_data = yf.download(
                self.tickers, start=start, end=end, progress=False
            )
        else:
            self.portfolio_data = yf.download(
                self.tickers, period=period, progress=False
            )
        return self.portfolio_data

    def calculate_returns(self, df):
        """
        Calculate daily and cumulative returns.

        Args:
            df (pandas.DataFrame): Price data with 'Close' column

        Returns:
            pandas.DataFrame: DataFrame with returns added
        """
        result = df.copy()
        result['Daily_Return'] = result['Close'].pct_change()
        result['Cumulative_Return'] = (1 + result['Daily_Return']).cumprod() - 1
        return result

    def sharpe_ratio(self, returns, annualize=True):
        """
        Calculate Sharpe Ratio (risk-adjusted return).

        Sharpe = (Return - Risk-Free Rate) / Standard Deviation
        Higher is better. >1 is good, >2 is great, >3 is excellent.

        Args:
            returns (pandas.Series): Daily return series
            annualize (bool): Convert daily to annual (multiply by sqrt(252))

        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - (self.RISK_FREE_RATE / 252)
        if annualize:
            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return excess_returns.mean() / excess_returns.std()

    def max_drawdown(self, prices):
        """
        Calculate maximum drawdown (worst peak-to-trough decline).

        Args:
            prices (pandas.Series): Closing price series

        Returns:
            float: Maximum drawdown as negative percentage
        """
        peak = prices.cummax()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def portfolio_summary(self, period="1y"):
        """
        Generate a comprehensive summary for all portfolio stocks.

        Returns:
            pandas.DataFrame: Metrics for each stock
        """
        if self.portfolio_data is None:
            self.fetch_portfolio(period=period)

        summary_rows = []

        for ticker in self.tickers:
            try:
                if len(self.tickers) == 1:
                    closes = self.portfolio_data['Close']
                else:
                    closes = self.portfolio_data['Close'][ticker]

                closes = closes.dropna()
                daily_returns = closes.pct_change().dropna()

                total_return = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
                volatility = daily_returns.std() * np.sqrt(252) * 100
                sharpe = self.sharpe_ratio(daily_returns)
                drawdown = self.max_drawdown(closes) * 100
                current_price = closes.iloc[-1]

                summary_rows.append({
                    'Ticker': ticker,
                    'Current Price': f'${current_price:.2f}',
                    'Total Return %': f'{total_return:.1f}%',
                    'Annual Volatility': f'{volatility:.1f}%',
                    'Sharpe Ratio': f'{sharpe:.2f}',
                    'Max Drawdown': f'{drawdown:.1f}%',
                })
            except Exception as e:
                print(f"[Finance] Could not process {ticker}: {e}")

        return pd.DataFrame(summary_rows)

    def plot_portfolio_performance(self, period="1y", save_path=None):
        """
        Plot normalized performance comparison of all stocks.

        Args:
            period (str): Data period
            save_path (str): Optional path to save chart
        """
        if self.portfolio_data is None:
            self.fetch_portfolio(period=period)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=14, fontweight='bold')

        # --- Normalized price performance ---
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.tickers)))

        for ticker, color in zip(self.tickers, colors):
            try:
                if len(self.tickers) == 1:
                    closes = self.portfolio_data['Close']
                else:
                    closes = self.portfolio_data['Close'][ticker]

                closes = closes.dropna()
                normalized = closes / closes.iloc[0] * 100
                ax1.plot(normalized.index, normalized.values, label=ticker,
                        color=color, linewidth=2)
            except Exception:
                continue

        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Base (100)')
        ax1.set_ylabel('Normalized Price (Base=100)')
        ax1.set_title('Relative Performance (normalized to 100)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Rolling volatility ---
        ax2 = axes[1]
        for ticker, color in zip(self.tickers, colors):
            try:
                if len(self.tickers) == 1:
                    closes = self.portfolio_data['Close']
                else:
                    closes = self.portfolio_data['Close'][ticker]

                returns = closes.dropna().pct_change()
                rolling_vol = returns.rolling(window=21).std() * np.sqrt(252) * 100
                ax2.plot(rolling_vol.index, rolling_vol.values, label=ticker,
                        color=color, linewidth=1.5)
            except Exception:
                continue

        ax2.set_ylabel('Annualized Volatility (%)')
        ax2.set_title('21-Day Rolling Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Finance] Chart saved to: {save_path}")

        plt.show()

    def correlation_heatmap(self, period="1y", save_path=None):
        """
        Plot a correlation matrix heatmap for the portfolio.

        Args:
            period (str): Data period
            save_path (str): Optional path to save chart
        """
        if len(self.tickers) < 2:
            print("[Finance] Need at least 2 tickers for correlation analysis.")
            return

        if self.portfolio_data is None:
            self.fetch_portfolio(period=period)

        closes = self.portfolio_data['Close'].dropna()
        returns = closes.pct_change().dropna()
        corr = returns.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap,
                   center=0, square=True, linewidths=0.5, ax=ax,
                   vmin=-1, vmax=1)

        ax.set_title('Portfolio Correlation Matrix\n(Returns-based, Pearson)', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def get_spy_portfolio(self):
        """
        Get top holdings from SPY ETF (reuses Class-6 funds.py approach).

        Returns:
            list: Ticker symbols of SPY top holdings
        """
        print("[Finance] Fetching SPY top holdings (from Class-6 funds.py pattern)...")
        try:
            spy = yf.Ticker("SPY").funds_data
            holdings = spy.top_holdings.index.tolist()
            print(f"[Finance] SPY top holdings: {holdings}")
            return holdings
        except Exception as e:
            print(f"[Finance] Could not fetch SPY data: {e}")
            return ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

    def summary_text(self):
        """
        Return portfolio summary as text for LLM context.

        Returns:
            str: Formatted summary
        """
        df = self.portfolio_summary()
        return f"Portfolio Tickers: {', '.join(self.tickers)}\n\n{df.to_string(index=False)}"
