import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# CONFIGURATION
# =========================
TICKERS = ["SPY", "QQQ", "GLD", "TLT", "USO", "BTC-USD"]
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"
ROLLING_WINDOW = 30


# =========================
# DOWNLOAD DATA
# =========================
def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all")
    return prices


# =========================
# CALCULATIONS
# =========================
def calculate_daily_returns(prices):
    return prices.pct_change().dropna() #percentage change between consecutive values 


def calculate_correlation_matrix(daily_returns):
    return daily_returns.corr() #computes the correlation


def calculate_rolling_correlation(daily_returns, asset_1, asset_2, window=30):
    return daily_returns[asset_1].rolling(window=window).corr(daily_returns[asset_2])


def extract_correlation_pairs(corr_matrix):
    pairs = []

    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            asset_1 = columns[i]
            asset_2 = columns[j]
            corr_value = corr_matrix.iloc[i, j]
            pairs.append((asset_1, asset_2, corr_value))

    pairs_df = pd.DataFrame(pairs, columns=["Asset 1", "Asset 2", "Correlation"])
    return pairs_df.sort_values(by="Correlation", ascending=False)


def build_summary(daily_returns, corr_matrix):
    summary = pd.DataFrame(index=daily_returns.columns)

    summary["Mean Daily Return"] = daily_returns.mean()
    summary["Daily Volatility"] = daily_returns.std()
    summary["Annualized Volatility"] = daily_returns.std() * np.sqrt(252)
    summary["Average Correlation"] = corr_matrix.mean()

    return summary.sort_values(by="Average Correlation", ascending=False)


# =========================
# MAIN
# =========================
def main():
    print("Downloading market data...")
    prices = download_prices(TICKERS, START_DATE, END_DATE)

    print("\nLast 5 rows of price data:")
    print(prices.tail())

    daily_returns = calculate_daily_returns(prices)
    corr_matrix = calculate_correlation_matrix(daily_returns)

    rolling_corr_spy_qqq = calculate_rolling_correlation(
        daily_returns, "SPY", "QQQ", ROLLING_WINDOW
    )

    corr_pairs = extract_correlation_pairs(corr_matrix)
    summary = build_summary(daily_returns, corr_matrix)

    print("\n================ SUMMARY TABLE ================\n")
    print(summary)

    print("\n================ CORRELATION MATRIX ================\n")
    print(corr_matrix)

    print("\n================ STRONGEST CORRELATIONS ================\n")
    print(corr_pairs.head(5))

    print("\n================ WEAKEST CORRELATIONS ================\n")
    print(corr_pairs.tail(5))

    print(f"\n================ LAST 5 VALUES: {ROLLING_WINDOW}-DAY ROLLING CORRELATION (SPY vs QQQ) ================\n")
    print(rolling_corr_spy_qqq.tail())

    print("\n================ DIVERSIFICATION NOTE ================\n")
    print("Lower correlation between assets may help diversification.")
    print("Higher correlation means assets tend to move together.")


if __name__ == "__main__":
    main()