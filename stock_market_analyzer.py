"""
Stock Market Performance Analyzer
---------------------------------
End-to-end analytics pipeline for four Indian IT stocks (TCS, INFY, WIPRO, HCLTECH):

  1. Data      - download daily prices (yfinance) and cache them to data/stock_data.csv
  2. Analytics - daily & cumulative returns, SMA-20 / SMA-50, 20-day annualised volatility
  3. Charts    - 5 chart types saved to screenshots/
  4. ML        - classifier that predicts next-day price direction (up / down)

Run:  python stock_market_analyzer.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TICKERS = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
}
START_DATE = "2021-01-01"
END_DATE = None  # None = up to today
TRADING_DAYS = 252

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "stock_data.csv"
SHOT_DIR = BASE_DIR / "screenshots"


# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------
def download_prices() -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(
        list(TICKERS.values()), start=START_DATE, end=END_DATE,
        auto_adjust=True, progress=False,
    )["Close"]
    raw = raw.rename(columns={v: k for k, v in TICKERS.items()})
    return raw[list(TICKERS)].dropna(how="all")


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Load prices from the CSV cache, downloading them first if needed."""
    if refresh or not DATA_PATH.exists():
        print("Downloading price data from Yahoo Finance...")
        prices = download_prices()
        DATA_PATH.parent.mkdir(exist_ok=True)
        prices.to_csv(DATA_PATH, index_label="Date")
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return prices.sort_index().ffill().dropna()


# ----------------------------------------------------------------------------
# 2. Analytics
# ----------------------------------------------------------------------------
def compute_metrics(prices: pd.DataFrame) -> dict:
    daily = prices.pct_change().dropna()
    cumulative = (1 + daily).cumprod() - 1
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    volatility = daily.rolling(20).std() * np.sqrt(TRADING_DAYS)  # annualised
    return {
        "daily": daily, "cumulative": cumulative,
        "sma20": sma20, "sma50": sma50, "volatility": volatility,
    }


def print_summary(prices: pd.DataFrame, m: dict) -> None:
    summary = pd.DataFrame({
        "Total Return (%)": m["cumulative"].iloc[-1] * 100,
        "Annualised Volatility (%)": m["daily"].std() * np.sqrt(TRADING_DAYS) * 100,
        "Avg Daily Return (%)": m["daily"].mean() * 100,
    }).round(2)
    print(f"\nPeriod: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(summary.to_string())
    print(f"\nBest return : {summary['Total Return (%)'].idxmax()}")
    print(f"Most volatile: {summary['Annualised Volatility (%)'].idxmax()}\n")


# ----------------------------------------------------------------------------
# 3. Charts
# ----------------------------------------------------------------------------
def plot_all(prices: pd.DataFrame, m: dict) -> None:
    SHOT_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    def chart_price(ax):
        prices.plot(ax=ax)
        ax.set(title="Price History", ylabel="Price (INR)")

    def chart_cumulative(ax):
        (m["cumulative"] * 100).plot(ax=ax)
        ax.set(title="Cumulative Returns", ylabel="Return (%)")

    def chart_sma(ax):
        stock = prices.columns[0]
        prices[stock].plot(ax=ax, label="Close", lw=1)
        m["sma20"][stock].plot(ax=ax, label="SMA-20")
        m["sma50"][stock].plot(ax=ax, label="SMA-50")
        ax.set(title=f"{stock}: Moving Averages", ylabel="Price (INR)")
        ax.legend()

    def chart_vol(ax):
        (m["volatility"] * 100).plot(ax=ax)
        ax.set(title="20-Day Rolling Volatility (annualised)", ylabel="Volatility (%)")

    def chart_dist(ax):
        for col in m["daily"]:
            ax.hist(m["daily"][col] * 100, bins=60, alpha=0.5, label=col)
        ax.set(title="Daily Return Distribution", xlabel="Daily return (%)", ylabel="Frequency")
        ax.legend()

    charts = {
        "1_price_history": chart_price,
        "2_cumulative_returns": chart_cumulative,
        "3_moving_averages": chart_sma,
        "4_volatility": chart_vol,
        "5_return_distribution": chart_dist,
    }

    # individual charts
    for name, fn in charts.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        fn(ax)
        fig.tight_layout()
        fig.savefig(SHOT_DIR / f"{name}.png", dpi=150)
        plt.close(fig)

    # combined dashboard -> screenshots/output.png (use this one in the README)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    for ax, fn in zip(axes.ravel(), charts.values()):
        fn(ax)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Indian IT Stocks - Performance Dashboard", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(SHOT_DIR / "output.png", dpi=130)
    plt.close(fig)
    print(f"Charts saved to {SHOT_DIR}/")


# ----------------------------------------------------------------------------
# 4. Machine learning: predict next-day direction
# ----------------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(prices: pd.DataFrame, m: dict) -> pd.DataFrame:
    frames = []
    for s in prices.columns:
        p = prices[s]
        df = pd.DataFrame({
            "ret_1d": m["daily"][s],
            "ret_5d": p.pct_change(5),
            "sma_ratio": m["sma20"][s] / m["sma50"][s],
            "price_vs_sma20": p / m["sma20"][s] - 1,
            "volatility": m["volatility"][s],
            "rsi": rsi(p),
        })
        # Target: does tomorrow's close finish higher than today's? (1 = up)
        df["target"] = (p.shift(-1) > p).astype(int)
        df.loc[p.shift(-1).isna(), "target"] = np.nan  # last row has no "tomorrow"
        df["stock"] = s
        frames.append(df.dropna())
    return pd.concat(frames).sort_index()


def run_model(prices: pd.DataFrame, m: dict) -> None:
    data = build_features(prices, m)
    features = [c for c in data.columns if c not in ("target", "stock")]

    # Time-based split (no shuffling) so the model never sees the future
    split_date = data.index.unique()[int(len(data.index.unique()) * 0.8)]
    train, test = data[data.index < split_date], data[data.index >= split_date]
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print("=" * 60)
    print("NEXT-DAY DIRECTION PREDICTION")
    print("=" * 60)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows (from {split_date.date()})")
    print(f"Naive baseline (always predict majority class): {baseline:.3f}\n")

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"{name}: accuracy = {accuracy_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred, target_names=["Down", "Up"], digits=3))

    rf = models["Random Forest"]
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("Random Forest feature importance:")
    print(importance.round(3).to_string())
    print("\nNote: daily stock direction is close to a coin flip - treat results as a "
          "learning exercise, not trading advice.")


# ----------------------------------------------------------------------------
def main() -> None:
    prices = load_prices()
    metrics = compute_metrics(prices)
    print_summary(prices, metrics)
    plot_all(prices, metrics)
    run_model(prices, metrics)


if __name__ == "__main__":
    main()
