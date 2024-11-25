# Bid-Ask Estimate From Yahoo Finance Datasets

This repository contains Python-based financial models that estimate bid and ask prices for various time intervals using data fetched from Yahoo Finance. These models use features derived from historical price and volume data and a linear regression model to predict spreads and calculate bid/ask prices.

## Features

- Fetches historical data for specified tickers using Yahoo Finance.
- Calculates various features such as:
  - Returns (`RET`)
  - Volatility (`VOLATILITY`)
  - Mid Price (`MID_PRICE`)
  - Spread Estimate (`SPREAD_ESTIMATE`)
  - Average Volume (`VOL_AVG`)
  - Volume Ratio (`VOL_RATIO`)
- Implements linear regression to predict spread estimates.
- Dynamically calculates bid and ask prices based on predicted spreads.

## Installation

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install pandas numpy yfinance scikit-learn
```

## Usage

### Import the Module

```python
from models import models
```

### Call Price Estimation Functions

Each function corresponds to a specific time interval for fetching and processing data. Supported intervals:

- 1 Minute: `models.price_1m(ticker)`
- 5 Minutes: `models.price_5m(ticker)`
- 15 Minutes: `models.price_15m(ticker)`
- 30 Minutes: `models.price_30m(ticker)`
- 1 Hour: `models.price_1h(ticker)`
- 1 Day: `models.price_1d(ticker)`
- 1 Week: `models.price_1w(ticker)`
- 1 Month: `models.price_1mo(ticker)`
- 1 Quarter: `models.price_1q(ticker)`

### Example Usage

```python
import pandas as pd
from models import models

# Fetch and calculate data for a specific ticker
ticker = "AAPL"
data = models.price_1d(ticker)

# Display the processed data
print(data.head())
```

### Output DataFrame

The resulting DataFrame contains the following columns:
- `CLOSE`: Adjusted closing price
- `BID`: Estimated bid price
- `ASK`: Estimated ask price
- `VOLUME`: Trading volume
- `VOL_AVG`: Average trading volume
- `RET`: Return (price change percentage)
- `VOLATILITY`: Rolling volatility

## How It Works

1. **Data Fetching:** Retrieves historical data for a given ticker using Yahoo Finance.
2. **Feature Engineering:** Computes features based on historical price and volume data.
3. **Model Training:** Trains a `LinearRegression` model using selected features.
4. **Price Estimation:** Uses the trained model to predict spreads and calculates bid/ask prices.

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `yfinance`: Fetching stock data
- `scikit-learn`: Machine learning (linear regression)

## Limitations

- Relies on Yahoo Finance data, which may have rate limits or availability constraints.
- Linear regression might not capture non-linear relationships in the data.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for improvements or bug fixes.
