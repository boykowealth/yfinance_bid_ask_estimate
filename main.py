import pandas as pd
import numpy as np

import yfinance as yf
import datetime

from sklearn.linear_model import LinearRegression

#-----------------------------------------Models-----------------------------------------#
class models():
    
    def price_1m(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=7)), end=datetime.date.today(), interval='1m')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_5m(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=30)), end=datetime.date.today(), interval='5m')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_15m(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=30)), end=datetime.date.today(), interval='15m')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_30m(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=30)), end=datetime.date.today(), interval='30m')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_1h(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=60)), end=datetime.date.today(), interval='1h')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_1d(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=1825)), end=datetime.date.today(), interval='1d')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1 
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_1w(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=1825)), end=datetime.date.today(), interval='1w')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1 
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data

    def price_1mo(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=1825)), end=datetime.date.today(), interval='1mo')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1 
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
    
    def price_1q(ticker):
        # Fetch and preprocess data
        data = yf.download(tickers=ticker, start=(datetime.date.today() - datetime.timedelta(days=1825)), end=datetime.date.today(), interval='3mo')
        data = data[['High', 'Low', 'Adj Close', 'Volume']].copy()
        data.columns = ['HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        # Add features
        data.loc[:, 'RET'] = (data['CLOSE'] / data['CLOSE'].shift(1)) - 1 
        data.loc[:, 'VOLATILITY'] = data['RET'].rolling(window=10).std()
        data.loc[:, 'MID_PRICE'] = (data['HIGH'] + data['LOW']) / 2
        data.loc[:, 'SPREAD_ESTIMATE'] = (data['HIGH'] - data['LOW']) / data['CLOSE']
        data.loc[:, 'VOL_AVG'] = data['VOLUME'].rolling(window=10).mean()
        data.loc[:, 'VOL_RATIO'] = data['VOLUME'] / data['VOL_AVG']
        data.dropna(inplace=True)
        
        # Define features and target
        X = data[['HIGH', 'LOW', 'VOLATILITY', 'VOLUME', 'VOL_RATIO']]
        y = data['SPREAD_ESTIMATE']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict spread and calculate bid/ask
        data.loc[:, 'SPREAD_ESTIMATE'] = model.predict(X)
        data.loc[:, 'SPREAD_ESTIMATE'] = np.clip(data['SPREAD_ESTIMATE'], 0.001, 0.02 * data['CLOSE'])
        data.loc[:, 'BID'] = np.maximum(data['CLOSE'] - data['SPREAD_ESTIMATE'] / 2, 0)
        data.loc[:, 'ASK'] = data['CLOSE'] + data['SPREAD_ESTIMATE'] / 2

        data = data[['CLOSE', 'BID', 'ASK', 'VOLUME', 'VOL_AVG', 'RET', 'VOLATILITY']]
        
        return data
