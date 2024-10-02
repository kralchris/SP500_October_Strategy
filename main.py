"""
S&P 500 Prediction Model - Quantitative Trading with Machine Learning and Macroeconomic Indicators

@author: Kristijan <kristijan.sarin@gmail.com>
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD


# 1. Load data
def load_data(ticker="^GSPC", period="5y"):
    df = yf.download(ticker, period=period)
    return df

# Load macro
def load_macroeconomic_data():
    tickers = {
        'US10Y': '^TNX',
        'DXY': 'DX-Y.NYB',
        'WTI': 'CL=F',
        'Gold': 'GC=F',
        'SP500': '^GSPC',
        'EURIBOR': '^IRX',
        'US2Y': '^FVX',
        'Copper': 'HG=F',
        'Brent': 'BZ=F',
        'VIX': '^VIX',
    }

    data = {}
    for name, ticker in tickers.items():
        data[name] = yf.download(ticker, period='5y')['Close']

    macro_df = pd.DataFrame(data)
    return macro_df

# Add sentiment
def add_sentiment_feature(df):
    # Create a simple random sentiment feature for demonstration
    np.random.seed(42)
    df['Sentiment'] = np.random.choice([1, 0, -1], size=len(df))
    return df

# 2. Feature Engineering: Tech indicators and macro
def add_technical_indicators(df, macro_df):
    rsi_7 = RSIIndicator(df['Close'], window=7).rsi()
    rsi_14 = RSIIndicator(df['Close'], window=14).rsi()
    rsi_21 = RSIIndicator(df['Close'], window=21).rsi()

    df['RSI_7'] = rsi_7
    df['RSI_14'] = rsi_14
    df['RSI_21'] = rsi_21

    macd = MACD(df['Close']).macd_diff()
    df['MACD'] = macd

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df = df.join(macro_df, how='inner')
    df = df.fillna(method='ffill')

    df = add_sentiment_feature(df)
    
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df.dropna()

# 3. Prepare data
def prepare_data(df):
    features = ['RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'SMA_10', 'SMA_50', 'SMA_100', 'SMA_200',
                'US10Y', 'DXY', 'WTI', 'Gold', 'SP500', 'EURIBOR', 'US2Y', 'Copper', 'Brent', 'VIX', 'Sentiment']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# backtest function
def backtest_trades(prices, predictions, profit_target=0.01, stop_loss=0.02):
    trades = []
    portfolio_balance = [10000]  # Start with $10,000
    balance = 10000
    successful_trades = 0
    for i in range(len(predictions) - 1):
        if predictions[i] == 1:
            buy_price = prices.iloc[i]
            for j in range(i + 1, len(prices)):
                price_change = (prices.iloc[j] - buy_price) / buy_price
                if price_change >= profit_target:
                    balance *= 1 + price_change
                    successful_trades += 1
                    break
                elif price_change <= -stop_loss:
                    balance *= 1 + price_change
                    break
        portfolio_balance.append(balance)
    
    return portfolio_balance, successful_trades / len(predictions)

# 4. Training, Evaluation, and Backtesting Models
def train_evaluate_and_backtest(models, X_train, X_test, y_train, y_test, df):
    results = {}
    portfolio_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}")

        portfolio_balance, success_rate = backtest_trades(df['Close'], y_pred)
        portfolio_results[name] = portfolio_balance

        print(f"{name} Success Rate: {success_rate * 100:.2f}%")

        # Plot the portfolio balance over time
        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_balance, label=f"{name} Portfolio Balance")
        plt.title(f"Portfolio Balance Over Time ({name})")
        plt.xlabel("Trades")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()

    return results, portfolio_results

# 5. Heatmap
def plot_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# 6. Main Execution
df = load_data()
macro_df = load_macroeconomic_data()
df = add_technical_indicators(df, macro_df)

# Plot heatmap
plot_heatmap(df)

X_train, X_test, y_train, y_test = prepare_data(df)
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results, portfolio_results = train_evaluate_and_backtest(models, X_train, X_test, y_train, y_test, df)

# Output results
print("Model Accuracy and Portfolio Performance:")
for model_name, accuracy in results.items():
    print(f"{model_name}: Accuracy = {accuracy:.2f}, Final Portfolio Value = {portfolio_results[model_name][-1]:.2f}")
