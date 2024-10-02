# S&P 500 Prediction Model: Quantitative Trading with Machine Learning and Macroeconomic Indicators

## Overview
This repository demonstrates a sophisticated approach to predicting asset price movements using machine learning techniques, technical indicators, and macroeconomic data. As a quantitative trader and portfolio manager, I have combined several models to optimize trading signals and backtest strategies on the S&P 500 index. The objective is to leverage machine learning models to generate high-confidence trade signals, maximize portfolio performance, and control risk through systematic backtesting using predefined profit and stop-loss targets.

### Key Features
- **Machine Learning Algorithms:** Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM) to predict market direction.
- **Technical Indicators:** Incorporation of momentum and trend-following indicators such as RSI, MACD, and various SMAs.
- **Macroeconomic Data:** Integration of key macroeconomic indicators (US Treasury yields, dollar strength, commodity prices, volatility index) to enhance predictive power.
- **Sentiment Analysis:** Implementation of a sentiment feature to add a behavioral dimension to the models.
- **Backtesting Engine:** A fully functional backtest engine that simulates trades based on model outputs, using a 1% take profit and 2% stop-loss rule for portfolio optimization.

## Table of Contents
1. [Installation](#installation)
2. [Data Sources](#data-sources)
3. [Machine Learning Models](#machine-learning-models)
4. [Backtesting](#backtesting)
5. [Usage](#usage)
6. [Results](#results)
7. [Next Steps](#next-steps)
8. [License](#license)

### Key Dependencies:
- `pandas`, `numpy`: For data manipulation and numerical computations.
- `yfinance`: For fetching historical financial data.
- `scikit-learn`: Machine learning algorithms and evaluation tools.
- `ta`: For calculating technical analysis indicators.
- `matplotlib`, `seaborn`: For visualization.
  
## Data Sources
The model is built on historical price data for the S&P 500 (^GSPC), combined with a range of macroeconomic indicators fetched through the `yfinance` API:
- **S&P 500 Index (^GSPC)**: Price data for the primary asset.
- **US10Y**: U.S. 10-year Treasury yield.
- **DXY**: U.S. Dollar Index.
- **WTI (Crude Oil)**: West Texas Intermediate crude oil price.
- **Gold**: Gold futures price.
- **EURIBOR**: EURIBOR interest rate.
- **VIX**: Volatility Index.

These indicators are used to create a more robust feature set for the machine learning models.

## Machine Learning Models
The repository includes the implementation of several machine learning models trained to predict the movement of the S&P 500:
- **Logistic Regression:** A simple but effective classifier for binary outcomes (up/down market movements).
- **Decision Tree:** A non-parametric model that captures non-linear relationships in the data.
- **Random Forest:** An ensemble learning method that improves on decision trees by reducing overfitting and variance.
- **Support Vector Machine (SVM):** A powerful classifier that works well in high-dimensional spaces, separating data by finding the optimal hyperplane.

### Feature Engineering
- **Technical Indicators:** RSI (7, 14, 21 periods), MACD, and moving averages (10, 50, 100, and 200 periods) are used to capture price momentum and trends.
- **Macroeconomic Data:** The macroeconomic indicators listed above are integrated to provide additional context to price movements.
- **Sentiment Feature:** A placeholder sentiment feature is included, which can be extended using real sentiment data.

## Backtesting
The backtesting engine simulates trades based on model predictions, applying a strict risk management strategy:
- **Take Profit:** 1% gain triggers an exit from a position.
- **Stop Loss:** 2% loss triggers an exit from a position to limit downside risk.

The backtest outputs:
- **Portfolio Balance:** Evolution of the portfolio's value over time.
- **Successful Trade Rate:** The percentage of trades that hit the take profit target before the stop loss.

## Usage
1. **Loading Data:**
   The model uses `yfinance` to download S&P 500 data and macroeconomic indicators over the past 5 years.

2. **Training Models:**
   Use the provided Python script to train models on historical data, perform predictions, and evaluate model accuracy.

```python
python main.py
```

3. **Backtesting:**
   The script will backtest each model, displaying both the percentage of successful trades and the final portfolio balance.

```python
# Example function call to initiate backtest
train_evaluate_and_backtest(models, X_train, X_test, y_train, y_test, df)
```

## Results
The results are displayed as:
- **Model Accuracy:** The predictive accuracy of each machine learning model on test data.
- **Backtest Portfolio Balance:** Portfolio performance over time, highlighting the impact of each trading strategy.
- **Heatmap of Correlations:** Visual representation of correlations between different features, assisting in feature selection and strategy refinement.

## Next Steps
- **Enhanced Sentiment Analysis:** Incorporate advanced sentiment data from news, social media, or financial reports.
- **Deep Learning Models:** Extend the analysis by experimenting with LSTMs or neural networks for improved prediction accuracy.
- **Dynamic Risk Management:** Introduce more sophisticated risk management techniques such as trailing stop-losses or position sizing algorithms.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

As a quant trader, I continuously optimize these strategies by integrating cutting-edge machine learning models with rigorous backtesting frameworks. The methods presented here are designed to maximize returns while controlling risk in an ever-evolving financial landscape.


## Output
![notebook_1](https://github.com/user-attachments/assets/8203bcc6-9f72-4b2b-a259-6d79ffa200a7)
![notebook_2](https://github.com/user-attachments/assets/fe21efae-6332-4e5f-bed0-76f76a97b4fb)
