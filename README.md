# y-intercept-interview


This strategy is a simplified and modified version of my work in Huatai as an intern.

## Step 1 Data preprocessing

File: s1_data_preprocessing.py

Change the data to time-series, and drop assets that have more than have more than 40% total val which are missing and drop assets that have more than 20% val in recent one year which are missing.

## Step 2 Substrategy for every asset

File: s2_substrategy.py

1. I use three common technical analysis factors MACD, RSI and Bollinger Bands with serveral parameter combinations to generate long signal (the short signal is set to 0 for long-only setting). Here is the parameter dictionary:

   strat_params = {
        'macd': {'fastperiod': [7, 12, 17], 'slowperiod': [22, 26, 31], 'signalperiod': 9},
        'rsi': {'timeperiod': [4, 14, 24], 'buy_threshold': [20, 30, 40], 'sell_threshold': [60, 70, 80]},
        'bolling': {'timeperiod': [10, 20, 30], 'nbdev': [1, 2, 3]},
    }

2. For every asset, I test all above substrategies on it, and get the portfolio value columns. Then I use the modified combinatorially syymetric CV to compute PBO (Probability of Backtest Overfitting) which means the probability that for the best in-sample substrategy, its out-sample performance ranking falls behind the median.

## Step 3 Stock Picking and Portfolio Construction

File: s3_stock_picking.py

1. I pick the stock with PBO less than 0.2, and there are 20 left.
   
2. I construct a simple long-only equal-weighted portfolio.

3. I backtest it from 2013-01-24 to 2021-03-19. The result is the following:

|    |   Total Return |   Return p.a. |   Volatility |   Sharpe Ratio |   Max Drawdown |   Calmar Ratio |
|---:|---------------:|--------------:|-------------:|---------------:|---------------:|---------------:|
|  0 |        7.20286 |      0.294482 |      0.22749 |        1.29448 |      -0.273301 |         1.0775 |

I save the cumulative pnl curve in the /pic/cumulative_pnl.png
The weight table is in /data/weight_table.csv

## Flaws and Potential Improvements

1. The current version cares little about the correlation between assets. Maybe can use some optimizer in the portfolio construction step.
2. Increase the number of technical factors and parameters.
3. Use cash based backtesting framework.