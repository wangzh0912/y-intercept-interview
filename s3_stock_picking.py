#%%
import pandas as pd
import numpy as np
from os.path import join


from utils.path import DATA_PATH
from s2_substrategy import MACD_timing, RSI_timing, Bolling_timing

# choose substrategy that has PBO < 0.2
df_pbo_res = pd.read_csv(join(DATA_PATH, 'pbo_result.csv'), index_col=0)
df_selected = df_pbo_res[df_pbo_res['pbo'] < 0.2]
df_price = pd.read_csv(join(DATA_PATH, 'price_cleaned.csv'), index_col=0, parse_dates=True)


res_list = []
for row in df_selected.iterrows():
    name = row[0]
    strat = row[1].loc['best_strat'].split('-')
    strat = [strat[0]] + [int(x) for x in strat[1:]]

    df_single_price = df_price[name].to_frame()
    df_single_price['benchmark_ret'] = df_single_price[name].pct_change()
    df_single_price.dropna(inplace=True)

    if strat[0] == 'MACD':
        pars = {'fastperiod': [strat[1]], 'slowperiod': [strat[2]], 'signalperiod': strat[3]}
        res = MACD_timing(df_single_price, name, pars, verbose=True, long_only=True)
    elif strat[0] == 'RSI':
        pars = {'timeperiod': [strat[1]], 'buy_threshold': [strat[2]], 'sell_threshold': [strat[3]]}
        res = RSI_timing(df_single_price, name, pars, verbose=True, long_only=True)
    elif strat[0] == 'BBANDS':
        pars = {'timeperiod': [strat[1]], 'nbdev': [strat[2]]}
        res = Bolling_timing(df_single_price, name, pars, verbose=True, long_only=True)

    res = res['position'].to_frame()
    res.columns = [name]
    res_list.append(res)

# construct a simple long-only equal-weighted portfolio
df_res = pd.concat(res_list, axis=1)
df_res = df_res.where(df_res > 0, 0)

df_res = df_res.div(df_res.sum(axis=1), axis=0)
df_res.dropna(inplace=True)
df_res.to_csv(join(DATA_PATH, 'weight_table.csv'))
df_ret = df_price.loc[df_res.index, df_res.columns].pct_change()

df_port_res = (df_res * df_ret).sum(axis=1)
df_nav = (1 + df_port_res).cumprod()
df_nav.plot()

#%%
RF = 0
pv = df_nav.copy()
report = pd.DataFrame(None, index=[0])
start = pd.to_datetime(pv.index[0])
end = pd.to_datetime(pv.index[-1])
period = (end - start).days
report['Total Return'] = pv.iloc[-1] - 1
report['Return p.a.'] = np.power(report['Total Return'] + 1, 365. / period) - 1
daily_return = pv.pct_change().dropna()
daily_return = daily_return[daily_return != 0]
report['Volatility'] = daily_return.std() * np.sqrt(252)
report['Sharpe Ratio'] = (report['Return p.a.'] - RF) / report['Volatility']
report['Max Drawdown'] = (pv.div(pv.cummax()) - 1.).min()
report['Calmar Ratio'] = report['Return p.a.'] / abs(report['Max Drawdown'])
print(report.to_markdown())
# %%
