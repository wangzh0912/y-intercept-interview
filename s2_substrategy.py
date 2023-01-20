#%%
import pandas as pd
import numpy as np
import talib as ta
from os.path import join
import itertools

from utils.path import DATA_PATH


def MACD_timing(df_single_price, asset, pars, verbose=False, long_only=False):
    result = pd.DataFrame()
    for fp in pars['fastperiod']:
        for sp in pars['slowperiod']:
            signalperiod = pars['signalperiod']
            name = 'MACD-' + str(fp) + '-' + str(sp) + '-' + str(signalperiod)
            df = df_single_price.copy()
            df['DIF'], df['DEA'], df['MACD'] = ta.MACD(np.array(df[asset]),
                                                        fastperiod=fp,
                                                        slowperiod=sp,
                                                        signalperiod=signalperiod)
            df['yes_MACD'] = df['MACD'].shift(1)
            df['daybeforyes_MACD'] = df['MACD'].shift(2)
            df['yes_DIF'] = df['DIF'].shift(1)
            df['yes_DEA'] = df['DEA'].shift(1)
            # Long: DIF and DEA are positive, DIF moves upwards crossing DEA
            df['position'] = np.where(
                ((df['yes_MACD'] > 0) & (df['daybeforyes_MACD'] < 0) & (
                            df['yes_DIF'] > 0) & (df['yes_DEA'] > 0)), 1,
                np.nan)
            # Short: DIF and DEA are negative, DIF moves downwards crossing DEA
            df['position'] = np.where(
                ((df['yes_MACD'] < 0) & (df['daybeforyes_MACD'] > 0) & (
                            df['yes_DIF'] < 0) & (df['yes_DEA'] < 0)), -1,
                df['position'])
            

            df['position'] = df['position'].ffill().fillna(0)

            if long_only:
                df['position'] = df['position'].where(df['position'] > 0, 0)

            df['MACD_return'] = df['position'] * df['benchmark_ret']
            result[name] = df['MACD_return']
    if verbose:
        return df
    return result


def RSI_timing(df_single_price, asset, pars, verbose=False, long_only=False):
    result = pd.DataFrame()
    for tp in pars['timeperiod']:
        for bt in pars['buy_threshold']:
            for st in pars['sell_threshold']:
                name = 'RSI-' + str(tp) + '-' + str(bt) + '-' + str(st)
                df = df_single_price.copy()
                df['RSI'] = ta.RSI(np.array(df[asset]), timeperiod=tp)
                df['yes_RSI'] = df['RSI'].shift(1)
                # Long: RSI < buy threshold
                df['position'] = np.where(df['yes_RSI'] < bt, 1, np.nan)
                # Short: RSI > sell threshold
                df['position'] = np.where(df['yes_RSI'] > st, -1,
                                          df['position'])
                df['position'] = df['position'].ffill().fillna(0)

                if long_only:
                    df['position'] = df['position'].where(df['position'] > 0, 0)


                df['RSI_return'] = df['position'] * df['benchmark_ret']
                result[name] = df['RSI_return']
    if verbose:
        return df
    return result


def Bolling_timing(df_single_price, asset, pars, verbose=False, long_only=False):
    result = pd.DataFrame()
    for tp in pars['timeperiod']:
        for nbdev in pars['nbdev']:
            name = 'BBANDS-' + str(tp) + '-' + str(nbdev)
            df = df_single_price.copy()
            df['ceiling'], df['middle'], df['floor'] = ta.BBANDS(
                np.asarray(df[asset]),
                timeperiod=tp, nbdevup=nbdev, nbdevdn=nbdev,
                matype=0)
            df['yes_close'] = df[asset].shift(1)
            df['yes_floor'] = df['floor'].shift(1)
            df['yes_ceiling'] = df['ceiling'].shift(1)

            df['daybeforeyes_close'] = df[asset].shift(2)
            df['daybeforeyes_floor'] = df['floor'].shift(2)
            df['daybeforeyes_ceiling'] = df['ceiling'].shift(2)
            # Long: close price moves upwards crossing the ceiling
            df['position'] = np.where(
                (df['daybeforeyes_close'] < df['daybeforeyes_ceiling']) & (
                            df['yes_close'] > df['yes_ceiling']), 1, np.nan)
            # Short: close price moves downwards crossing the floor
            df['position'] = np.where(
                (df['daybeforeyes_close'] > df['daybeforeyes_floor']) & (
                            df['yes_close'] < df['yes_floor']), -1,
                df['position'])
            df['position'] = df['position'].ffill().fillna(0)

            if long_only:
                df['position'] = df['position'].where(df['position'] > 0, 0)

            df['Bolling_return'] = df['position'] * df['benchmark_ret']
            result[name] = df['Bolling_return']
    if verbose:
        return df
    return result


def single_asset_strat(df_price, asset: str, strat_params: dict):
    df_single_price = df_price[asset].to_frame()
    df_single_price['benchmark_ret'] = df_single_price[asset].pct_change()
    df_single_price.dropna(inplace=True)

    # 1. MACD
    df_tmp1 = MACD_timing(df_single_price, asset, strat_params['macd'], long_only=True)
    df_tmp1 = (df_tmp1 + 1).cumprod(axis=0)
    # 2. RSI
    df_tmp2 = RSI_timing(df_single_price, asset, strat_params['rsi'], long_only=True)
    df_tmp2 = (df_tmp2 + 1).cumprod(axis=0)
    # 3. Bolling
    df_tmp3 = Bolling_timing(df_single_price, asset, strat_params['bolling'], long_only=True)
    df_tmp3 = (df_tmp3 + 1).cumprod(axis=0)

    df_nav = pd.concat((df_tmp1, df_tmp2, df_tmp3), axis=1)
    
    S = 8
    df_ret = df_nav.pct_change()
    df_ret.dropna(inplace=True)
    T, N = df_ret.shape

    T2S = int(np.ceil(T / S))
    id_chunks = [df_ret.index[i:i + T2S] for i in range(0, len(df_ret), T2S)] # sepreate dates into S chunks

    # K fold CV
    Cs = list(itertools.combinations(range(S), S // 2))
    df_results = pd.DataFrame({'w': [], 'sr_in': [], 'sr_out': [], 'best_in': []})
    for i_iter in range(len(Cs)):
        # in sample and out sample
        id_in_sample = list()
        for j_mat in Cs[i_iter]:
            id_in_sample.extend(id_chunks[j_mat])
        j_in_sample = df_ret.loc[id_in_sample, :]
        j_out_sample = df_ret.drop(axis=0, index=id_in_sample, inplace=False)
        # compute sharpe ratio
        sr_in_sample = j_in_sample.mean() * 250 / (j_in_sample.std() * np.sqrt(250))
        sr_out_sample = j_out_sample.mean() * 250 / (j_out_sample.std() * np.sqrt(250))
        # rank
        sr_in_sample.dropna(inplace=True)
        id_best = sr_in_sample.values.argmax()
        name_best = sr_in_sample.index[id_best]
        rank_best = sr_out_sample.rank(ascending=False)[id_best]
        w = rank_best / (N + 1)
        # w = 1, then the best in-sample is the worst out-sample (overfitting)

        # save results
        df_results.loc[i_iter, :] = [w, sr_in_sample[id_best], sr_out_sample[id_best], name_best]

    # compute pbo: the frequency that w is larger than 50%
    pbo = (df_results['w'] >= 0.5).sum() / len(Cs)
    # print(asset + ': PBO = %.2f among %d strategies' % (pbo, N))
    # print(asset + ': best strategy is %s' % df_results['best_in'].value_counts().index[0])
    return pd.DataFrame(index=[asset], data={'pbo': pbo, 'best_strat': df_results['best_in'].value_counts().index[0]})


def main(strat_params):

    df_price = pd.read_csv(join(DATA_PATH, 'price_cleaned.csv'), index_col=0, parse_dates=True)
    res = []
    for asset_id, asset in enumerate(df_price.columns):
        tmp = single_asset_strat(df_price, asset, strat_params)
        res.append(tmp)
        if asset_id % 20 == 0:
            print(f'Testing {asset_id}/{df_price.shape[1]} assets.')

    df_pbo_res = pd.concat(res)
    df_pbo_res.to_csv(join(DATA_PATH, 'pbo_result.csv'))


if __name__ == '__main__':
    strat_params = {
            'macd': {'fastperiod': [7, 12, 17], 'slowperiod': [22, 26, 31], 'signalperiod': 9},
            'rsi': {'timeperiod': [4, 14, 24], 'buy_threshold': [20, 30, 40], 'sell_threshold': [60, 70, 80]},
            'bolling': {'timeperiod': [10, 20, 30], 'nbdev': [1, 2, 3]},
        }
    main(strat_params)
# %%
