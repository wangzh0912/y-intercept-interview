#%%
import pandas as pd
from dateutil.relativedelta import relativedelta

from os.path import join
from utils.path import DATA_PATH


def filter_missing_value(df: pd.DataFrame, val: float) -> list:
    nan_pct = df.isna().sum() / len(df)
    return nan_pct[nan_pct < val].index.to_list()


def main():

    df = pd.read_csv(join(DATA_PATH, 'data.csv'), index_col=0)
    df.date = pd.to_datetime(df.date)

    # 1. Change the data to time-series type

    df_price = pd.pivot_table(df, values='last', index='date', columns='ticker')
    df_volume = pd.pivot_table(df, values='volume', index='date', columns='ticker')

    # 2. Drop assets that have more than 40% total val which are missing
    # drop assets that have more than 20% val in recent one year which are missing

    total_no_missing = set(filter_missing_value(df_price, 0.4))
    pre_year = df_price.index[-1] - relativedelta(years=1)
    one_year_no_missing = set(filter_missing_value(df_price[df_price.index >= pre_year], 0.2))

    no_missing = total_no_missing & one_year_no_missing

    df_price1 = df_price[no_missing].sort_index(axis=1)
    df_volume1 = df_volume[df_price1.columns]

    df_price1.to_csv(join(DATA_PATH, 'price_cleaned.csv'))
    df_price1.to_csv(join(DATA_PATH, 'volume_cleaned.csv'))


if __name__ == '__main__':
    main()
# %%
