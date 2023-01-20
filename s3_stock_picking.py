#%%
import pandas as pd
from os.path import join

from utils.path import DATA_PATH
from s2_substrategy import MACD_timing, RSI_timing, Bolling_timing

df_pbo_res = pd.read_csv(join(DATA_PATH, 'pbo_result.csv'), index_col=0)
df_selected = df_pbo_res[df_pbo_res['pbo'] > 0.75]

for row in df_selected.iterrows():
    print(row[1])
# %%
