import pandas as pd
import numpy as np
df_list = []
df3 = pd.read_pickle('/mnt/HDD16TB/huahao/portfolio_construction/data/raw_data/index_comp_data3/IDX_Smprat.pkl')
df2 = pd.read_pickle('/mnt/HDD16TB/huahao/portfolio_construction/data/raw_data/index_comp_data2/IDX_Smprat.pkl')
df1 = pd.read_pickle('/mnt/HDD16TB/huahao/portfolio_construction/data/raw_data/index_comp_data1/IDX_Smprat.pkl')
def clean_data(df):
    df = df[(df['Enddt'] != '截止日期') & (df['Enddt'] != '没有单位')]
    df['Enddt'] = pd.to_datetime(df['Enddt'])
    return df
df3 = clean_data(df3)
df2 = clean_data(df2)
df1 = clean_data(df1)
# 'Indexcd', 'Enddt', 'Stkcd', 'Constdnme'
df = pd.concat([df3, df2, df1])
df = df[df['Indexcd'] == '000300']
df = df.sort_values(by=['Enddt'])
df['mask'] = 1
df = df.fillna(0)
grouped = df.groupby('Enddt')['mask'].sum()
print(grouped)
df.to_pickle('/mnt/HDD16TB/huahao/portfolio_construction/data/HS300_component.pkl')
date_list = df['Enddt'].unique()
stkcd_list = []
period_stkcd_list = []
chg_date_list = [date_list[0]]
for idx, date in enumerate(date_list):
    stk = df[df['Enddt'] == date]['Stkcd'].unique()
    stkcd_list.append(stk)

    if len(stkcd_list) > 1:
        a = np.intersect1d(stkcd_list[-1], stkcd_list[-2])
        if len(a) != 300:
            stkcd_a = stkcd_list[-2]
            chg_date_list.append(date_list[idx-1])
            period_stkcd_list.append(stkcd_a)

res_list = []
print(chg_date_list)
for i in range(len(chg_date_list) - 1):
    period = chg_date_list[i], chg_date_list[i+1]
    stkcd = period_stkcd_list[i]
    res_list.append([[period, stkcd]])


