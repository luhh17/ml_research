import os
import pandas as pd

def load_pred_result(path: str, suffix='_pred_result') -> pd.DataFrame:
    """
    load prediction result for each year, and combine them into a dataframe
    Parameters:
        path: str
    Returns:
        pred_df: pd.DataFrame
    """
    file_list = os.listdir(path)
    pred_file_list = [f for f in file_list if f.endswith(suffix)]
    pred_df = []
    for pred in pred_file_list:
        df = pd.read_pickle(f'{path}/{pred}')
        df = df.rename(columns={'weight': 'pred'})
        pred_df.append(df)
    if len(pred_df) == 0:
        return None
    pred_df = pd.concat(pred_df, axis=0)
    if 'ret' in pred_df.columns:
        pred_df = pred_df.drop(columns=['ret'])
    pred_df['date'] = pd.to_datetime(pred_df['date'], format='%Y%m%d')
    if 'mask' in pred_df.columns:
        pred_df = pred_df[pred_df['mask'] == True]
    pred_df = pred_df.sort_values(by=['date', 'stkcd'])
    pred_df = pred_df[['date', 'stkcd', 'pred']]
    pred_df['stkcd'] = pred_df['stkcd'].astype(str)
    pred_df['stkcd'] = pred_df['stkcd'].str.zfill(6)
    return pred_df

