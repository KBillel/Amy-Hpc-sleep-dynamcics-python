import os
import pandas as pd

def save_dataframe(df,name):
    df.to_csv(f'Analysis/{name}.csv')

def load_dataframe(name):
    print(f'Analysis/{name}.csv')
    if os.path.exists(f'Analysis/{name}.csv'):
        return pd.read_csv(f'Analysis/{name}.csv')
    else:
        print('Could not find the requested analysis file')
        return False
        