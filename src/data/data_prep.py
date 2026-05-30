import numpy as np
import pandas as pd
import os

#train_data = pd.read_csv('./data/raw/train_data.csv')
#test_data = pd.read_csv('./data/raw/test_data.csv')
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}:{e}')

def fill_missing_values_with_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_values=df[column].mean()
                df[column]=df[column].fillna(mean_values)
        return df
    except Exception as e:
        raise Exception(f'Error filling the missing : {e}')

def save_data(df:pd.DataFrame, filepath:str)->None:
    try:
        df.to_csv(filepath)
    except Exception as e:
        raise Exception(f'Error occured saving to {filepath}:{e}')

def main():
    try:
        raw_data_path = './data/raw'
        processed_data_path = './data/processed'

        os.makedirs(processed_data_path)

        train_data = load_data(os.path.join(raw_data_path, 'train_data.csv'))
        test_data = load_data(os.path.join(raw_data_path, 'test_data.csv'))

        processed_train = fill_missing_values_with_mean(train_data)
        processed_test = fill_missing_values_with_mean(test_data)

        save_data(processed_train, os.path.join(processed_data_path, 'train_processed_mean.csv'))
        save_data(processed_test, os.path.join(processed_data_path, 'test_processed_mean.csv'))
    except Exception as e:
        raise Exception(f'Error occured :{e}')
if __name__=="__main__":
    main()

#processed_train_data = fill_missing_with_median(train_data)
#processed_test_data = fill_missing_with_median(test_data)

#data_path = os.path.join('data','processed')

#os.makedirs(data_path)

#processed_train_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
#processed_test_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)