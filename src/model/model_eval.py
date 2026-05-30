import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
import json

#test_data = pd.read_csv('./data/processed/test_processed.csv')
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}:{e}')

#X_test = test_data.drop(columns=['Potability'])
#y_test = test_data['Potability']
def data_prep(data:pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    try:
        X=data.drop(columns=['Potability'])
        y=data["Potability"]
        return X,y
    except Exception as e:
        raise Exception(f'Error preparing data : {e}')

#model = pickle.load(open('model.pkl','rb'))
def load_model(filepath:str):
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f'Error loading model : {e}')

def model_eval(model, X_test:pd.DataFrame, y_test:pd.Series)-> dict:
    try:
        y_pred = model.predict(X_test)

        acc=accuracy_score(y_test, y_pred)
        precision=precision_score(y_test, y_pred)
        recall=recall_score(y_test, y_pred)
        f1score = f1_score(y_test,y_pred)

        metrics_dict ={
            'acc':acc,
            'precision':precision,
            'recall':recall,
            'f1score':f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f'Error evaluating model : {e}')

def save_metrics(metrics_dict:dict, filepath:str)-> None:
    try:
        with open(filepath, 'w')as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f'Error saving metrics : {e}')

def main():
    try:
        data_path = './data/processed/test_processed_mean.csv'
        model_path='models/model.pkl'
        metrics_path='reports/metrics.json'

        test_data=load_data(data_path)
        X_test,y_test=data_prep(test_data)
        model=load_model(model_path)
        metrics = model_eval(model, X_test, y_test)

        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f'Error occured')
    
if __name__=="__main__":
    main()


#y_pred = model.predict(X_test)

#acc = accuracy_score(y_test,y_pred)
#precision=precision_score(y_test,y_pred)
#recall=recall_score(y_test,y_pred)
#f1score=f1_score(y_test,y_pred)

#metric_dict = {
#    'acc': acc,
#    'precision':precision,
#    'recall':recall,
#    'f1score':f1score
#}

#os.makedirs('metrics', exist_ok=True)

#with open('metrics.json','w')as file:
#    json.dump(metric_dict, file, indent=4)