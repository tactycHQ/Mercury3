import requests
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#GLOBAL VARIABLES
DEV_HOST = 'http://localhost:8501/v1/models/mercury:predict'
PROD_HOST = 'http://tf-serving-server:8501/v1/models/mercury:predict'
data_path = ".//test_data//predict_from_server.csv"

def predict_results(df):
    toPredict = df.values.tolist()[0]
    print(toPredict)
    payload={
        "instances":[{'dense_input':toPredict}]
    }
    response = requests.post(DEV_HOST,json = payload)
    predictions = response.json()['predictions'][0]
    print(predictions)
    return predictions

def load_data(data_path):
    df = pd.read_csv(data_path, low_memory=False, skip_blank_lines=True, header=None)
    return df

def normalize_data(df):
    sc = StandardScaler()
    sc.fit(df.values)
    x_pred = sc.transform(df.values)
    return x_pred

if __name__ == '__main__':
    test = np.loadtxt(".\\test_data\\predict_from_server.csv", delimiter=",")
    test_list = test.tolist()
    print(test_list)
    df = load_data(data_path)
    predictions = predict_results(df)


