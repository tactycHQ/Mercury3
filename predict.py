import os
import numpy as np
import pandas as pd
from data_loader.data_loader import DataLoader
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#GLOBAL VARIABLES
predict_path = "D:\\Dropbox\\9. Data\\Mercury Data\\XLS\\predict\\CIQ_AAPL_predict.csv"
model_path = "C:\\Users\\anubhav\\Desktop\\Projects\\Mercury2\\saved_models\\run17.h5"
window=45
threshold=0.035

def load_data(predict_path):
    df = pd.read_csv(predict_path,low_memory=False)
    df = df.drop(['DATE'],axis=1)
    return df

def process_data(df):
    prices = df['IQ_LASTSALEPRICE'].values.reshape(-1, 1)
    bmark = df['BENCHMARK'].values.reshape(-1, 1)

    len = prices.shape[0]
    priceReturns = np.empty((len, 1))
    bmarkReturns = np.empty((len, 1))
    for i in range (0,len-window):
        priceReturns[i] = prices[i+window,0]/prices[i,0]-1
        bmarkReturns[i] = bmark[i+window, 0]/bmark[i, 0] - 1

    priceReturns=priceReturns[:-window]
    bmarkReturns=bmarkReturns[:-window]
    relReturns = priceReturns - bmarkReturns
    targets = []
    for ret in relReturns:
        if ret > threshold:targets.append(1)
        elif ret < -threshold:targets.append(-1)
        else: targets.append(0)
    targets = np.array(targets).reshape(-1, 1)
    unique, counts = np.unique(targets, return_counts=True)
    print("Target counts are %s %s", unique, counts)

    ohe = OneHotEncoder(categories='auto')
    targets_ohe = ohe.fit_transform(targets).toarray()

    print("prices:\n", prices)
    print("priceReturns:\n", priceReturns)
    print("bmarkReturns:\n", bmarkReturns)
    print("relReturns:\n", relReturns)
    print("targets:\n", targets)
    print("targets_ohe:\n", targets_ohe)

    return targets_ohe

def normalize_data(df):
    sc = StandardScaler()
    sc.fit(df.values)
    x_pred = sc.transform(df.values)
    np.savetxt(".\\x_pred.csv", x_pred, delimiter=",")
    return x_pred


def predict_results(model_path):
    dense_model =load_model(model_path)
    pred_results = dense_model.predict(x_pred)
    np.savetxt(".\\test_data\\prediction_results.csv",pred_results,delimiter=",")
    print("pred:\n", pred_results)
    return pred_results

if __name__ == '__main__':
    df = load_data(predict_path)
    targets_ohe = process_data(df)
    x_pred = normalize_data(df)
    pred_results = predict_results(model_path)



