import requests
import json
import numpy as np

toPredict = np.loadtxt(".\\test_data\\predict_from_server.csv",delimiter=",")
toPredict_list = toPredict.tolist()
payload={
    "instances":[{'dense_input':toPredict_list}]
}
response = requests.post('http://localhost:8501/v1/models/mercury:predict',json = payload)
predictions = response.json()['predictions'][0]
print(predictions)




