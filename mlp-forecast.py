from numpy.core.numeric import full
from tqdm.std import tqdm
from utils import iterChunks
import more_itertools as mit
import streamlit as st
import numpy as np
import pandas as pd
import plotly_express as px
from sklearn.model_selection import train_test_split
from tensorflow import keras

from experiments import loadDataset
from datasets import spxLoader

model = keras.models.load_model('models/mlp')
train, test = loadDataset(spxLoader.datasetName)

def mlpForecast(history):
    historySize = model.input_shape[1]
    return model.predict(history.tail(historySize).trainValue.values[None,])

# @st.cache()
# def forecast():
#     fullseries = train.copy()[['value', 'time']]
#     fullseries['trainValue'] = fullseries['value']
#     fullseries['testValue'] = None
#     forecastHorizon = 1
#     hyperparams = {}
#     predict = mlpForecast
#     chunks = list(iterChunks(test, forecastHorizon))
#     for i, chunk in enumerate(tqdm(chunks)):
#         if len(fullseries) > len(train) + i*forecastHorizon: continue
#         ypred = predict(fullseries, **hyperparams)
#         newData = chunk.copy()[['testValue', 'time']]
#         newData['trainValue'] = newData.testValue
#         newData['prediction'] = ypred[:len(chunk)]
#         fullseries = fullseries.append(newData)
#     return fullseries

# fullseries = forecast()
# fullseries

# newData = fullseries.tail(len(test))
# newData.to_csv('mlp-predictions.csv', index=False)

st.write(
    mlpForecast(train.rename(columns={'value':'trainValue'}))
)
