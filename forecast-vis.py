import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from experiments import loadExperimentsAndResults, loadExperimentResults, loadDataset

selectedDataset = st.selectbox('Selecione a dataset', DATASET_NAMES)
train, test = loadDataset(selectedDataset)
experiments, predictions = loadExperimentsAndResults(selectedDataset)
fullSeries = pd.concat([train, predictions])

st.write('Experimentos')
st.write(
    experiments
)

predColumns = [col for col in fullSeries.columns if 'pred_' in col]
st.write("Predição")
st.write(
    px.line(fullSeries, x=fullSeries.index, y=['value', 'testValue', *predColumns])
)

errors = pd.DataFrame()
for predCol in predColumns:
    errors[predCol] = (predictions.testValue - predictions[predCol])**2

st.write("Erro quadrático médio (RSME)")
st.write(
    pd.DataFrame({
        'RMSE': [
            mean_squared_error(predictions.testValue.values, predictions[predCol].values, squared=False)
        for predCol in predColumns],
    }, index=predColumns)
)

st.write("Erros quadráticos")
st.write(
    errors
)

# initialWindow = int(len(test)*0.01)
# window = st.number_input('Janela de analise do erro', value=initialWindow, min_value=1, max_value=len(errors), step=1)

# st.write("Gráfico dos erros")
# holtRollingError = errors.holtSquaredError.rolling(window)
# expsmoRollingError = errors.expsmoSquaredError.rolling(window)
# errors['holtMeanSquaredError'] = holtRollingError.mean()
# errors['expsmoMeanSquaredError'] = expsmoRollingError.mean()

st.write(
    px.line(errors.astype(np.float64), y=predColumns)
)
