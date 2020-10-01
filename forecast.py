import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
selectedDataset = st.selectbox('Dataset', DATASET_NAMES)

@st.cache()
def load(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    return loader, train, test

loader, train, test = load(selectedDataset)
train = train.rename(columns={ loader.valueColumn: 'value' })
test = test.rename(columns={ loader.valueColumn: 'testValue' })

train['time'] = train.index

st.write('train:', train.shape)
st.write('test:', test.shape)

exponentialSmoothing = SimpleExpSmoothing(train.value)
exponentialSmoothing.fit()

initialPeriod=25
period = st.number_input('Period', value=initialPeriod, min_value=1, max_value=len(train), step=1)

holt = ExponentialSmoothing(train.value, trend='add', seasonal='add', seasonal_periods=period)
holt.fit()

startIndex = len(train.value)-1
endIndex = startIndex + len(test)-1
test['expsmo'] = exponentialSmoothing.predict(exponentialSmoothing.params, start=startIndex, end=endIndex)
test['holt'] = holt.predict(holt.params, start=startIndex, end=endIndex)

# test
fullSeries = pd.concat([
    train,
    test
])
st.write(
    px.line(fullSeries, x=fullSeries.index, y=['value', 'expsmo', 'holt', 'testValue'])
)

errors = pd.DataFrame()
errors['holtSquaredError'] = (test.testValue - test.holt)**2
errors['expsmoSquaredError'] = (test.testValue - test.expsmo)**2

st.write(
    errors
)

st.write(
    pd.DataFrame({
        'MSE': [
            mean_squared_error(test.testValue.values, test.holt.values),
            mean_squared_error(test.testValue.values, test.expsmo.values)
        ],
    }, index=['Holt', 'Exponential Smoothing'])
)
initialWindow = int(len(test)*0.01)
window = st.number_input('Rolling Error Window', value=initialWindow, min_value=1, max_value=len(errors), step=1)
holtRollingError = errors.holtSquaredError.rolling(window)
expsmoRollingError = errors.expsmoSquaredError.rolling(window)
errors['holtMeanSquaredError'] = holtRollingError.mean()
errors['expsmoMeanSquaredError'] = expsmoRollingError.mean()
st.write(
    px.line(errors, y=['holtSquaredError', 'expsmoSquaredError', 'holtMeanSquaredError', 'expsmoMeanSquaredError'])
)
