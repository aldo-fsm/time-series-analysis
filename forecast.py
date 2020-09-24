import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
from statsmodels.tsa.api import Holt, ExponentialSmoothing

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

exponentialSmoothing = ExponentialSmoothing(train.value)
exponentialSmoothing.fit()

holt = Holt(train.value)
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
    px.line(fullSeries, x=fullSeries.index, y=['value', 'expsmo', 'holt'])
)
