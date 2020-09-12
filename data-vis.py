import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame

selectedDataset = st.selectbox('Dataset', DATASET_NAMES)

@st.cache()
def load(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    return loader, train, test

loader, train, test = load(selectedDataset)
train = train.rename(columns={ loader.valueColumn: 'value' })
test = test.rename(columns={ loader.valueColumn: 'value' })

st.write('train:', train.shape)
st.write('test:', test.shape)

train

st.write(
    px.line(x=train.index, y=train.value),
)
