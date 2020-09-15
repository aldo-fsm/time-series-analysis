import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
import calendar
from utils.parsers import decomposeDate

selectedDataset = st.selectbox('Dataset', DATASET_NAMES)

@st.cache()
def load(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    return loader, train, test

loader, train, test = load(selectedDataset)
train = train.rename(columns={ loader.valueColumn: 'value' })
test = test.rename(columns={ loader.valueColumn: 'value' })

train['time'] = train.index

st.write('train:', train.shape)
st.write('test:', test.shape)

train = decomposeDate(train)

st.write(
    train.head(100)
)

initialWindow = int(len(train)*0.01)
window = st.number_input('Window', value=initialWindow, min_value=1, max_value=len(train), step=1)
rolling = train.value.rolling(window)
train['rolling_mean'] = rolling.mean()
train['rolling_std'] = rolling.std()

st.write(
    px.line(train, x=train.index, y=[train.value, train.rolling_mean, train.rolling_std])
)

st.write(
    px.box(
        x=train.month.apply(lambda month: calendar.month_name[month]),
        y=train.value,
        labels=dict(x='Mês')
    ),
    px.box(
        x=train.weekday.apply(lambda weekday: calendar.day_name[weekday]),
        y=train.value,
        labels=dict(x='Dia da semana')
    ),
    px.box(
        x=train.day,
        y=train.value,
        labels=dict(x='Dia do mês')
    ),
    px.box(
        x=train.hour,
        y=train.value,
        labels=dict(x='Horário')
    )
)
