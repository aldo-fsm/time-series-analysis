import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
import calendar
from utils.parsers import decomposeDate
from statsmodels.tsa.stattools import adfuller, kpss
from transforms import LogTransformer
from sklearn.preprocessing import PowerTransformer

selectedDataset = st.selectbox('Dataset', DATASET_NAMES)

@st.cache()
def load(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    return loader, train, test

# def returnTransform(series):
#     pastSeries = np.pad(series, 1)[:len(series)]
#     return np.log(series/pastSeries)

@st.cache()
def applyTransforms(dataset):
    logTransformer = LogTransformer()
    yeojohnson = PowerTransformer(method='yeo-johnson')
    boxcox = PowerTransformer(method='box-cox')
    train['log'] = logTransformer.fit_transform(train.value[:, None])
    train['yeojohnson'] = yeojohnson.fit_transform(train.value[:, None])
    train['boxcox'] = boxcox.fit_transform(train.value[:, None])
    # train['lnreturn'] = returnTransform(train.value)
    return train

@st.cache()
def testStationarity(series):
    adfStatistic, adfPvalue, _, _, adfCritical, _ = adfuller(series, autolag='AIC')
    kpssStatistic, kpssPvalue, _, kpssCritical = kpss(series)
    print('adf', adfPvalue)
    print('kpss', kpssPvalue)
    return pd.DataFrame([
        dict(name='ADF', pvalue=adfPvalue, statistic=adfStatistic, **adfCritical),
        dict(name='KPSS', pvalue=kpssPvalue, statistic=kpssStatistic, **kpssCritical),
    ])

loader, train, test = load(selectedDataset)
train = train.rename(columns={ loader.valueColumn: 'value' })
test = test.rename(columns={ loader.valueColumn: 'value' })

st.write('train:', train.shape)
st.write('test:', test.shape)

train['time'] = train.index

train = decomposeDate(train)

st.write(
    train.head(100),
    testStationarity(train.value)
)

initialWindow = int(len(train)*0.01)
window = st.number_input('Window', value=initialWindow, min_value=1, max_value=len(train), step=1)
rolling = train.value.rolling(window)
train['rolling_mean'] = rolling.mean()
train['rolling_std'] = rolling.std()

train = applyTransforms(train)

st.write(
    px.line(train, x=train.index, y=[train.value.astype(float), train.rolling_mean, train.rolling_std]),
    px.line(train, x=train.index, y=[train.log, train.yeojohnson, train.boxcox]),
)

st.write(
    px.violin(
        train,
        x=train.value,
        orientation='h',
        box=True,
    ),
    px.violin(
        train,
        x=train.log,
        orientation='h',
        box=True,
    ),
    px.violin(
        train,
        x=train.yeojohnson,
        orientation='h',
        box=True,
    ),
    px.violin(
        train,
        x=train.boxcox,
        orientation='h',
        box=True,
    ),
    px.box(
        x=train.month.apply(lambda month: calendar.month_name[month]),
        y=train.value,
        labels=dict(x='Mês')
    ),
    px.box(
        x=train.weekday.apply(lambda weekday: calendar.day_name[weekday]),
        y=train.value,
        labels=dict(x='Dia da semana'),
    ),
    px.box(
        x=train.day,
        y=train.value,
        labels=dict(x='Dia do mês'),
    ),
    px.box(
        x=train.hour,
        y=train.value,
        labels=dict(x='Horário')
    )
)
