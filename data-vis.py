import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
import calendar
from utils.parsers import decomposeDate
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from transforms import LogTransformer
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot as plt
plt.style.use('seaborn')

selectedDataset = st.selectbox('Selecione a dataset', DATASET_NAMES)

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
    train['boxcox'] = boxcox.fit_transform(train.value[:, None] - dataset.value.min() + 1E-10)
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

@st.cache()
def fourierTransform(values):
    size = len(values)
    fourier = np.fft.rfft(values)
    freqs = np.fft.fftfreq(size)[:int(1+size/2)]
    return pd.DataFrame(dict(
        frequency=freqs,
        period=1/freqs,
        magnitude=np.abs(fourier)
    ))

@st.cache()
def decompose(dataset, period):
    # freq = pd.infer_freq(dataset.tail(100).index)
    # print(freq)
    # dataset = dataset.asfreq(freq).interpolate()
    decomposition = sm.tsa.seasonal_decompose(dataset.value, model='additive', period=period)
    return decomposition

loader, train, test = load(selectedDataset)
train = train.rename(columns={ loader.valueColumn: 'value' })
test = test.rename(columns={ loader.valueColumn: 'value' })

train['time'] = train.index

train = decomposeDate(train)

st.write("Dataset selecionanda")
st.write(train.head(100))

# ADF: Hipótese nula é que não é estacionária
# KPSS: Hipótese nula é que é estacionária
st.write("Análise de estacionariedade")
st.write(testStationarity(train.value))

initialWindow = int(len(train)*0.01)
window = st.number_input('Janela de tempo', value=initialWindow, min_value=1, max_value=len(train), step=1)
rolling = train.value.rolling(window)
train['rolling_mean'] = rolling.mean()
train['rolling_std'] = rolling.std()

train = applyTransforms(train)

st.write("Análise")
st.write(
    px.line(train, x=train.index, y=[train.value.astype(float), train.rolling_mean, train.rolling_std]),
)

fftResult = fourierTransform(train.value)
st.write("Transformada de Fourier")
st.write(
    px.line(fftResult, x='frequency', y='magnitude'),
    px.line(fftResult, x='period', y='magnitude'),
)

pivot = train.pivot_table(index='year',columns='month',values='value')

initialPeriod = int(len(train)*0.05)
period = st.number_input('Periodo para sazonalidade', value=initialPeriod, min_value=1, max_value=len(train), step=1)

decomposition = decompose(train, period)

decomposed = pd.DataFrame(dict(
    value=decomposition.observed,
    trend=decomposition.trend,
    resid=decomposition.resid,
    seasonal=decomposition.seasonal
))
st.write(
    decomposition.plot(),
    # px.line(decomposed, x=decomposed.index, y=['trend', 'resid', 'seasonal']),
    plot_acf(train.value, lags=100),
    plot_pacf(train.value, lags=100),
)

st.write(
    px.imshow(pivot, y=pivot.index),
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

st.write("Transformações")
st.write(
    px.line(train, x=train.index, y=[train.log, train.yeojohnson, train.boxcox]),
)
