import more_itertools as mit
import streamlit as st
import numpy as np
import pandas as pd
import plotly_express as px
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam

from experiments import loadDataset
from datasets import spxLoader

HISTORY_SIZE=30
FORECAST_HORIZON=1
HIDDEN_SIZES=[10]
model = Sequential(layers=[
    InputLayer(input_shape=(HISTORY_SIZE,)),
    *[Dense(hiddenLayerSize, activation=relu) for hiddenLayerSize in HIDDEN_SIZES],
    # Dense(HIDDEN_SIZES[0], input_shape=(HISTORY_SIZE,), activation=relu),
    # *[Dense(hiddenLayerSize, activation=relu) for hiddenLayerSize in HIDDEN_SIZES[1:]],
    Dense(FORECAST_HORIZON),
])

model.compile(optimizer=Adam(), loss=mean_squared_error)

st.write(
    model
)

train, test = loadDataset(spxLoader.datasetName)

X = []
y = []
for window in mit.windowed(train['value'].values, HISTORY_SIZE+FORECAST_HORIZON):
    X.append(window[:HISTORY_SIZE])
    y.append(window[HISTORY_SIZE:])

X = np.array(X)
y = np.array(y)

# Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.15)

history = model.fit(X, y, validation_split=0.15, epochs=100)
history = pd.DataFrame(history.history)
model.save('models/mlp')
st.write(
    px.line(history)
)
