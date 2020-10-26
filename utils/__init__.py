import pandas as pd
import numpy as np

def iterChunks(data, chunkSize):
    return (data[pos:pos + chunkSize] for pos in range(0, len(data), chunkSize))

def formatDict(data: dict):
    '; '.join(
        f'{key}={value}' for key, value in data.items()
    )

def fourierTransform(values) -> pd.DataFrame:
    size = len(values)
    fourier = np.fft.rfft(values)
    freqs = np.fft.fftfreq(size)[:int(1+size/2)]
    return pd.DataFrame(dict(
        frequency=freqs,
        period=1/freqs,
        magnitude=np.abs(fourier)
    ))
