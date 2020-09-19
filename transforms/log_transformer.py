from . import BaseTransformer
import numpy as np

class LogTransformer(BaseTransformer):

    def transform(self, X):
        return np.log(X)

    def inverse_transform(self, X):
        return np.exp(X)
