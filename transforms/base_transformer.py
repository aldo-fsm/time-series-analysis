class BaseTransformer:
    def fit(self, X):
        pass

    def transform(self, X):
        raise NotImplementedError()

    def inverse_transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

