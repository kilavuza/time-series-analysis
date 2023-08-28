import numpy as np


class ARModel:
    def __init__(self, order):
        self.order = order
        self.coeffs = np.random.randn(order)

    def fit(self, data):
        n = len(data)
        for i in range(self.order, n):
            prediction = np.sum(self.coeffs * data[i - self.order:i])
            data[i] = prediction + np.random.randn()
        return data


class MAModel:
    def __init__(self, order):
        self.order = order
        self.coeffs = np.random.randn(order)

    def fit(self, data):
        n = len(data)
        for i in range(self.order, n):
            prediction = np.sum(self.coeffs * data[i - self.order:i])
            data[i] = prediction + np.random.randn()
        return data


class ARMAModel:
    def __init__(self, ar_order, ma_order):
        self.ar_model = ARModel(ar_order)
        self.ma_model = MAModel(ma_order)

    def fit(self, data):
        data = self.ar_model.fit(data)
        data = self.ma_model.fit(data)
        return data


class ARIMAModel:
    def __init__(self, ar_order, diff_order, ma_order):
        self.ar_model = ARModel(ar_order)
        self.ma_model = MAModel(ma_order)
        self.diff_order = diff_order

    def difference(self, data):
        diff_data = np.diff(data, self.diff_order)
        return diff_data

    def fit(self, data):
        for _ in range(self.diff_order):
            data = self.difference(data)
        data = self.ar_model.fit(data)
        data = self.ma_model.fit(data)
        for _ in range(self.diff_order):
            data = np.insert(data, 0, data[0])
            data = np.cumsum(data)
        return data