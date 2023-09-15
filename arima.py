import numpy as np

class ARModel:
    def __init__(self, order):
        self.order = order
        self.phi = np.random.randn(order)  # Autoregressive coefficients
        self.intercept = np.random.randn()  # Intercept term (c)

    def fit(self, data):
        n = len(data)
        predictions = np.zeros_like(data, dtype=float)

        for i in range(self.order, n):
            # Calculate the autoregressive component of the prediction
            ar_component = np.sum(self.phi * data[i - self.order:i])

            # Calculate the overall prediction
            predictions[i] = self.intercept + ar_component

        return predictions


class MAModel:
    def __init__(self, order):
        self.order = order
        self.theta = np.random.randn(order)  # Moving average coefficients
        self.intercept = np.random.randn()  # Intercept term (c)

    def fit(self, data):
        n = len(data)
        predictions = np.zeros_like(data, dtype=float)

        for i in range(self.order, n):
            # Calculate the moving average component of the prediction
            ma_component = np.sum(self.theta * data[i - self.order:i])

            # Calculate the overall prediction
            predictions[i] = self.intercept + ma_component

        return predictions

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
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.diff_order = diff_order
        self.ar_model = ARModel(ar_order)
        self.ma_model = MAModel(ma_order)

    def difference(self, data):
        diff_data = np.diff(data, self.diff_order)
        return diff_data

    def integrate(self, data, original_data):
        for _ in range(self.diff_order):
            data = np.cumsum(data)
            data = np.insert(data, 0, original_data[:self.diff_order])
        return data

    def fit(self, data):
        original_data = data.copy()  # Make a copy of the original data
        for _ in range(self.diff_order):
            data = self.difference(data)
        data = self.ar_model.fit(data)
        data = self.ma_model.fit(data)
        data = self.integrate(data, original_data)
        return data
