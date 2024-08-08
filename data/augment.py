
import random
import numpy as np
import torch

class NoiseTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        """
        Adding random Gaussian noise with mean 0
        """
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)
        return X + noise

class SubAnomaly(object):
    def __init__(self, portion_len):
        self.portion_len = portion_len

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int= None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None
                                 ):
        """
        Injects an anomaly into a multivariate time series window by manipulating a
        subsequence of the window.

        :param window: The multivariate time series window represented as a 2D tensor.
        :param subsequence_length: The length of the subsequence to manipulate. If None,
                                   the length is chosen randomly between 20% and 90% of
                                   the window length.
        :param compression_factor: The factor by which to compress the subsequence.
                                   If None, the compression factor is randomly chosen
                                   between 2 and 5.
        :param scale_factor: The factor by which to scale the subsequence. If None,
                             the scale factor is chosen randomly between 0.1 and 2.0
                             for each feature in the multivariate series.
        :return: The modified window with the anomaly injected.
        """

        # Clone the input tensor to avoid modifying the original data
        window = window.copy()

        # Set the subsequence_length if not provided
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.2)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)

        # Set the compression_factor if not provided
        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)

        # Set the scale_factor if not provided
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
            print('test')

        # Randomly select the start index for the subsequence
        if start_index is None:
            start_index = np.random.randint(0, len(window) - subsequence_length)
        end_index = min(start_index + subsequence_length, window.shape[0])

        if trend_end:
            end_index = window.shape[0]

        # Extract the subsequence from the window
        anomalous_subsequence = window[start_index:end_index]

        # Concatenate the subsequence by the compression factor, and then subsample to compress it
        anomalous_subsequence = np.repeat(anomalous_subsequence, compression_factor, axis=0) #torch.cat([anomalous_subsequence] * compression_factor, dim=0)
        anomalous_subsequence = anomalous_subsequence[::compression_factor]

        # Scale the subsequence and replace the original subsequence with the anomalous subsequence
        anomalous_subsequence = anomalous_subsequence * scale_factor

        # Trend
        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = 1
        if np.random.uniform() < 0.5: coef = -1
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            anomalous_subsequence = window[start_index] + (np.random.rand(len(anomalous_subsequence)) * 0.1).reshape(-1,
                                                                                                                     1)
        window[start_index:end_index] = anomalous_subsequence

        return np.squeeze(window)

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
        """
        window = X.copy()
        anomaly_seasonal = np.zeros_like(window)
        anomaly_trend = np.zeros_like(window)
        anomaly_global = np.zeros_like(window)
        anomaly_contextual = np.zeros_like(window)
        anomaly_shapelet = np.zeros_like(window)
        if (window.ndim > 1):
            num_features = window.shape[1]
            min_len = int(window.shape[0] * 0.2)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)
            start_index = np.random.randint(0, len(window) - subsequence_length)
            num_dims = np.random.randint(1, int(num_features/5))
            for k in range(num_dims):
                i = np.random.randint(0, num_features)

                temp_win = window[:, i].reshape((window.shape[0], 1))
                anomaly_seasonal[:, i] = self.inject_frequency_anomaly(temp_win,
                                                              scale_factor=1,
                                                              trend_factor=0,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                anomaly_trend[:, i] = self.inject_frequency_anomaly(temp_win,
                                                             compression_factor=1,
                                                             scale_factor=1,
                                                             trend_end=True,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                anomaly_global[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=3,
                                                            compression_factor=1,
                                                            scale_factor=5,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_contextual[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=5,
                                                            compression_factor=1,
                                                            scale_factor=2,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_shapelet[:, i] = self.inject_frequency_anomaly(temp_win,
                                                          compression_factor=1,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          shapelet_factor=True,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

        else:
            temp_win = window.reshape((len(window), 1))
            anomaly_seasonal = self.inject_frequency_anomaly(temp_win,
                                                          scale_factor=1,
                                                          trend_factor=0)

            anomaly_trend = self.inject_frequency_anomaly(temp_win,
                                                         compression_factor=1,
                                                         scale_factor=1,
                                                         trend_end=True)

            anomaly_global = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=3,
                                                        compression_factor=1,
                                                        scale_factor=5,
                                                        trend_factor=0)

            anomaly_contextual = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=5,
                                                        compression_factor=1,
                                                        scale_factor=2,
                                                        trend_factor=0)

            anomaly_shapelet = self.inject_frequency_anomaly(temp_win,
                                                      compression_factor=1,
                                                      scale_factor=1,
                                                      trend_factor=0,
                                                      shapelet_factor=True)

        anomalies = [anomaly_seasonal,
                     anomaly_trend,
                     anomaly_global,
                     anomaly_contextual,
                     anomaly_shapelet
                     ]

        anomalous_window = random.choice(anomalies)

        return anomalous_window

class PointAnomaly(object):
    def __call__(self, X):
        """
        Adding point anomaly
        """
        x_len = X.shape[0]
        x_dim = X.shape[1]
        ts_tmp = X.copy()

        nume_dims = np.random.randint(1, 5)
        for k in range(nume_dims):
            p_j = random.randint(0, x_dim - 1)
            num_spikes = np.random.randint(10, 40)
            for s in range(num_spikes):
                p_i = random.randint(1, x_len-1)
                factor = np.random.randint(2, 5)  #the larger, the outliers are farther from inliers
                maximum, minimum = max(X[:, p_j]), min(X[:, p_j])
                local_std = ts_tmp[:, p_j].std()
                coef = np.random.choice([-1,1])
                ts_tmp[p_i:p_i+1, p_j] = ts_tmp[p_i:p_i+1, p_j] * coef * factor
                if 0 <= ts_tmp[p_i, p_j] < maximum: ts_tmp[p_i::p_i+1, p_j] = 2 * maximum
                if 0 > ts_tmp[p_i, p_j] > minimum: ts_tmp[p_i:p_i+1, p_j] = 2 * minimum

        return ts_tmp


class SubAnomaly1(object):
    def __init__(self, portion_len):
        self.portion_len = portion_len

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
        """
        window = X.copy()
        num_features = window.shape[1]
        x_len = window.shape[0]
        s_len = int(x_len * self.portion_len)
        change_probability = 0.1  # np.random.random()
        for i in range(num_features):
            if np.random.uniform() < change_probability:
                p_i = np.random.randint(0, x_len)
                interval_len = np.random.randint(1, s_len)
                start, end = p_i, p_i + interval_len
                if end > x_len: end = x_len
                interval_len = end - start
                local_std = window[:, i].std()
                factor = random.randint(2, 5)
                frequency = random.randint(1, 10)
                if local_std == 0: local_std = 1
                ls = np.array(list(range(0,x_len)))
                sinusoid = factor * np.sin(frequency * ls)
                window[start:end, i] = sinusoid[start:end]
        return window

class SubAnomaly2(object):
    def __call__(self, X):
        """
        Adding anomaly to whole window
        """
        window = X.copy()
        num_features = window.shape[1]
        i = np.random.randint(0, num_features)
        w_max = max(abs(max(window[:, i])), abs(min(window[:,i])))
        if w_max==0: w_max=1
        factor = np.random.randint(2, 5)
        return np.zeros(X.shape) + (np.random.choice([-1,1]) * factor * w_max)

class ScaleTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        """
        Scaling by a random factor
        """
        scaling_factor = np.random.normal(loc=1.0, scale=self.sigma, size=(X.shape[0], 1, X.shape[2]))
        return X * scaling_factor

class Crop(object):
    def __init__(self, wsize):
        self.wsize = wsize

    def __call__(self, X):
        """Crop the given ts.
        Args:
            j: Left pixel coordinate.
            w: Width of the cropped."""
        j = np.random.randint(0, X.shape[0] - self.wsize)
        return X[j: j + self.wsize]




