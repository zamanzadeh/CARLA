
import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        """
        Adding random Gaussian noise with mean 0
        """
        if X.device.type == 'cuda':  # Check if X is on GPU
            X = X.cpu()  # Move tensor to CPU
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)  # NumPy operation
        
        return torch.tensor(X.numpy() + noise, dtype=torch.float32, device=device)  # Move back to GPU

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
        window = window.clone() #.copy()

        # Set the subsequence_length if not provided
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
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
        # anomalous_subsequence = np.tile(anomalous_subsequence, (compression_factor, 1))
        anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)  # cuda! PyTorch equivalent of np.tile()
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
            # anomalous_subsequence = window[start_index] + (np.random.rand(len(anomalous_subsequence)) * 0.1).reshape(-1, 1)
            anomalous_subsequence = window[start_index] + (torch.rand_like(window[start_index]) * 0.1)  #cuda use!

        window[start_index:end_index] = anomalous_subsequence

        return np.squeeze(window)

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
        """
        window = X.clone() #X.copy()
        anomaly_seasonal = window.clone() #.copy()
        anomaly_trend = window.clone() #.copy()
        anomaly_global = window.clone() #.copy()
        anomaly_contextual = window.clone() #.copy()
        anomaly_shapelet = window.clone() #.copy()
        min_len = int(window.shape[0] * 0.1)
        max_len = int(window.shape[0] * 0.9)
        subsequence_length = np.random.randint(min_len, max_len)
        start_index = np.random.randint(0, len(window) - subsequence_length)
        if (window.ndim > 1):
            num_features = window.shape[1]
            num_dims = np.random.randint(int(num_features/10), int(num_features/2)) #(int(num_features/5), int(num_features/2))
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
                                                            subsequence_length=2,
                                                            compression_factor=1,
                                                            scale_factor=8,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_contextual[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=4,
                                                            compression_factor=1,
                                                            scale_factor=3,
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
                                                          trend_factor=0,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

            anomaly_trend = self.inject_frequency_anomaly(temp_win,
                                                         compression_factor=1,
                                                         scale_factor=1,
                                                         trend_end=True,
                                                         subsequence_length=subsequence_length,
                                                         start_index = start_index)

            anomaly_global = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=3,
                                                        compression_factor=1,
                                                        scale_factor=8,
                                                        trend_factor=0,
                                                        start_index = start_index)

            anomaly_contextual = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=5,
                                                        compression_factor=1,
                                                        scale_factor=3,
                                                        trend_factor=0,
                                                        start_index = start_index)

            anomaly_shapelet = self.inject_frequency_anomaly(temp_win,
                                                      compression_factor=1,
                                                      scale_factor=1,
                                                      trend_factor=0,
                                                      shapelet_factor=True,
                                                      subsequence_length=subsequence_length,
                                                      start_index = start_index)

        anomalies = [anomaly_seasonal,
                     anomaly_trend,
                     anomaly_global,
                     anomaly_contextual,
                     anomaly_shapelet
                     ]

        anomalous_window = random.choice(anomalies)

        return anomalous_window







