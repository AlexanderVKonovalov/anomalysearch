""" Базовые юнит-тесты. """
from unittest import TestCase

import numpy as np

from anomalysearch import AnomalySearch
from main import load_A_n


class TestAnomalySearch(TestCase):
    """ Базовые юнит-тесты. """

    def test_compute_x(self):
        A_x_y_td_ts = np.array([[[[0, 0], [0, 0], [1, 0]], [[0, 0], [0, 2], [1, 1]]],
                                [[[0, 2], [0, 0], [1, 0]], [[0, 0], [0, 2], [0, 1]]],
                                [[[0, 0], [4, 0], [7, 0]], [[0, 0], [0, 0], [1, 0]]],
                                ], dtype='float64')
        n_td = np.array([0.1, 4.0, 6.8], dtype='float64')
        anomaly_search = AnomalySearch(try_use_gpu=True)
        X_x_y_ts = anomaly_search.compute_X(A_x_y_td_ts, n_td)
        new_n = np.tensordot(A_x_y_td_ts, X_x_y_ts, axes=((0, 1, 3), (0, 1, 2)))
        np.testing.assert_allclose(n_td, new_n, atol=1e-3, err_msg='Error in equation solution', verbose=True)

    def test_compute_x_real_data(self):
        A_x_y_td_ts, n_td = load_A_n(path='../data/Anomaly')
        anomaly_search = AnomalySearch(try_use_gpu=True)
        X_x_y_ts = anomaly_search.compute_X(A_x_y_td_ts, n_td)
        new_n = np.tensordot(A_x_y_td_ts, X_x_y_ts, axes=((0, 1, 3), (0, 1, 2)))
        np.testing.assert_allclose(n_td, new_n, atol=1e-3, err_msg='Error in equation solution', verbose=True)
