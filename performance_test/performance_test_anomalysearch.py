""" Базовые тесты производительности. """

from unittest import TestCase
from parameterized import parameterized
import numpy as np

from anomalysearch import AnomalySearch


class TestAnomalySearchPerformance(TestCase):
    """ Базовые тесты производительности. """

    @parameterized.expand([(size, use_gpu) for size in (2, 5, 10, 50) for use_gpu in (True, False)])
    def test_compute_x(self, size: int, use_gpu: bool):
        np.random.seed(10)
        A_x_y_td_ts = np.random.rand(size, size, size, size)
        n_td = np.random.rand(size)
        anomaly_search = AnomalySearch(try_use_gpu=use_gpu)
        X_x_y_ts = anomaly_search.compute_X(A_x_y_td_ts, n_td)
        new_n = np.tensordot(A_x_y_td_ts, X_x_y_ts, axes=((0, 1, 3), (0, 1, 2)))
        np.testing.assert_allclose(n_td, new_n, atol=1e-3,
                                   err_msg=f'Error in equation solution {size=} {use_gpu=}',
                                   verbose=True)
