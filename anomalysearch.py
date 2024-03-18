""" Поиск источников аномальных выбросов. """

from dataclasses import dataclass

import torch
import numpy as np
from math import prod


@dataclass
class StaticSourceInfo:
    """Информация о статически расположенном источнике."""
    predicted_emission_x_y: np.ndarray
    """Матрица значений предполагаемых выбросов."""
    max_emission_indices: (int, int)
    """Индексы ячейки с максимальным значением предполагаемых выбросов."""


class AnomalySearch:
    """
    Класс, выполняющий поиск аномалий.
    """

    def __init__(self, *, try_use_gpu: bool = True):
        """
        Конструктор
        :param try_use_gpu: Использовать ли GPU-ускоритель при расчётах.
        """
        self._use_gpu = try_use_gpu and torch.cuda.is_available()
        self._device = 'cuda' if self._use_gpu else 'cpu'

    def compute_X(self, A_x_y_td_ts: np.ndarray, n_td: np.ndarray) -> np.ndarray:
        """
        Вычислить тензор X в СЛАУ `A_x_y_td_ts dot X_x_y_ts = n_td`.
        Если система уравнений недоопределена, возвращается одно из решений.
        :param A_x_y_td_ts: тензор переноса - количество частиц, испущенных из точки расположения датчика в момент (ts)
               и зафиксированных в точке (x,y) в момент (td).
        :param n_td: вектор количеств зарегистрированных датчиком частиц в момент (ts).
        :return: тензор X_x_y_ts - прогноз количеств испущенных частиц в ячейках (x,y) в момент (ts).
        """
        # Транспонируем x y td ts -> td x y ts
        A_td_x_y_ts = np.transpose(A_x_y_td_ts, (2, 0, 1, 3))
        Xshape = A_td_x_y_ts.shape[1:]
        A_td_xyts = A_td_x_y_ts.reshape((A_td_x_y_ts.shape[0], prod(A_td_x_y_ts.shape[1:])))
        A_td_xyts = torch.Tensor(A_td_xyts).to(self._device)
        n_td = torch.Tensor(n_td).to(self._device)

        res = torch.linalg.lstsq(A_td_xyts, n_td)
        (solution, residuals, rank, singular_values) = res

        X_xyts = solution.cpu().numpy()
        X_x_y_ts = np.reshape(X_xyts, Xshape)
        return X_x_y_ts

    def get_static_source(self, X_x_y_ts: np.ndarray) -> StaticSourceInfo:
        """
        Получить информацию о статически расположенном источнике по тензору выбросов X_x_y_ts.
        :param X_x_y_ts: прогноз количеств испущенных частиц в ячейках (x,y) в момент (ts).
        :return: информацию о статически расположенном источнике (предполагаемые координаты).
        """
        X_x_y = np.sum(X_x_y_ts, axis=2)
        ij = np.argmax(X_x_y)
        (i, j) = np.unravel_index(ij, X_x_y.shape)
        return StaticSourceInfo(X_x_y, (i, j))
