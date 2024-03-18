""" Поиск источников аномальных выбросов - демо-программа. """
import numpy as np
import matplotlib.pyplot as plt

from anomalysearch import AnomalySearch


def load_A_n(path: str = 'data/Anomaly') -> (np.ndarray, np.ndarray):
    """
    Загрузить тестовые данные из файлов в каталоге path.
    :param path: Каталог, из которого загружатся данные.
    :return: пару (A,n) - тензоры, загруженные из {path}/A.npy и {path}/n.csv соответственно.
        А - тензор переноса.
        n - вектор количества испущенных частиц.
    """
    A = np.load(f'{path}/A.npy')
    n = np.loadtxt(f'{path}/n.csv')
    return A, n


def run():
    """ Запустить расчёт и отображение источников для тестовых данных. """
    A, n = load_A_n()
    anomaly_search = AnomalySearch()
    X = anomaly_search.compute_X(A, n)
    static_source = anomaly_search.get_static_source(X)
    fig, subplots = plt.subplots(2, X.shape[2], sharey=True)
    sensor_pos_x, sensor_pos_y = ([6], [7])
    for i in range(X.shape[2]):
        p = subplots[0, i]
        p.matshow(X[:, :, i].transpose())
        p.set_xlabel("x")
        if i == 0:
            p.set_ylabel("y")
        p.set_title(f"X[t_s={i}]")
        p.plot(sensor_pos_x, sensor_pos_y, 'x', color='orange')
    p = subplots[1, 0]
    p.matshow(static_source.predicted_emission_x_y.transpose())
    p.set_xlabel("x")
    p.set_ylabel("y")
    p.set_title(f"emission (X)")
    p.plot(sensor_pos_x, sensor_pos_y, 'x', color='orange')
    p.plot([static_source.max_emission_indices[0]], [static_source.max_emission_indices[1]], 'gx')

    for i in range(1, X.shape[2]):
        subplots[1, i].axis('off')
    plt.show()


if __name__ == '__main__':
    run()
