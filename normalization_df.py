import pickle

import numpy as np
import pandas as pd
import redis
from accessify import protected
from sklearn.cluster import DBSCAN


class Normalization_Df(object):
    def __init__(self, redis_connection: redis.client.Redis):
        """
        Класс для нормализации входных данных
        Args:
            redis_connection: Подключение к хранилищу Redis
        """

        self.redis_connection = redis_connection
        self.offset = None
        self.std = None

    def offset_std_calculation(self, df: pd.DataFrame, mean_or_median: bool):
        """
        Подчсет смещения данных, то есть среднего (медианы) по каждому столбцу
        Args:
            df: Датафрейм с тестовыми/тренировочными данными
            mean_or_median: Если True, то подсчет Среднего по каждому столбцу, иначе Медианы

        Returns:
            Без возвращаемого значения
        """

        if mean_or_median:
            self.offset = np.mean(df, axis=0)
        else:
            self.offset = np.median(df, axis=0)
        self.std = np.std(df, axis=0)

        try:
            self.redis_connection.set('offset', pickle.dumps(self.offset))
            self.redis_connection.set('std', pickle.dumps(self.std))
        except redis.ConnectionError:
            # TODO ЗАМЕНИТЬ НА ЛОГИРОВАНИЕ
            print('Connection Redis ERROR')

    @protected
    def normalization(self, df: pd.DataFrame, mean_or_median: bool = True, test: bool = True):
        """
        Нормализация входных данных с использованием смещения и стандартного отклонения
        Args:
            df: Датафрейм с тестовыми/тренировочными данными
            test: Тестовые данные или нет
            mean_or_median: Если True, то подсчет Среднего по каждому столбцу, иначе Медианы
        Returns:
            Нормализованный набор данных
        """

        if not test:
            # Подсчет смещения и стандартного отклонения
            self.offset_std_calculation(df=df, mean_or_median=mean_or_median)

        elif self.redis_connection.exists('offset', 'std') == 2:
            # Если число существующих ключей равно 2
            self.offset = pickle.loads(self.redis_connection.get('offset'))
            self.std = pickle.loads(self.redis_connection.get('std'))

        else:
            # TODO обработка отсутствия значений в Редис
            pass

        # Нормализация
        df -= self.offset
        df /= self.std
        return df

    @protected
    def anomaly_detection(self, df: pd.DataFrame):
        """
        Метод получения индексов точек, относящихся к "аномалиям" (выбросам)
        Args:
            df: Набор данных для поиска аномалий

        Returns:
            Список индексов точек, являющихся аномалиями, т.е. выбросами
        """
        outlier_detection = DBSCAN(min_samples=2, eps=3)
        # Кластеризация данных, если кластер является -1, то это "шум"
        clusters = outlier_detection.fit_predict(df)
        return np.where(clusters != -1)
