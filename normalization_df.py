import pickle

import numpy as np
import pandas as pd
import redis
from accessify import private


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

    @private
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
        self.redis_connection.set('offset', pickle.dumps(self.offset))
        self.redis_connection.set('std', pickle.dumps(self.std))

    def normalization(self, df: pd.DataFrame, mean_or_median: bool = True, test: bool = True):
        """
        Нормализация входных данных с использованием смещения и стандартного отклонения
        Args:
            df:
            test: Тестовые данные или нет
            mean_or_median: Если True, то подсчет Среднего по каждому столбцу, иначе Медианы
        Returns:
            Без возвращаемого значения
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
        self.df -= self.offset
        self.df /= self.std
