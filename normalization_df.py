import pickle

import numpy as np
import pandas as pd

from settings import redis_connect


class Normalization_Df(object):
    def __init__(self, df: pd.DataFrame, test: bool = True):
        """
        Класс для нормализации входных данных
        Args:
            df: Датафрейм с входными данными
            test: Тестовые данные или нет
        """
        self.offset = None
        self.std = None
        self.test = test
        self.df = df.values

    def offset_calculation(self, mean: bool):
        """
        Подчсет смещения данных, то есть среднего (медианы) по каждому столбцу
        Args:
            mean: Если True, то подсчет Среднего по каждому столбцу, иначе Медианы

        Returns:
            Без возвращаемого значения
        """
        if mean:
            self.offset = np.mean(self.df, axis=0)
        else:
            self.offset = np.median(self.df, axis=0)

        redis_connect.set('offset', pickle.dumps(self.offset))

    def std_calculation(self):
        """
        Подсчет стандартного отклонения данных
        Returns:
            Без возвращаемого значения
        """
        self.std = np.std(self.df, axis=0)

        redis_connect.set('std', pickle.dumps(self.std))

    def normalization(self, mean: bool = True):
        """
        Нормализация входных данных с использованием смещения и стандартного отклонения
        Args:
            mean: Если True, то подсчет Среднего по каждому столбцу, иначе Медианы

        Returns:
            Без возвращаемого значения
        """
        if not self.test:
            # Подсчет смещения и стандартного отклонения
            self.offset_calculation(mean=mean)
            self.std_calculation()
        elif redis_connect.exists('offset', 'std') == 2:
            # Если число существующих ключей равно 2
            self.offset = pickle.loads(redis_connect.get('offset'))
            self.std = pickle.loads(redis_connect.get('std'))
        else:
            # TODO обработка отсутствия значений в Редис
            pass
        # Нормализация
        self.df -= self.offset
        self.df /= self.std
