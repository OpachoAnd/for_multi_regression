import pickle

import pandas as pd
import redis
from sklearn import tree
from sklearn.model_selection import KFold, train_test_split

from normalization_df import Normalization_Df


class Train_Model(Normalization_Df):
    def __init__(self, redis_connection: redis.client.Redis):
        super().__init__(redis_connection=redis_connection)

        self.model_Cu_Cd = None

    def train(self, df: pd.DataFrame, target_columns_cu_cd: pd.DataFrame):
        """
        Метод для обучения модели "Решающее дерево регрессии"
        Args:
            df: Набор данных для загрузки в модель (Обучающие данные)
            target_columns_cu_cd: Два столбца с истинными данными для меди и кадмия
        Returns:
            Нет возвращаемого значения
        """
        train_df = self.normalization(df=df,
                                      mean_or_median=True,
                                      test=False
                                      )
        # Обновление индексов в датафреймах train и target
        train_df.reset_index(drop=True, inplace=True)
        target_columns_cu_cd.reset_index(drop=True, inplace=True)

        # Метки выбросов, определенных с помощью DBScan
        labels_without_anomalies_DBScan = self.anomaly_detection_DBSCAN(df=train_df)
        train_df = train_df.loc[labels_without_anomalies_DBScan]
        target_columns_cu_cd = target_columns_cu_cd.loc[labels_without_anomalies_DBScan]

        # Обновление индексов в датафреймах train и target
        train_df.reset_index(drop=True, inplace=True)
        target_columns_cu_cd.reset_index(drop=True, inplace=True)

        # Метки выбросов, определенных с помощью STD
        labels_without_anomalies_STD = self.anomaly_detection_STD(df=train_df)
        train_df = train_df.loc[labels_without_anomalies_STD]
        target_columns_cu_cd = target_columns_cu_cd.loc[labels_without_anomalies_STD]

        # Обновление индексов в датафреймах train и target
        train_df.reset_index(drop=True, inplace=True)
        target_columns_cu_cd.reset_index(drop=True, inplace=True)

        # Кросс-валидация:
        trees = {}
        k_fold = KFold(n_splits=5, shuffle=True)

        for train, test in k_fold.split(train_df):
            model_cu_cd = tree.DecisionTreeRegressor(criterion='friedman_mse',
                                                     max_features='auto',
                                                     random_state=1
                                                     )
            model_cu_cd.fit(train_df.loc[train], target_columns_cu_cd.loc[train])
            score = model_cu_cd.score(train_df.loc[test], target_columns_cu_cd.loc[test])
            trees[score] = model_cu_cd

            print(score)

            # q = model_cu_cd.predict(train_df.values[test])
            # for i in range(len(q)):
            #     print(q[i], target_columns_cu_cd.values[test][i])

        self.model_Cu_Cd = trees[max(trees.keys())]

        try:
            self.redis_connection.set('model_Cu_Cd', pickle.dumps(self.model_Cu_Cd))

        except redis.ConnectionError:
            # TODO ЗАМЕНИТЬ НА ЛОГИРОВАНИЕ
            print('Connection Redis ERROR')

    def train_gradient_boost(self):
        pass



    # def predict(self, test_df: pd.DataFrame):
    #     """
    #     Метод для тестирования модели
    #     Args:
    #         test_df: Набор данных для тестирования
    #
    #     Returns:
    #         Модель, выдающая предсказание для Меди и Кадмия, либо None в случае её отсутствия
    #     """
    #     if self.model_Cu_Cd:
    #         return self.model_Cu_Cd.predict(test_df)
    #
    #     try:
    #         self.redis_connection.exists('model_Cu_Cd')
    #         self.model_Cu_Cd = pickle.loads(self.redis_connection.get('model_Cu_Cd'))
    #         return self.model_Cu_Cd.predict(test_df)
    #     except redis.ConnectionError:
    #         print('Connection Redis ERROR')
    #         return None
