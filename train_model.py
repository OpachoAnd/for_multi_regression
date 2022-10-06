import pickle

import numpy as np
import pandas as pd
import redis
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from normalization_df import Normalization_Df


class Train_Model(Normalization_Df):
    def __init__(self, redis_connection: redis.client.Redis):
        super().__init__(redis_connection=redis_connection)

        self.model_Cu_Cd = None
        self.model_trees_grad_boosting = []

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

    def train_gradient_boost(self,
                             df: pd.DataFrame,
                             target_columns_cu_cd: pd.DataFrame,
                             nu: float = 0.1,
                             n: int = 10):
        train_columns = ['P510_expense', 'Cu_AT501', 'Cd_AT501', 'Zn_AT501', 'temp', 'pH', 'auto_W503']
        df = df.copy(deep=True)

        model_pred = tree.DecisionTreeRegressor(criterion='friedman_mse',
                                                max_features='auto',
                                                random_state=1
                                                )
        model_pred.fit(df, target_columns_cu_cd)
        self.model_trees_grad_boosting.append(model_pred)

        predict = model_pred.predict(X=df)
        predict = np.transpose(predict, axes=[1, 0])

        df['y_pred_Cu_AT502'] = predict[0]
        df['y_pred_Cd_AT502'] = predict[1]

        for i in range(n):
            df['residual_Cu_AT502'] = target_columns_cu_cd['Cu_AT502'] - df['y_pred_Cu_AT502']
            df['residual_Cd_AT502'] = target_columns_cu_cd['Cd_AT502'] - df['y_pred_Cd_AT502']

            targets = pd.concat([df['residual_Cu_AT502'], df['residual_Cd_AT502']], axis=1)
            tree_Cu_Cd_AT502 = tree.DecisionTreeRegressor(criterion='friedman_mse', max_features='auto', random_state=1)
            tree_Cu_Cd_AT502.fit(df[train_columns], targets)

            predict_residual = tree_Cu_Cd_AT502.predict(df[train_columns])
            df['y_pred_Cu_AT502'] += nu * np.transpose(predict_residual, axes=[1, 0])[0]
            df['y_pred_Cd_AT502'] += nu * np.transpose(predict_residual, axes=[1, 0])[1]

            self.model_trees_grad_boosting.append(tree_Cu_Cd_AT502)

        try:
            self.redis_connection.set('model_gradient_boost', pickle.dumps(self.model_trees_grad_boosting))
        except redis.ConnectionError:
            # TODO ЗАМЕНИТЬ НА ЛОГИРОВАНИЕ
            print('Connection Redis ERROR')


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
