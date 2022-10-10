import logging
import pickle

import numpy as np
import pandas as pd
import redis
from accessify import private
from sklearn import tree
from sklearn.model_selection import KFold

from normalization_df import Normalization_Df


class Train_Model(Normalization_Df):
    def __init__(self, redis_connection: redis.client.Redis):
        super().__init__(redis_connection=redis_connection)

        self.model_Cu_Cd = None
        self.model_trees_grad_boosting = []

    @private
    def clean_data(self, df: pd.DataFrame, target_columns_cu_cd: pd.DataFrame, removing_anomalies: bool):
        """
        Метод, выполняющий комплексную очистку данных от шума
        Args:
            df: Датафрейм для очистки
            target_columns_cu_cd: Метки этого датафрейма
            removing_anomalies: Если true, то чистка от выбросов в данных
        Returns:
            Очищенные df и target_columns_cu_cd
        """
        train_df = self.normalization(df=df, mean_or_median=True, test=False)
        target_columns_cu_cd = target_columns_cu_cd

        if removing_anomalies:
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

        return train_df, target_columns_cu_cd

    def train(self, df: pd.DataFrame, target_columns_cu_cd: pd.DataFrame, removing_anomalies: bool = True):
        """
        Метод для обучения модели "Решающее дерево регрессии"
        Args:
            df: Набор данных для загрузки в модель (Обучающие данные)
            target_columns_cu_cd: Два столбца с истинными данными для меди и кадмия
            removing_anomalies: Если true, то чистка от выбросов в данных
        Returns:
            Нет возвращаемого значения
        """
        train_df, target_columns_cu_cd = self.clean_data(df, target_columns_cu_cd, removing_anomalies)

        # Cross-validation:
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

        self.model_Cu_Cd = trees[max(trees.keys())]

        try:
            self.redis_connection.set('model_Cu_Cd', pickle.dumps(self.model_Cu_Cd))

        except redis.ConnectionError:
            logging.warning('Redis connection ERROR')

    def train_gradient_boost(self,
                             df: pd.DataFrame,
                             target_columns_cu_cd: pd.DataFrame,
                             removing_anomalies: bool = True,
                             nu: float = 0.1,
                             n: int = 10):
        """
        Метод для обучения моделей "Решающее дерево регрессии" с применением метода "Градиентный бустинг"
        Args:
            df: Набор данных для загрузки в модель (Обучающие данные)
            target_columns_cu_cd: Два столбца с истинными данными для меди и кадмия
            removing_anomalies: Если true, то чистка от выбросов в данных
            nu: Скорость уменьшения ошибки
            n: Число деревьев для градиентного бустинга
        Returns:
            Нет возвращаемого значения
        """

        train_columns = ['P510_expense', 'Cu_AT501', 'Cd_AT501', 'Zn_AT501', 'temp', 'pH', 'auto_W503']
        df = df.copy(deep=True)

        df, target_columns_cu_cd = self.clean_data(df, target_columns_cu_cd, removing_anomalies)

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
            logging.warning('Redis connection ERROR')

    def predict(self, test_df: pd.DataFrame, gradient_boosting: bool = False, nu: float = 0.1):
        """
        Метод для тестирования модели
        Args:
            test_df: Набор данных для тестирования
            nu: Скорость уменьшения ошибки
            gradient_boosting: Если True, то применяется метод градиентного бустинга для предсказания

        Returns:
            DataFrame, с предсказанием для Меди и Кадмия, либо None в случае отсутствия обученной модели
        """
        test_df = self.normalization(df=test_df, mean_or_median=True, test=False)
        test_df.reset_index(drop=True, inplace=True)

        if not gradient_boosting and self.model_Cu_Cd:
            test_predict = self.model_Cu_Cd.predict(test_df)
            test_predict = pd.DataFrame(test_predict, columns=['Cu_AT502', 'Cd_AT502']).reset_index()

            return pd.concat([test_df, test_predict[['Cu_AT502', 'Cd_AT502']]], axis=1)

        elif gradient_boosting and self.model_trees_grad_boosting:
            test_predict = self.model_trees_grad_boosting[0].predict(test_df)
            for i in self.model_trees_grad_boosting[1::]:
                test_predict += nu * i.predict(test_df)

            test_predict = pd.DataFrame(test_predict, columns=['Cu_AT502', 'Cd_AT502']).reset_index()
            return pd.concat([test_df, test_predict[['Cu_AT502', 'Cd_AT502']]], axis=1)

        try:
            if not gradient_boosting:
                self.redis_connection.exists('model_Cu_Cd')
                self.model_Cu_Cd = pickle.loads(self.redis_connection.get('model_Cu_Cd'))
                test_predict = self.model_Cu_Cd.predict(test_df)
                test_predict = pd.DataFrame(test_predict, columns=['Cu_AT502', 'Cd_AT502']).reset_index()

                return pd.concat([test_df, test_predict[['Cu_AT502', 'Cd_AT502']]], axis=1)

            else:
                self.redis_connection.exists('model_gradient_boost')
                self.model_trees_grad_boosting = pickle.loads(self.redis_connection.get('model_gradient_boost'))

                test_predict = self.model_trees_grad_boosting[0].predict(test_df)
                for i in self.model_trees_grad_boosting[1::]:
                    test_predict += nu * i.predict(test_df)

                test_predict = pd.DataFrame(test_predict, columns=['Cu_AT502', 'Cd_AT502']).reset_index()

                return pd.concat([test_df, test_predict[['Cu_AT502', 'Cd_AT502']]], axis=1)

        except redis.ConnectionError:
            logging.warning('Redis connection error')
            return None
