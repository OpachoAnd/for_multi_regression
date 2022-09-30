import pickle

import pandas as pd
import redis
from sklearn import tree
from sklearn.model_selection import KFold

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

        # Кросс-валидация:
        trees = {}
        k_fold = KFold(n_splits=3, shuffle=False, random_state=1)
        for train, test in k_fold.split(X=train_df):
            model_cu_cd = tree.DecisionTreeRegressor(criterion='friedman_mse',
                                                     max_features='auto',
                                                     random_state=1
                                                     )
            model_cu_cd.fit(train_df.values[train], target_columns_cu_cd.values[train])
            score = model_cu_cd.score(train_df.values[test], target_columns_cu_cd.values[test])
            trees[score] = model_cu_cd

        self.model_Cu_Cd = trees[max(trees.keys())]

        try:
            self.redis_connection.set('model_Cu_Cd', pickle.dumps(self.model_Cu_Cd))

        except redis.ConnectionError:
            # TODO ЗАМЕНИТЬ НА ЛОГИРОВАНИЕ
            print('Connection Redis ERROR')

    def predict(self, test_df: pd.DataFrame):
        """
        Метод для тестирования модели
        Args:
            test_df: Набор данных для тестирования

        Returns:
            Модель, выдающая предсказание для Меди и Кадмия, либо None в случае её отсутствия
        """
        if self.model_Cu_Cd:
            return self.model_Cu_Cd.predict(test_df)

        try:
            self.redis_connection.exists('model_Cu_Cd')
            self.model_Cu_Cd = pickle.loads(self.redis_connection.get('model_Cu_Cd'))
            return self.model_Cu_Cd.predict(test_df)
        except redis.ConnectionError:
            print('Connection Redis ERROR')
            return None

