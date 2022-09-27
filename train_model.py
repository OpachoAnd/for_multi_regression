import pickle

import pandas as pd
import redis
from sklearn import tree

from normalization_df import Normalization_Df


class Train_Model(Normalization_Df):
    def __init__(self, redis_connection: redis.client.Redis):
        super().__init__(redis_connection=redis_connection)

        self.model_Cu = None
        self.model_Cd = None

    def train(self, train_df: pd.DataFrame, target_column_cu: pd.Series, target_column_cd: pd.Series):
        self.model_Cu = tree.DecisionTreeRegressor(criterion='friedman_mse',
                                                max_features='auto',
                                                random_state=1
                                                )
        self.model_Cd = tree.DecisionTreeRegressor(criterion='friedman_mse',
                                                max_features='auto',
                                                random_state=1
                                                )

        self.model_Cu.fit(train_df, target_column_cu)
        self.model_Cd.fit(train_df, target_column_cd)

    def predict(self):
        if self.redis_connection.ping() and self.redis_connection.exists('model_Cu', 'model_Cd') == 2:
            self.model_Cu = pickle.loads(self.redis_connection.get('model_Cu'))
            self.model_Cd = pickle.loads(self.redis_connection.get('model_Cd'))

