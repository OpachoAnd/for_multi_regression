import numpy as np
import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import os
from prepare_df import Prepare_Df


class Prepare_Df_Test(Prepare_Df):
    def __init__(self,
                 path_test_df: str,
                 path_time_point: str,
                 name_target_column_cuprum: str,
                 name_target_column_cadmium: str
                 ):
        # Конструктор для родительского класса
        super().__init__(path=path_test_df,
                         name_target_column_cuprum=name_target_column_cuprum,
                         name_target_column_cadmium=name_target_column_cadmium)

        if '.csv' in path_time_point:
            self.df_time_point = pd.read_csv(path_time_point)
        elif '.xls' in path_time_point:
            self.df_time_point = pd.read_excel(path_time_point)

        self.selecting_df = self.df.loc[self.df_time_point['time']]
