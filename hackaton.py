import numpy as np
import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import os


class Zinc_Impurities(object):
    def __init__(self, path_train: str, name_target_column: str, path_test: str, path_out: str):
        self.name_target_column = name_target_column
        self.df_train = self.deleting_incorrect_data_1(pd.read_csv(path_train))

        self.target_column_df_train = None
        self.median_data_train = None
        self.std_data_train = None

        self.df_test = self.deleting_incorrect_data_1(pd.read_excel(path_test))

    def deleting_incorrect_data_1(self, df: pd.DataFrame):
        df = df.copy(deep=True)
        need_rows = df[df['A_bool_1'] == 0.0]
        need_rows = need_rows[need_rows['B_bool_1'] == 0.0]
        df.drop(labels=need_rows.index, axis=0, inplace=True)

        need_rows = df[df['A_bool_1'] == 1.0]
        need_rows = need_rows[need_rows['B_bool_1'] == 1.0]
        df.drop(labels=need_rows.index, axis=0, inplace=True)

        need_rows = df[df['A_bool_2'] == 0.0]
        need_rows = need_rows[need_rows['B_bool_2'] == 0.0]
        df.drop(labels=need_rows.index, axis=0, inplace=True)

        need_rows = df[df['A_bool_2'] == 1.0]
        need_rows = need_rows[need_rows['B_bool_2'] == 1.0]
        df.drop(labels=need_rows.index, axis=0, inplace=True)

        df = self.removing_redundant_data_2(df=df)
        df = self.deleting_an_output_3(df=df)
        return df

    def removing_redundant_data_2(self, df):
        # A_bool_1 = 1

        need_rows_A = df[df['A_bool_1'] == 1.0].copy(deep=True)
        need_rows_A.drop(labels=['A_bool_1',
                                 'B_bool_1',
                                 'P510_expense_B',
                                 'Cu_AT501_B',
                                 'Cd_AT501_B',
                                 'Zn_AT501_B'], axis=1, inplace=True)

        # A_bool_1 = 1; A_bool_2 = 1
        need_rows_A_A = need_rows_A[need_rows_A['A_bool_2'] == 1.0].copy(deep=True)
        need_rows_A_A.drop(labels=['A_bool_2',
                                   'B_bool_2',
                                   'temp_B',
                                   'pH_B',
                                   'Cu_AT502_B',
                                   'Cd_AT502_B',
                                   'Zn_AT502_B'], axis=1, inplace=True)

        # A_bool_1 = 1; B_bool_2 = 1
        need_rows_A_B = need_rows_A[need_rows_A['B_bool_2'] == 1.0].copy(deep=True)
        need_rows_A_B.drop(labels=['A_bool_2',
                                   'B_bool_2',
                                   'temp_А',
                                   'pH_А',
                                   'Cu_AT502_A',
                                   'Cd_AT502_A',
                                   'Zn_AT502_A'], axis=1, inplace=True)

        # B_bool_1 = 1
        need_rows_B = df[df['B_bool_1'] == 1.0].copy(deep=True)
        need_rows_B.drop(labels=['A_bool_1',
                                 'B_bool_1',
                                 'P510_expense_A',
                                 'Cu_AT501_A',
                                 'Cd_AT501_A',
                                 'Zn_AT501_A'], axis=1, inplace=True)

        # B_bool_1 = 1; A_bool_2 = 1
        need_rows_B_A = need_rows_B[need_rows_B['A_bool_2'] == 1.0].copy(deep=True)
        need_rows_B_A.drop(labels=['A_bool_2',
                                   'B_bool_2',
                                   'temp_B',
                                   'pH_B',
                                   'Cu_AT502_B',
                                   'Cd_AT502_B',
                                   'Zn_AT502_B'], axis=1, inplace=True)

        # B_bool_1 = 1; B_bool_2 = 1
        need_rows_B_B = need_rows_B[need_rows_B['B_bool_2'] == 1.0].copy(deep=True)
        need_rows_B_B.drop(labels=['A_bool_2',
                                   'B_bool_2',
                                   'temp_А',
                                   'pH_А',
                                   'Cu_AT502_A',
                                   'Cd_AT502_A',
                                   'Zn_AT502_A'], axis=1, inplace=True)

        _columns = list(need_rows_A_A.columns)
        for i in range(len(_columns)):
            _columns[i] = _columns[i][:-2]
        need_rows_A_A.columns = _columns

        _columns = list(need_rows_A_B.columns)
        for i in range(len(_columns)):
            _columns[i] = _columns[i][:-2]
        need_rows_A_B.columns = _columns

        _columns = list(need_rows_B_A.columns)
        for i in range(len(_columns)):
            _columns[i] = _columns[i][:-2]
        need_rows_B_A.columns = _columns

        _columns = list(need_rows_B_B.columns)
        for i in range(len(_columns)):
            _columns[i] = _columns[i][:-2]
        need_rows_B_B.columns = _columns

        frames = [need_rows_A_A, need_rows_A_B, need_rows_B_A, need_rows_B_B]

        result = pd.concat(frames)

        #result = result.dropna(how='any', axis=0)

        df = result
        return df

    def deleting_an_output_3(self, df: pd.DataFrame):
        # Удаление лишних выходных столбцов
        df = df.copy(deep=True)
        df = df[df['Zn_AT502'].notna()]

        self.target_column_df_train = df[self.name_target_column]  # 'Cu_AT502']
        feat = df.drop('Cu_AT502', axis=1)
        feat = feat.drop('Cd_AT502', axis=1)
        feat = feat.drop('Zn_AT502', axis=1)
        df = feat
        return df

    def normal_distribution_train_df(self):
        # Медиана для каждого параметра (т.е. по столбцам)
        # print(self.df_train.values.shape)
        self.median_data_train = np.mean(self.df_train.values, axis=0)  # X_train.mean(axis=0)
        # print(self.median_data_train)
        # Стандартное отклонение
        self.std_data_train = self.df_train.values.std(axis=0)
        # Приведение к numpy-массиву
        self.df_train = self.df_train.values
        # Нормализация
        self.df_train -= self.median_data_train
        self.df_train /= self.std_data_train

    def normal_distribution_test_df(self):
        self.df_test.drop(labels=['ti'], axis=1, inplace=True)
        # self.df_test = self.df_test.values
        # # Нормализация
        # self.df_test -= self.median_data_train
        # self.df_test /= self.std_data_train
        # print(self.median_data_train)

    def download_weights(self):
        with open(r'C:\Users\Андрей\Documents\GitHub\for_multi_regression\model_cu.pkl', 'rb') as fp1:
            self.model_cu = pickle.load(fp1)
        with open(r'C:\Users\Андрей\Documents\GitHub\for_multi_regression\model_cd.pkl', 'rb') as fp2:
            self.model_cd = pickle.load(fp2)
        # with open(os.path.join(os.getcwd(), 'model_cu.pkl'), 'rb') as fp1:
        #     self.model_cu = pickle.load(fp1)
        # with open(os.path.join(os.getcwd(), 'model_cd.pkl'), 'rb') as fp2:
        #     self.model_cd = pickle.load(fp2)

    def prediction(self):
        self.download_weights()
        predict_cu = self.model_cu.predict(self.df_train)
        predict_cd = self.model_cd.predict(self.df_train)

        for i, j in zip(self.df_out['Unnamed: 0'], self.df_out.index):
            if i == 'Дата Время':
                continue
            index = list(self.indexes).index(i)

            p_cu = predict_cu[index]
            p_cd = predict_cd[index]
            self.df_out.loc[j, 'Cu - AT502'] = p_cu
            self.df_out.loc[j, 'Cd - AT502'] = p_cd

        self.df_out.drop(labels=['Unnamed: 3', 'Unnamed: 4', ' '], axis=1, inplace=True)
        self.df_out.columns = ['', 'Cu - AT502', 'Cd - AT502']
        self.df_out.to_excel(self.path_out, index=False)


# if __name__ == '__main__':
#     zi = Zinc_Impurities(path=r'C:\Users\Андрей\Documents\GitHub\Данные 2018_07 - 2019_06\2019_07 - проверка - входные переменные.xls',
#                          path_out=r'C:\Users\Андрей\Documents\GitHub\Данные 2018_07 - 2019_06\2019_07 - временные точки.xls')
#     zi.deleting_incorrect_data()
#     zi.removing_redundant_data()
#     zi.download_weights()
#     zi.prediction()
