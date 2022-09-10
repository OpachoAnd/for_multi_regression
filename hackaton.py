import numpy as np
import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import os


class Zinc_Impurities(object):
    def __init__(self, path: str, path_out: str):
        self.features = pd.read_excel(path)
        self.features_out = pd.read_excel(path_out)
        self.indexes = ''
        self.model_cu = ''
        self.model_cd = ''
        self.path_out = path_out
    def deleting_incorrect_data(self):
        need_rows = self.features[self.features['CHPAC1_B051_LV  --   AT501A'] == 0.0]
        need_rows = need_rows[need_rows['CHPAC1_B052_LV  --   AT501B'] == 0.0]
        self.features.drop(labels=need_rows.index, axis=0, inplace=True)

        need_rows = self.features[self.features['CHPAC1_B051_LV  --   AT501A'] == 1.0]
        need_rows = need_rows[need_rows['CHPAC1_B052_LV  --   AT501B'] == 1.0]
        self.features.drop(labels=need_rows.index, axis=0, inplace=True)

        need_rows = self.features[self.features['CHPAC1_B053_LV  --   AT502A'] == 0.0]
        need_rows = need_rows[need_rows['CHPAC1_B054_LV  --   AT502B'] == 0.0]
        self.features.drop(labels=need_rows.index, axis=0, inplace=True)

        need_rows = self.features[self.features['CHPAC1_B053_LV  --   AT502A'] == 1.0]
        need_rows = need_rows[need_rows['CHPAC1_B054_LV  --   AT502B'] == 1.0]
        self.features.drop(labels=need_rows.index, axis=0, inplace=True)
        # Удаление второй строки
        self.features.drop(labels=0, axis=0, inplace=True)

        # Пустые столбецы
        self.features.drop(labels=' ', axis=1, inplace=True)
        self.features.drop(labels='CHPAC1_A033_PV  --   Cu - AT502A', axis=1, inplace=True)
        self.features.drop(labels='CHPAC1_A035_PV  --   Cd - AT502A', axis=1, inplace=True)
        self.features.drop(labels='CHPAC1_A034_PV  --   Zn - AT502A', axis=1, inplace=True)
        self.features.drop(labels='CHPAC1_A038_PV  --   Cu - AT502B', axis=1, inplace=True)
        self.features.drop(labels='CHPAC1_A040_PV  --   Cd - AT502B', axis=1, inplace=True)
        self.features.drop(labels='CHPAC1_A039_PV  --   Zn - AT502B', axis=1, inplace=True)

    def removing_redundant_data(self):
        # A_bool_1 = 1
        need_rows_A = self.features[self.features['CHPAC1_B051_LV  --   AT501A'] == 1.0].copy(deep=True)
        need_rows_A.drop(labels=['CHPAC1_B051_LV  --   AT501A',
                                 'CHPAC1_B052_LV  --   AT501B',
                                 'CHPAC1_M004_PV  --   расход после P510B',
                                 'CHPAC1_A028_PV  --   Cu - AT501B',
                                 'CHPAC1_A030_PV  --   Cd - AT501B',
                                 'CHPAC1_A029_PV  --   Zn - AT501B'], axis=1, inplace=True)

        # A_bool_1 = 1; A_bool_2 = 1
        need_rows_A_A = need_rows_A[need_rows_A['CHPAC1_B053_LV  --   AT502A'] == 1.0].copy(deep=True)
        need_rows_A_A.drop(labels=['CHPAC1_B053_LV  --   AT502A',
                                   'CHPAC1_B054_LV  --   AT502B',
                                   'CHPAC1_M100_PV  --   температура TK501B',
                                   'CHPAC1_M130_PV  --   pH - TK501B'], axis=1, inplace=True)

        # A_bool_1 = 1; B_bool_2 = 1
        need_rows_A_B = need_rows_A[need_rows_A['CHPAC1_B054_LV  --   AT502B'] == 1.0].copy(deep=True)
        need_rows_A_B.drop(labels=['CHPAC1_B053_LV  --   AT502A',
                                   'CHPAC1_B054_LV  --   AT502B',
                                   'CHPAC1_M099_PV  --   температура TK501А',
                                   'CHPAC1_M129_PV  --   pH - TK501А'], axis=1, inplace=True)

        # B_bool_1 = 1
        need_rows_B = self.features[self.features['CHPAC1_B052_LV  --   AT501B'] == 1.0].copy(deep=True)
        need_rows_B.drop(labels=['CHPAC1_B051_LV  --   AT501A',
                                 'CHPAC1_B052_LV  --   AT501B',
                                 'CHPAC1_M003_PV  --   расход после P510A',
                                 'CHPAC1_A023_PV  --   Cu - AT501A',
                                 'CHPAC1_A025_PV  --   Cd - AT501A',
                                 'CHPAC1_A024_PV  --   Zn - AT501A'], axis=1, inplace=True)

        # B_bool_1 = 1; A_bool_2 = 1
        need_rows_B_A = need_rows_B[need_rows_B['CHPAC1_B053_LV  --   AT502A'] == 1.0].copy(deep=True)
        need_rows_B_A.drop(labels=['CHPAC1_B053_LV  --   AT502A',
                                   'CHPAC1_B054_LV  --   AT502B',
                                   'CHPAC1_M100_PV  --   температура TK501B',
                                   'CHPAC1_M130_PV  --   pH - TK501B'], axis=1, inplace=True)

        # B_bool_1 = 1; B_bool_2 = 1
        need_rows_B_B = need_rows_B[need_rows_B['CHPAC1_B054_LV  --   AT502B'] == 1.0].copy(deep=True)
        need_rows_B_B.drop(labels=['CHPAC1_B053_LV  --   AT502A',
                                   'CHPAC1_B054_LV  --   AT502B',
                                   'CHPAC1_M099_PV  --   температура TK501А',
                                   'CHPAC1_M129_PV  --   pH - TK501А'], axis=1, inplace=True)

        _columns = list(need_rows_A_A.columns)
        for i in range(1, len(_columns)):
            _columns[i] = _columns[i][:-1]
            _columns[i] = _columns[i][21:]
        need_rows_A_A.columns = _columns

        _columns = list(need_rows_A_B.columns)
        for i in range(1, len(_columns)):
            _columns[i] = _columns[i][:-1]
            _columns[i] = _columns[i][21:]
        need_rows_A_B.columns = _columns

        _columns = list(need_rows_B_A.columns)
        for i in range(1, len(_columns)):
            _columns[i] = _columns[i][:-1]
            _columns[i] = _columns[i][21:]
        need_rows_B_A.columns = _columns

        _columns = list(need_rows_B_B.columns)
        for i in range(1, len(_columns)):
            _columns[i] = _columns[i][:-1]
            _columns[i] = _columns[i][21:]
        need_rows_B_B.columns = _columns

        frames = [need_rows_A_A, need_rows_A_B, need_rows_B_A, need_rows_B_B]

        result = pd.concat(frames)
        result.index = result['Unnamed: 0']
        result.drop(labels='Unnamed: 0', axis=1, inplace=True)
        self.indexes = result.index
        self.features = result.values

        self.features = np.array(self.features, dtype=np.float64)

        # Среднее значение np.median(
        mean = np.mean(self.features, axis=0)  # X_train.mean(axis=0)
        # # Стандартное отклонение
        std = np.std(self.features, axis=0)
        self.features -= mean
        self.features /= std

    def download_weights(self):
        with open(os.path.join(os.getcwd(), 'model_cu.pkl'), 'rb') as fp1:
            self.model_cu = pickle.load(fp1)
        with open(os.path.join(os.getcwd(), 'model_cd.pkl'), 'rb') as fp2:
            self.model_cd = pickle.load(fp2)

    def prediction(self):
        self.download_weights()
        predict_cu = self.model_cu.predict(self.features)
        predict_cd = self.model_cd.predict(self.features)
        #print(self.features_out['Unnamed: 0'])
        # self.features_out.index = self.features_out['Unnamed: 0']
        #print(self.features_out.index)

        for i, j in zip(self.features_out['Unnamed: 0'], self.features_out.index):
            # print(f'{i}, {j}')
            if i == 'Дата Время':
                continue
            index = list(self.indexes).index(i)

            p_cu = predict_cu[index]
            p_cd = predict_cd[index]

            self.features_out.loc[j, 'Cu - AT502'] = p_cu
            self.features_out.loc[j, 'Cd - AT502'] = p_cd
        self.features_out.to_excel(self.path_out, index=False)
            # self.features_out.where(self.features_out['Unnamed: 0'] == i)
            #
            # # self.features_out.iloc[i]['Cu - AT502'] = predict_cu[index]
            # # self.features_out.iloc[i]['Cd - AT502'] = predict_cd[index]
            # print(i)
            # print(self.features_out.iloc[i])

if __name__ == '__main__':
    zi = Zinc_Impurities(path=r'C:\Users\Андрей\Documents\GitHub\Данные 2018_07 - 2019_06\2019_07 - проверка - входные переменные.xls',
                         path_out=r'C:\Users\Андрей\Documents\GitHub\Данные 2018_07 - 2019_06\2019_07 - временные точки.xls')
    zi.deleting_incorrect_data()
    zi.removing_redundant_data()
    zi.download_weights()
    zi.prediction()
