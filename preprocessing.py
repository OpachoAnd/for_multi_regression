import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv(r'C:\Users\opacho\Documents\GitHub\for_multi_regression\source_data\Fish.csv')
df = pd.DataFrame(data, columns=['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight'])


def preprocessing_dataframe(df):
    # преобразование текстовых столбцов и удаление строк с пустыми элементами
    processed_df = df.copy(deep=True)
    for i in processed_df.columns:
        if str(processed_df[i].dtype) != 'float64':
            processed_df[i] = processed_df[i].factorize()[0]
    processed_df.dropna(inplace=True)
    return processed_df


def data_visualization(df, name_column: str):
    # визуализация распределения данных указанного столбца относительно общего количества элементов в датасете
    plt.figure(figsize=(10, 6))
    plt.hist(df[name_column], bins=50, ec='black', color='#2196f3')
    plt.xlabel('Weight Fish')
    plt.ylabel('Count')
    plt.show


def correlation(df, name_target_column: str):
    # корреляция параметров относительно выходного значения
    # должна стремиться либо к -1, либо к +1, если к 0, то зависимости параметра и выхода нет
    for i in df.columns[:-1]:
        print(f'корреляция столбца {str(i)} относительно выхода {str(name_target_column)}'
              f' равна {df[name_target_column].corr(df[i])}')
    print(f'\n{df.corr()}')


def data_logarithm(df):
    # Логарифмирование датасета для нормализации данных
    df_log = pd.DataFrame()
    for i in df.columns:
        y_log = np.log(df[i])
        df_log[i] = y_log
    return df_log


def data_separation(df, name_target_column: str):
    target_column = df[name_target_column]
    features = df.drop(name_target_column, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, target_column, test_size=0.2)
    return X_train, X_test, y_train, y_test


def VIF(X_train, threshold=5):
    # мера зависимости параметров друг от друга, должна быть меньше 5
    # возвращает df со строками, не прошедшими threshold
    X_const = sm.add_constant(X_train)
    vif = [variance_inflation_factor(exog=X_const.values, exog_idx=1) for i in range(X_const.shape[1])]
    # print(vif)
    df_vif = pd.DataFrame({'coef_name': X_const.columns, 'vif': np.around(vif, 2)})
    df_vif_threshold = df_vif.loc[df_vif['vif'] > threshold]
    return df_vif_threshold


if __name__ == "__main__":
    # correlation(df, 'Weight')
    df = preprocessing_dataframe(df)
    X_train, X_test, y_train, y_test = data_separation(df, 'Weight')
    (VIF(X_train))
    # variance_inflation_factor(exog=)
