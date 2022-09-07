import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\opacho\Documents\GitHub\for_multi_regression\source_data\Fish.csv')
df = pd.DataFrame(data, columns=['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight'])
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
    df_log = pd.DataFrame()
    for i in df.columns:
        y_log = np.log(df[i])
        df_log[i] = y_log
    return df_log


if __name__ == "__main__":
    #correlation(df, 'Weight')
    print(data_logarithm(df))