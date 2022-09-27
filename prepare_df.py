import pandas as pd
from accessify import private


class Prepare_Df(object):
    def __init__(self, name_target_column_cuprum: str, name_target_column_cadmium: str):
        """
        Класс для предварительной обработки исходных данных
        Args:
            name_target_column_cuprum: Имя столбца с истинными значениями Меди
            name_target_column_cadmium: Имя столбца с истинными значениями Кадмия
        """
        self.name_target_column_Cuprum = name_target_column_cuprum
        self.name_target_column_Cadmium = name_target_column_cadmium

        self.target_column_Cuprum = None
        self.target_column_Cadmium = None

    def deleting_incorrect_data_1(self, df: pd.DataFrame, test: bool = True) -> pd.DataFrame:
        """
        Метод для удаления некорректных данных
        Args:
            test: Тестовый набор данных или нет
            df: Датафрейм с данными

        Returns:
            Обработанный датафрейм
        """
        df = df.copy(deep=True)
        df = df[(df['Cu_AT501_A'] > 1) &
                (df['Cd_AT501_A'] > 1) &
                (df['Zn_AT501_A'] > 1) &

                (df['Cu_AT501_B'] > 1) &
                (df['Cd_AT501_B'] > 1) &
                (df['Zn_AT501_B'] > 1) &

                (df['Cu_AT502_A'] > 1) &
                (df['Cd_AT502_A'] > 1) &
                (df['Zn_AT502_A'] > 1) &

                (df['Cu_AT502_B'] > 1) &
                (df['Cd_AT502_B'] > 1) &
                (df['Zn_AT502_B'] > 1)
                ]
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
        df = self.deleting_an_output_3(df=df, test=test)
        return df

    @private
    def removing_redundant_data_2(self, df) -> pd.DataFrame:
        """
        Удаление избыточных данных
        Args:
            df: Датафрейм с данными

        Returns:
            Обработанный датафрейм
        """
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

        df = result
        return df

    @private
    def deleting_an_output_3(self, df: pd.DataFrame, test: bool) -> pd.DataFrame:
        """
        Удаление лишних выходных столбцов
        Args:
            test: Тестовый набор данных или нет
            df: Датафрейм с данными

        Returns:
            Обработанный датафрейм
        """

        df = df.copy(deep=True)

        if not test:
            # Если набор данных ТРЕНИРОВОЧНЫЙ, то удаляем строки с пустыми столбцами
            df = df.dropna(how='any', axis=0)

        self.target_column_Cuprum = df[self.name_target_column_Cuprum].copy(deep=True)  # 'Cu_AT502']
        self.target_column_Cadmium = df[self.name_target_column_Cadmium].copy(deep=True)

        df = df.drop(labels=['Cu_AT502', 'Cd_AT502', 'Zn_AT502'], axis=1)

        if test:
            # Если набор данных ТЕСТОВЫЙ, то удаляем строки с пустыми столбцами
            df = df.dropna(how='any', axis=0)

        if 'Ti' in df.columns:
            df.index = df['Ti']
            df.drop(labels=['Ti'], axis=1, inplace=True)
        return df
