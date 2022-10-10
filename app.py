import logging
import os
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import *
from tkinter import ttk

import pandas as pd

from prepare_df import Prepare_Df
from settings import REDIS_CONNECTION
from train_model import Train_Model

logging.basicConfig(filename=os.path.join(os.getcwd(), 'info_app.log'),
                    format=u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s]  %(message)s',
                    level=logging.WARNING)


class User_Interface(tk.Frame):
    def __init__(self, prepare_dataframe: Prepare_Df, train_models: Train_Model, window=None):
        """
        Класс для пользовательского интерфейса
        Args:
            prepare_dataframe: Объект класса для предобработки данных
            train_models: Объект класса для обучения модели
            window: Контейнер пользовательского интерфейса
        """
        super().__init__()
        self.window = window
        self.train_model = train_models
        self.prepare_df = prepare_dataframe

        self.text_enter_data = None
        self.train_test_listbox = None
        self.text_path_save_data = None
        self.enabled_gradient_boosting_check = None
        self.enabled_clean_data_check = None
        self.score = None

        self.initUI()
        self.window.mainloop()

    def choose_file(self):
        """
        Метод всплывающего окна выбора файла для предсказания
        Returns:
            Без возвращаемого значения
        """
        filetypes = (("Таблица", "*csv *.xlsx *.xls"),
                     ("Любой", "*"))
        directory = fd.askopenfilename(title="Выбрать файл",
                                       initialdir=os.getcwd(),
                                       filetypes=filetypes)
        if directory:
            self.text_enter_data.insert(0, directory)

    def choose_directory(self):
        """
        Метод всплывающего окна выбора папки для сохранения результатов предсказания
        Returns:
            Без возвращаемого значения
        """
        directory = fd.askdirectory(title="Open a folder", initialdir=os.getcwd())
        if directory:
            self.text_path_save_data.insert(0, directory)

    def ok(self):
        """
        Метод для запуска системы
        Returns:
            Без возвращаемого значения
        """
        path_data = self.text_enter_data.get()
        train_test_listbox = self.train_test_listbox.get()
        path_test_answer = self.text_path_save_data.get()
        gradient_boosting_check = self.enabled_gradient_boosting_check.get()
        clean_data_check = self.enabled_clean_data_check.get()

        if path_data != '':
            try:
                df = pd.read_csv(path_data)
            except UnicodeDecodeError:
                df = pd.read_excel(path_data)

            df = self.prepare_df.deleting_incorrect_data(df=df, test=False)

            if train_test_listbox == 'train' and gradient_boosting_check:
                train_model.train_gradient_boost(df=df,
                                                 target_columns_cu_cd=self.prepare_df.target_columns_Cu_Cd,
                                                 removing_anomalies=clean_data_check,
                                                 nu=0.1,
                                                 n=5)
                self.score.insert(0,
                                  f'The training is completed, the model understands the data on {train_model.score}%')

            elif train_test_listbox == 'train':
                train_model.train(df=df,
                                  target_columns_cu_cd=self.prepare_df.target_columns_Cu_Cd,
                                  removing_anomalies=clean_data_check)
                self.score.insert(0,
                                  f'The training is completed, the model understands the data on {train_model.score}%')

            elif train_test_listbox == 'test':
                if path_test_answer != '':
                    test_answer = train_model.predict(test_df=df,
                                                      gradient_boosting=gradient_boosting_check)
                    test_answer.to_excel(os.path.join(path_test_answer, 'test_answer.xlsx'))
                else:
                    logging.warning('An empty path to saving predicted data')
        else:
            logging.warning('Empty data path')

    def initUI(self):
        self.window['bg'] = 'white'
        self.window.title("Modeling of cuprum and cadmium deposition in zinc solution")
        self.window.geometry('500x235')

        line_frame = Frame(self.window)
        line_frame['bg'] = 'white'
        line_frame.grid(row=0, column=0, columnspan=2)

        lbl_enter_data = Label(line_frame,
                               text="The path to the data file for prediction:",
                               foreground="black",
                               background="white"
                               )
        lbl_enter_data.grid()

        self.text_enter_data = Entry(line_frame,
                                     width=65,
                                     foreground="black",
                                     background="white")
        self.text_enter_data.grid(row=1, column=0)

        button_choose_file = Button(line_frame,
                                    background='white',
                                    text="Choose file",
                                    command=self.choose_file)
        button_choose_file.grid(row=1, column=1, sticky=EW)

        lbl_train_or_test = Label(line_frame,
                                  text="Training or test data",
                                  foreground="black",
                                  background="white"
                                  )
        lbl_train_or_test.grid(row=2, column=0, sticky=EW)

        train_test = ['train', 'test']
        variable = StringVar(window)
        variable.set('train')
        self.train_test_listbox = ttk.Combobox(line_frame, textvariable=variable, values=train_test)
        self.train_test_listbox.grid(row=3, column=0, sticky=EW)

        self.enabled_gradient_boosting_check = BooleanVar()
        gradient_boosting_check = tk.Checkbutton(line_frame,
                                                 text="Применить метод Gradient Boosting",
                                                 variable=self.enabled_gradient_boosting_check,
                                                 bg='white')
        gradient_boosting_check.grid(row=6, column=0, sticky=E)

        lbl_path_save_data = Label(line_frame,
                                   text="The path to save the output data (if the data is test)",
                                   foreground="black",
                                   background="white"
                                   )
        lbl_path_save_data.grid(row=4, column=0, sticky=EW)

        self.text_path_save_data = Entry(line_frame,
                                         width=65,
                                         foreground="black",
                                         background="white")
        self.text_path_save_data.grid(row=5, column=0)

        button_choose_folder = Button(line_frame,
                                      background='white',
                                      text="Choose folder",
                                      command=self.choose_directory)
        button_choose_folder.grid(row=5, column=1, sticky=EW)

        self.enabled_clean_data_check = BooleanVar()
        clean_data_check = tk.Checkbutton(line_frame,
                                          text="Clearing data from anomalies",
                                          variable=self.enabled_clean_data_check,
                                          bg='white')
        clean_data_check.grid(row=7, column=0, sticky=E)

        self.score = Entry(line_frame,
                           width=65,
                           foreground="black",
                           background="white")
        self.score.grid(row=8, column=0)

        button_ok = Button(line_frame,
                           background='white',
                           text="Ок",
                           command=self.ok)
        button_ok.grid(row=9, column=1, sticky=EW)


if __name__ == '__main__':
    prepare = Prepare_Df(name_target_column_cuprum='Cu_AT502', name_target_column_cadmium='Cd_AT502')
    train_model = Train_Model(redis_connection=REDIS_CONNECTION)
    window = tk.Tk()

    ui = User_Interface(window=window, prepare_dataframe=prepare, train_models=train_model)
