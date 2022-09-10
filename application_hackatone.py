import os
from tkinter import *
from hackaton import Zinc_Impurities


def clicked():
    lbl.configure(text='Вы успешно отправили документ в обработку')
    
    zi = Zinc_Impurities(txt.get(), txt_2.get())
    zi.deleting_incorrect_data()
    zi.removing_redundant_data()
    zi.download_weights()
    zi.prediction()


if __name__ == '__main__':
    window = Tk()
    window.title("Анализ осаждения меди и кадмия")

    lbl = Label(window, text="Пожалуйста, введите полный путь до файла с данными для анализа")
    lbl.grid(column=0, row=0)

    txt = Entry(window, width=100)
    txt.grid(column=0, row=1)

    lbl = Label(window, text="Пожалуйста, введите полный путь до заполняемого файла")
    lbl.grid(column=0, row=2)

    txt_2 = Entry(window, width=100)
    txt_2.grid(column=0, row=3)

    btn = Button(window, text="Запуск анализа", command=clicked)
    btn.grid(column=0, row=4)

    window.geometry('800x200')

    window.mainloop()
