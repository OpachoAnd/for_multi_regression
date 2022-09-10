from tkinter import *
#from hackaton import *


def clicked():
    lbl.configure(text='Вы успешно отправили документ в обработку')
    zi = Zinc_Impurities(txt.get())


if __name__ == '__main__':
    window = Tk()
    window.title("Анализ осаждения меди и кадмия")

    lbl = Label(window, text="Пожалуйста, введите полный путь до файла с данными для анализа")
    lbl.grid(column=0, row=10)

    txt = Entry(window, width=50)
    txt.grid(column=0, row=1)

    btn = Button(window, text="Запуск анализа", command=clicked)
    btn.grid(column=0, row=12)

    window.geometry('400x80')

    window.mainloop()
