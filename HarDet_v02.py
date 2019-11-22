import tkinter.ttk
from tkinter import*
import tkinter.filedialog as fd
from PIL import Image, ImageGrab
from tkinter import messagebox as mb
import CornerDetection
import io
import time
import DataAnalysis
import os


class Automat(tkinter.ttk.Frame):
    def __init__(self, master=None, parent=None):
        super().__init__(master)
        self.parent = parent
        self.pack()
        self.place_automat_widgets()
        self.master.title('Автоматический поиск решения')
        self.master.resizable(False, False)
        self.focus_set()

    def place_automat_widgets(self):
        k_start = DoubleVar()
        k_start.set(0.04)
        k_finish = DoubleVar()
        k_finish.set(0.08)
        k_step = DoubleVar()
        k_step.set(0.01)
        p_start = DoubleVar()
        p_start.set(0.005)
        p_finish = DoubleVar()
        p_finish.set(1)
        p_step = DoubleVar()
        p_step.set(0.005)
        epsilon_start = DoubleVar()
        epsilon_start.set(0.01)
        epsilon_finish = DoubleVar()
        epsilon_finish.set(5)
        epsilon_step = DoubleVar()
        epsilon_step.set(0.01)
        t_start = IntVar()
        t_start.set(5)
        t_finish = IntVar()
        t_finish.set(80)
        t_step = IntVar()
        t_step.set(5)
        b_start = DoubleVar()
        b_start.set(0.05)
        b_finish = DoubleVar()
        b_finish.set(0.8)
        b_step = DoubleVar()
        b_step.set(0.05)
        sigma_start = DoubleVar()
        sigma_start.set(0.005)
        sigma_finish = DoubleVar()
        sigma_finish.set(1)
        sigma_step = DoubleVar()
        sigma_step.set(0.005)
        h_start = DoubleVar()
        h_start.set(0.005)
        h_finish = DoubleVar()
        h_finish.set(1)
        h_step = DoubleVar()
        h_step.set(0.005)
        r_start = IntVar()
        r_start.set(3)
        r_finish = IntVar()
        r_finish.set(9)
        r_step = IntVar()
        r_step.set(1)
        self.master.geometry("460x430+400+115")
        mmenu = Menu(self.master)
        self.master.config(menu=mmenu)
        mmenu.add_command(label="Выйти", command=self.exit_automat)
        parametr_lbl = Label(self.master, text="Параметр")
        start_lbl = Label(self.master, text="Начальное значение")
        finish_lbl = Label(self.master, text="Конечное значение")
        step_lbl = Label(self.master, text="Шаг")
        k_lbl = Label(self.master, text="k")
        k_start_entr = Entry(self.master, width=10, textvariable=k_start)
        k_finish_entr = Entry(self.master, width=10, textvariable=k_finish)
        k_step_entr = Entry(self.master, width=10, textvariable=k_step)
        p_lbl = Label(self.master, text="p")
        p_start_entr = Entry(self.master, width=10, textvariable=p_start)
        p_finish_entr = Entry(self.master, width=10, textvariable=p_finish)
        p_step_entr = Entry(self.master, width=10, textvariable=p_step)
        epsilon_lbl = Label(self.master, text="epsilon")
        epsilon_start_entr = Entry(self.master, width=10, textvariable=epsilon_start)
        epsilon_finish_entr = Entry(self.master, width=10, textvariable=epsilon_finish)
        epsilon_step_entr = Entry(self.master, width=10, textvariable=epsilon_step)
        t_lbl = Label(self.master, text="t")
        t_start_entr = Entry(self.master, width=10, textvariable=t_start)
        t_finish_entr = Entry(self.master, width=10, textvariable=t_finish)
        t_step_entr = Entry(self.master, width=10, textvariable=t_step)
        b_lbl = Label(self.master, text="b")
        b_start_entr = Entry(self.master, width=10, textvariable=b_start)
        b_finish_entr = Entry(self.master, width=10, textvariable=b_finish)
        b_step_entr = Entry(self.master, width=10, textvariable=b_step)
        sigma_lbl = Label(self.master, text="sigma")
        sigma_start_entr = Entry(self.master, width=10, textvariable=sigma_start)
        sigma_finish_entr = Entry(self.master, width=10, textvariable=sigma_finish)
        sigma_step_entr = Entry(self.master, width=10, textvariable=sigma_step)
        h_lbl = Label(self.master, text="h")
        h_start_entr = Entry(self.master, width=10, textvariable=h_start)
        h_finish_entr = Entry(self.master, width=10, textvariable=h_finish)
        h_step_entr = Entry(self.master, width=10, textvariable=h_step)
        r_lbl = Label(self.master, text="r")
        r_start_entr = Entry(self.master, width=10, textvariable=r_start)
        r_finish_entr = Entry(self.master, width=10, textvariable=r_finish)
        r_step_entr = Entry(self.master, width=10, textvariable=r_step)
        parametr_lbl.place(x=20, y=10)
        start_lbl.place(x=100, y=10)
        finish_lbl.place(x=240, y=10)
        step_lbl.place(x=380, y=10)
        k_lbl.place(x=20, y=50)
        k_start_entr.place(x=100, y=50)
        k_finish_entr.place(x=240, y=50)
        k_step_entr.place(x=380, y=50)
        p_lbl.place(x=20, y=90)
        p_start_entr.place(x=100, y=90)
        p_finish_entr.place(x=240, y=90)
        p_step_entr.place(x=380, y=90)
        epsilon_lbl.place(x=20, y=130)
        epsilon_start_entr.place(x=100, y=130)
        epsilon_finish_entr.place(x=240, y=130)
        epsilon_step_entr.place(x=380, y=130)
        t_lbl.place(x=20, y=170)
        t_start_entr.place(x=100, y=170)
        t_finish_entr.place(x=240, y=170)
        t_step_entr.place(x=380, y=170)
        b_lbl.place(x=20, y=210)
        b_start_entr.place(x=100, y=210)
        b_finish_entr.place(x=240, y=210)
        b_step_entr.place(x=380, y=210)
        sigma_lbl.place(x=20, y=250)
        sigma_start_entr.place(x=100, y=250)
        sigma_finish_entr.place(x=240, y=250)
        sigma_step_entr.place(x=380, y=250)
        h_lbl.place(x=20, y=290)
        h_start_entr.place(x=100, y=290)
        h_finish_entr.place(x=240, y=290)
        h_step_entr.place(x=380, y=290)
        r_lbl.place(x=20, y=330)
        r_start_entr.place(x=100, y=330)
        r_finish_entr.place(x=240, y=330)
        r_step_entr.place(x=380, y=330)
        find_solution_btn = tkinter.ttk.Button(self.master, text="Запустить автоматическое выполнение", width=40,
                                               command=lambda: Automat.find_solution(k_start.get(), k_finish.get(),
                                                                                     p_start.get(),  p_finish.get(),
                                                                                     epsilon_start.get(),
                                                                                     epsilon_finish.get(),
                                                                                     t_start.get(), t_finish.get(),
                                                                                     b_start.get(), b_finish.get(),
                                                                                     sigma_start.get(),
                                                                                     sigma_finish.get(),
                                                                                     h_start.get(), h_finish.get(),
                                                                                     r_start.get(), r_finish.get(),
                                                                                     k_step.get(), p_step.get(),
                                                                                     epsilon_step.get(), t_step.get(),
                                                                                     b_step.get(), sigma_step.get(),
                                                                                     h_step.get(), r_step.get()))
        find_solution_btn.place(x=10, y=390)

    @staticmethod
    def find_solution(k_start, k_finish, p_start, p_finish, epsilon_start, epsilon_finish, t_start, t_finish,
                      b_start, b_finish, sigma_start, sigma_finish, h_start, h_finish, r_start, r_finish,
                      k_step, p_step, epsilon_step, t_step, b_step, sigma_step, h_step, r_step):
        kk = -1
        eepsilon = -1
        tt = -1
        bb = -1
        ssigma = -1
        hh = -1
        pp = -1
        # rr = -1
        opportunity_lst, parametr_lst = determine_opportunities()
        start_conditions = parametr_lst
        '''if opportunity_lst[7] or opportunity_lst[8]:
            rr = r_start'''
        for filename in files:
            print(filename)
            if opportunity_lst[4]:
                kk = k_start
            if opportunity_lst[1]:
                eepsilon = epsilon_start
            if opportunity_lst[9]:
                tt = t_start
            if opportunity_lst[11]:
                bb = b_start
            if opportunity_lst[12]:
                ssigma = sigma_start
            if opportunity_lst[3]:
                hh = h_start
            if opportunity_lst[6] or opportunity_lst[8]:
                pp = p_start
            img = Image.open(filename)
            img = img.convert("L")
            img.save('new_img.png')
            img_name = filename[filename.rfind('/') + 1:len(filename):1]
            answer_corners = Application.get_answer_corners(img_name)
            while (kk <= k_finish) or (kk == -1):
                while (pp <= p_finish) or (pp == -1):
                    while (eepsilon <= epsilon_finish) or (eepsilon == -1):
                        while (tt <= t_finish) or (tt == -1):
                            while (bb <= b_finish) or (bb == -1):
                                while (ssigma <= sigma_finish) or (ssigma == -1):
                                    while (hh <= h_finish) or (hh == -1):
                                        if opportunity_lst[7] or opportunity_lst[8]:
                                            rr = r_start
                                        else:
                                            rr = -1
                                        while (rr <= r_finish) or (rr == -1):
                                            print(kk)
                                            print(tt)
                                            start_time = time.time()
                                            ccorners = CornerDetection.find_corners('new_img.png', delete_points.get(),
                                                                                    tt,
                                                                                    significant_region.get(),
                                                                                    another_derivatives.get(),
                                                                                    multi_scale.get(), bb,
                                                                                    weight_func.get(),
                                                                                    eepsilon, response.get(), kk, pp,
                                                                                    hh, ssigma, threshold.get(), rr)
                                            finish_time = time.time()
                                            work_time = finish_time - start_time
                                            llist_ch, llist_corners = DataAnalysis.count_corners(answer_corners,
                                                                                                 ccorners)
                                            llist_ch.append(llist_ch[5] + llist_ch[1] + llist_ch[2])
                                            parametr_lst = [eepsilon, hh, kk, pp, rr, tt, bb, ssigma]
                                            DataAnalysis.save_data(img_name, opportunity_lst, parametr_lst, llist_ch,
                                                                   work_time)
                                            DataAnalysis.analyze_data(llist_ch[0], llist_ch[1], llist_ch[2],
                                                                      llist_ch[4], llist_ch[5])
                                            if rr != -1:
                                                rr += r_step
                                                if r_step == 0:
                                                    break
                                            else:
                                                break
                                        if hh != -1:
                                            hh += h_step
                                            '''if start_conditions[4] == -1:
                                                rr = -1
                                            else:
                                                rr = r_start'''
                                            if h_step == 0:
                                                break
                                        else:
                                            break
                                    if ssigma != -1:
                                        ssigma += sigma_step
                                        if start_conditions[1] == -1:
                                            hh = -1
                                        else:
                                            hh = h_start
                                        if sigma_step == 0:
                                            break
                                    else:
                                        break
                                if bb != -1:
                                    bb += b_step
                                    if start_conditions[7] == -1:
                                        ssigma = -1
                                    else:
                                        ssigma = sigma_start
                                    if b_step == 0:
                                        break
                                else:
                                    break
                            if tt != -1:
                                tt += t_step
                                if start_conditions[6] == -1:
                                    bb = -1
                                else:
                                    bb = b_start
                                if t_step == 0:
                                    break
                            else:
                                if start_conditions[6] == -1:
                                    bb = -1
                                else:
                                    bb = b_start
                                break
                        if eepsilon != -1:
                            eepsilon += epsilon_step
                            if start_conditions[5] == -1:
                                tt = -1
                            else:
                                tt = t_start
                            if epsilon_step == 0:
                                break
                        else:
                            if start_conditions[5] == -1:
                                tt = -1
                            else:
                                tt = t_start
                            break
                    if pp != -1:
                        pp += p_step
                        if start_conditions[0] == -1:
                            eepsilon = -1
                        else:
                            eepsilon = epsilon_start
                        if p_step == 0:
                            break
                    else:
                        break
                if kk != -1:
                    kk += k_step
                    if start_conditions[3] == -1:
                        pp = -1
                    else:
                        pp = p_start
                    if k_step == 0:
                        break
                else:
                    break
        mb.showinfo("", "Поиск завершён")

    def exit_automat(self):
        self.master.destroy()


# ================================================================                  главаная форма
class Application(tkinter.ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.center_window()
        self.create_widgets()
        self.master.title('Настраиваемый детектор Харриса')

    # располагает форму по центру
    def center_window(self):
        w = 1200
        hheight = 600
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) / 2
        y = (sh - hheight) / 2 - 45
        self.master.geometry('%dx%d+%d+%d' % (w, hheight, x, y))

    # создаёт виджеты на главной форме
    def create_widgets(self):

        # меню
        menubar = Menu(self)
        self.master.config(menu=menubar)
        menubar.add_command(label="Выбрать папку", command=self.func_open)
        menubar.add_command(label="Сделать снимок", command=self.func_picture_of_screen)
        menubar.add_command(label="Справка")
        menubar.add_command(label="Выйти", command=self.func_exit)

        # список глобальных переменных (для изменения местоположения) --- виджеты и некоторые переменные,
        # отвечающие за значения полей, флажков и регуляторов
        global automat_btn, weight_func_lbl, response_lbl, additional_lbl, \
            gaus_rbtn, bessel_rbtn, b_spline_rbtn, nonlinear_tensor_rbtn, \
            harris_response_rbtn, forstner_response_rbtn, \
            delete_points_chk, significant_region_chk, another_derivatives_chk, multi_scale_chk, \
            weight_func, response, another_derivatives, delete_points, significant_region, weight_func, multi_scale, \
            threshold_rbtn, non_maxima_rbtn, non_p_rbtn, threshold

        # кнопка "Найти углы"
        automat_btn = Button(text="Автоматический поиск решения", width=50, command=self.show_automat)

        # объявление меток
        weight_func_lbl = Label(self.master, text="Выбор весовой функции")
        response_lbl = Label(self.master, text="Выбор меры отклика")
        additional_lbl = Label(self.master, text="Дополнительные возможности")

        # объявление и инициализация переменных, отвечающих за флажки, регуляторы, переключатели и поля
        weight_func = IntVar()
        response = IntVar()
        threshold = IntVar()
        weight_func.set(0)
        response.set(0)
        delete_points = IntVar()
        significant_region = IntVar()
        another_derivatives = IntVar()
        multi_scale = IntVar()

        # объявление переключателей
        gaus_rbtn = Radiobutton(text="Функция Гаусса", variable=weight_func, value=0)
        bessel_rbtn = Radiobutton(text="Функция Бесселя", variable=weight_func, value=1)
        b_spline_rbtn = Radiobutton(text="B-сплайн", variable=weight_func, value=2)
        nonlinear_tensor_rbtn = Radiobutton(text="Нелинейный структурный тензор",
                                            variable=weight_func, value=3)
        threshold_rbtn = Radiobutton(text="Параметр p", variable=threshold, value=0)
        non_maxima_rbtn = Radiobutton(text="Подавление немаксимумов", variable=threshold, value=1)
        non_p_rbtn = Radiobutton(text="Подавление не максимумов с применением p", variable=threshold, value=2)
        harris_response_rbtn = Radiobutton(text="Мера отклика Харриса", variable=response, value=0)
        forstner_response_rbtn = Radiobutton(text="Мера отклика Фёрстнера", variable=response, value=1)

        # объявление флажков
        delete_points_chk = Checkbutton(self.master, text='Выполнить предварительное отсечение точек',
                                        variable=delete_points, onvalue=1, offvalue=0)
        significant_region_chk = Checkbutton(self.master, text='Детектировать углы на значимом регионе',
                                             variable=significant_region, onvalue=1, offvalue=0)
        another_derivatives_chk = Checkbutton(self.master, text='Ввести другие вторые производные',
                                              variable=another_derivatives, onvalue=1, offvalue=0)
        multi_scale_chk = Checkbutton(self.master, text='Применить многомасштабное Гауссово сглаживание',
                                      variable=multi_scale, onvalue=1, offvalue=0)

        # обработчик события --- изменение размеров главной формы
        self.bind("<Configure>", self.change_sizes)

    # размещает виджеты в зависимости от размеров главной формы
    def change_sizes(self, _):
        automat_btn.place(x=self.master.winfo_width() - 500, y=450)
        weight_func_lbl.place(x=self.master.winfo_width() - 550, y=10)
        gaus_rbtn.place(x=self.master.winfo_width() - 550, y=30)
        bessel_rbtn.place(x=self.master.winfo_width() - 550, y=50)
        b_spline_rbtn.place(x=self.master.winfo_width() - 550, y=70)
        nonlinear_tensor_rbtn.place(x=self.master.winfo_width() - 550, y=90)
        response_lbl.place(x=self.master.winfo_width() - 250, y=10)
        harris_response_rbtn.place(x=self.master.winfo_width() - 250, y=30)
        forstner_response_rbtn.place(x=self.master.winfo_width() - 250, y=60)
        threshold_rbtn.place(x=self.master.winfo_width() - 550, y=150)
        non_maxima_rbtn.place(x=self.master.winfo_width() - 550, y=170)
        non_p_rbtn.place(x=self.master.winfo_width() - 550, y=190)
        additional_lbl.place(x=self.master.winfo_width() - 550, y=240)
        delete_points_chk.place(x=self.master.winfo_width() - 550, y=260)
        significant_region_chk.place(x=self.master.winfo_width() - 550, y=280)
        another_derivatives_chk.place(x=self.master.winfo_width() - 550, y=300)
        multi_scale_chk.place(x=self.master.winfo_width() - 550, y=320)

    # осуществляет выход из приложения
    def func_exit(self):
        self.quit()

    # открывает папку и получает названия файлов в ней
    @staticmethod
    def func_open():
        global files
        files = []
        tree = []
        dirs = ['1 неправ не целиком', '1 неправ целиком', '1 прав не целиком', '1 прав целиком', '2 неправ не целиком',
                '2 неправ целиком', '2 прав не целиком', '2 прав целиком']
        dlg = fd.askdirectory()
        main_dir = dlg
        if main_dir != '':
            for i in os.walk(main_dir):
                tree.append(i)
            if len(tree) > 6:
                for i in range(3, len(tree) - 1):
                    for name in tree[i][2]:
                        files.append(tree[i][0] + '/' + name)
            else:
                for name in tree[0][2]:
                    files.append(tree[0][0] + '/' + name)

    # получает список известных координат углов
    @staticmethod
    def get_answer_corners(file_name):
        answer_corners = []
        fv = io.open("C:\\Users\\Настя\\PycharmProjects\\FigCreator_v02\\HarDet_TestData.dat", 'r')
        s = ""
        for line in fv:
            if file_name in line:
                s = line
                break
        fv.close()
        if s != '':
            for i in range(0, s.count('(')):
                x = float(s[s.find('(') + 1:s.find(','):1])
                y = float(s[s.find(',') + 1:s.find(')'):1])
                answer_corners.append([x, y])
                s = s[s.find(')') + 2:s.rfind(')') + 1:1]
        return answer_corners

    # делает снимок окна приложения
    def func_picture_of_screen(self):
        self.master.update_idletasks()
        x_r = self.master.winfo_x()
        y_r = self.master.winfo_y()
        x_l = x_r + self.master.winfo_width()
        y_l = y_r + self.master.winfo_height()
        picture = ImageGrab.grab(
            (x_r + 10, y_r, x_l, y_l + 45))
        picture.save("screen.png", "PNG")
        mb.showinfo("", "Снимок успешно сделан")
        return ()

    def show_automat(self):
        Automat(master=tkinter.Toplevel(), parent=self)


def determine_opportunities():
    parametr_lst = []
    opportunity_lst = []
    if weight_func.get() == 0:
        opportunity_lst.append(1)
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        parametr_lst.append(-1)
        parametr_lst.append(-1)
    elif weight_func.get() == 1:
        opportunity_lst.append(0)
        opportunity_lst.append(1)
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        parametr_lst.append(epsilon.get())
        parametr_lst.append(-1)
    elif weight_func.get() == 2:
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        opportunity_lst.append(1)
        opportunity_lst.append(0)
        parametr_lst.append(-1)
        parametr_lst.append(-1)
    else:
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        opportunity_lst.append(1)
        parametr_lst.append(-1)
        parametr_lst.append(h.get())
    if response.get() == 0:
        opportunity_lst.append(1)
        opportunity_lst.append(0)
        parametr_lst.append(k.get())
    else:
        opportunity_lst.append(0)
        opportunity_lst.append(1)
        parametr_lst.append(-1)
    if threshold.get() == 0:
        opportunity_lst.append(1)
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        parametr_lst.append(p.get())
        parametr_lst.append(-1)
    elif threshold.get() == 1:
        opportunity_lst.append(0)
        opportunity_lst.append(1)
        opportunity_lst.append(0)
        parametr_lst.append(-1)
        parametr_lst.append(r.get())
    else:
        opportunity_lst.append(0)
        opportunity_lst.append(0)
        opportunity_lst.append(1)
        parametr_lst.append(p.get())
        parametr_lst.append(r.get())
    if delete_points.get():
        opportunity_lst.append(1)
        parametr_lst.append(t.get())
    else:
        opportunity_lst.append(0)
        parametr_lst.append(-1)
    if significant_region.get():
        opportunity_lst.append(1)
    else:
        opportunity_lst.append(0)
    if another_derivatives.get():
        opportunity_lst.append(1)
        parametr_lst.append(b.get())
    else:
        opportunity_lst.append(0)
        parametr_lst.append(-1)
    if multi_scale.get():
        opportunity_lst.append(1)
        parametr_lst.append(sigma.get())
    else:
        opportunity_lst.append(0)
        parametr_lst.append(-1)
    return opportunity_lst, parametr_lst


# запускает приложение
def main():
    root = tkinter.Tk()
    app = Application(master=root)
    global p, k, t, b, epsilon, h, sigma, corners, r
    p = DoubleVar()
    k = DoubleVar()
    t = IntVar()
    b = DoubleVar()
    h = DoubleVar()
    sigma = DoubleVar()
    epsilon = DoubleVar()
    r = IntVar()
    p.set(0.005)
    k.set(0.04)
    t.set(60)
    b.set(0.05)
    epsilon.set(0.1)
    h.set(0)
    sigma.set(1)
    r.set(3)
    corners = []
    root.mainloop()


if __name__ == '__main__':
    main()
