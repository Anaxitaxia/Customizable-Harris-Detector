from PIL import Image, ImageDraw  # , ImageFilter
import numpy as np
import math
import scipy.special
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_gradient_magnitude


new_img_file_name = "new_img.png"


# дополняет изображение поддельными пикселями
# входные данные: имя изображения
def repeat_border_pixels(file_name):
    img = Image.open(file_name)
    new_img = Image.new(img.mode, (img.size[0] + 2, img.size[1] + 2))
    draw = ImageDraw.Draw(new_img)
    for i in range(img.size[0] + 2):
        for j in[0, img.size[1] - 1]:
            if (i == 0) or (i == 1):
                k = 0
            else:
                k = i - 2
            if j == 0:
                m = 0
            else:
                m = j + 2
            draw.point((i, m), fill=img.getpixel((k, j)))
    for i in range(1, img.size[1] + 1):
        for j in[0, img.size[0] - 1]:
            if j == 0:
                m = 0
            else:
                m = j + 2
            if i == 1:
                k = 0
            else:
                k = i - 1
            draw.point((m, i), fill=img.getpixel((j, k)))
    new_img.save(new_img_file_name)
    new_img = Image.open(new_img_file_name)
    new_img.paste(img, (1, 1))
    new_img.save(new_img_file_name)


# расчитывает яркость для изображения в пространстве RGB
# входные данные: цвет пикселя
# выходные данные: яркость пикселя
def get_brightness(clr):
    brightness = round(0.3 * clr[0] + 0.59 * clr[1] + 0.11 * clr[2])
    return brightness


# находит производные
# входные данные: флаг - показатель того, нужно ли вводить "другие" вторые производные;
# параметр b для "других" производных
# список intensity: intensity[0] --- яркость центрального пикселя --- I(x,y)
# intensity[1] --- I(x+1, y)
# intensity[2] --- I(x-1, y)
# intensity[3] --- I(x, y+1)
# intensity[4] --- I(x, y-1)
def find_derivatives(fl, b, intensity):
    derivatives = []
    d_x = abs(intensity[1] - intensity[2]) / 2
    d_y = abs(intensity[3] - intensity[4]) / 2
    d_x_2 = d_x * d_x
    d_y_2 = d_y * d_y
    # d_x_x = abs(intensity[1] + intensity[2] - 2 * intensity[0])
    # d_y_y = abs(intensity[3] + intensity[4] - 2 * intensity[0])
    if fl:
        # print(b)
        # print((b - 1) / 2)
        # print(d_x_2 + d_y_2)
        if (d_x_2 == 0) and (d_y_2 == 0):
            d_x = 0
        else:
            d_x = (d_x_2 + d_y_2) ** ((b - 1) / 2) * d_x
        if (d_x_2 == 0) and (d_y_2 == 0):
            d_y = 0
        else:
            d_y = (d_x_2 + d_y_2) ** ((b - 1) / 2) * d_y
        d_x_2 = d_x * d_x
        d_y_2 = d_y * d_y
        # d_x_x = (d_x_x + d_y_y) ** ((b - 1) / 2) * d_x
        # d_y_y = (d_x_x + d_y_y) ** ((b - 1) / 2) * d_y
    derivatives.append(d_x_2)
    derivatives.append(d_x * d_y)
    derivatives.append(d_y_2)

    return derivatives


# формирует массивы из производных первого порядка
# входные данные: имя изображения; флаг, отвечающий за ввод "других" производных; параметр b
# выходные данные: массив, содержащий (Ix)^2 --- производные изображения по X;
#                  массив, содержащий (Iy)^2 --- производные изображения по Y;
#                  массив, содержащий IxIy
def find_derivatives_arrays(file_name, flag, b):
    img = Image.open(file_name)
    arr_i_x_2 = np.zeros((img.size[0] - 2, img.size[1] - 2))
    arr_i_y_2 = np.zeros((img.size[0] - 2, img.size[1] - 2))
    arr_i_x_y = np.zeros((img.size[0] - 2, img.size[1] - 2))
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            # получение яркостей пикселей из окна размером 3x3
            intensity = [0, 0, 0, 0, 0]
            # получение яркостей чёрно-белого изображения
            intensity[0] = img.getpixel((i, j))
            intensity[1] = img.getpixel((i + 1, j))
            intensity[2] = img.getpixel((i - 1, j))
            intensity[3] = img.getpixel((i, j + 1))
            intensity[4] = img.getpixel((i, j - 1))

            # поиск производной для пикселя с координатами (i, j)
            derivatives = find_derivatives(flag, b, intensity)

            # формирование массивов
            arr_i_x_2[i - 1, j - 1] = derivatives[0]
            arr_i_x_y[i - 1, j - 1] = derivatives[1]
            arr_i_y_2[i - 1, j - 1] = derivatives[2]
    return arr_i_x_2, arr_i_x_y, arr_i_y_2


# выбирает фильтр
# входные данные: function --- значение от 0 до 3, в зависимости от желаемой весовой функции
def choose_filter(function, epsilon, fl):
    filt = np.zeros((3, 3))
    if function == 0:
        filt = np.array([[0.04, 0.12, 0.04], [0.12, 0.36, 0.12], [0.04, 0.12, 0.04]])
    elif function == 1:
        mu = 0.04 ** 0.5
        for i in range(filt.shape[0]):
            for j in range(filt.shape[1]):
                d = ((abs(1 - i)) ** 2 + (abs(1 - j)) ** 2) ** 0.5
                if d <= (epsilon / 2):
                    filt[i, j] = 1
                else:
                    filt[i, j] = scipy.special.k0(d / mu) / scipy.special.k0(epsilon / (2 * mu))
        # print(filt)
    elif function == 2:
        filt = 1 / 36 * np.array([[1, 4, 1], [4, 16, 4], [1, 4, 1]])
    else:
        fl = True
    return filt, fl


def make_nonlin_kernel(file_name, param_h, x, y):
    filt = np.zeros((3, 3))
    img = Image.open(file_name)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            try:
                filt[1 + i, 1 + j] = math.exp(-1 * (abs((1 + i) - (1 + j)) ** 2) / param_h *
                                              (img.getpixel((x + i, y + j)) - img.getpixel((x, y))) / param_h)
            except OverflowError:
                print(x + i, y + j)
                # print((abs((1 + i) - (1 + j)) ** 2) / param_h)
                # print((img.getpixel((x + i, y + j)) - img.getpixel((x, y))) / param_h)
    return filt


# находит элементы автокорреляционной матрицы
# входные данные: массивы производных i_x_2, i_x_y, i_y_2; координаты обрабатываемого пикселя; фильтр h;
# выходные данные: значения A,B, C
def find_abc(i_x_2, i_x_y, i_y_2, x, y, h, fl, file_name, param_h):
    a = 0
    b = 0
    c = 0
    if fl:
        h = make_nonlin_kernel(file_name, param_h, x, y)
    for i in[-1, 0, 1]:
        for j in[-1, 0, 1]:
            a += i_x_2[x + i, y + j] * h[i + 1, j + 1]
            b += i_y_2[x + i, y + j] * h[i + 1, j + 1]
            c += i_x_y[x + i, y + j] * h[i + 1, j + 1]
    return a, b, c


# вычисляет меру отклика Харриса
# входные данные: массивы производных i_x_2, i_x_y, i_y_2; пороговое значение k; фильтр h
# выходные данные: массив откликов
def harris_response(i_x_2, i_x_y, i_y_2, k, h, fl, file_name, param_h, obj_map):
    r = np.zeros((i_x_2.shape[0] - 2, i_x_2.shape[1] - 2))
    if fl:
        repeat_border_pixels(file_name)
    if obj_map.size == 0:
        for i in range(1, i_x_2.shape[0] - 1):
            for j in range(1, i_x_2.shape[1] - 1):
                a, b, c = find_abc(i_x_2, i_x_y, i_y_2, i, j, h, fl, new_img_file_name, param_h)
                m = np.array([[a, c], [c, b]])
                det_m = np.linalg.det(m)
                trace_m = a + b
                r[i - 1, j - 1] = det_m - k * trace_m * trace_m
    else:
        for i in range(1, i_x_2.shape[0] - 1):
            for j in range(1, i_x_2.shape[1] - 1):
                if obj_map[i - 1, j - 1]:
                    a, b, c = find_abc(i_x_2, i_x_y, i_y_2, i, j, h, fl, new_img_file_name, param_h)
                    m = np.array([[a, c], [c, b]])
                    det_m = np.linalg.det(m)
                    trace_m = a + b
                    r[i - 1, j - 1] = det_m - k * trace_m * trace_m
                else:
                    r[i - 1, j - 1] = 0
    return r


# вычисляет меру отклика Фёрстнера
# входные данные: массивы производных i_x_2, i_x_y, i_y_2; пороговое значение k; фильтр h
# выходные данные: массив откликов
def forstner_response(i_x_2, i_x_y, i_y_2, h, fl, file_name, param_h, obj_map):
    r = np.zeros((i_x_2.shape[0] - 2, i_x_2.shape[1] - 2))
    if fl:
        repeat_border_pixels(file_name)
    if obj_map.size == 0:
        for i in range(1, i_x_2.shape[0] - 1):
            for j in range(1, i_x_2.shape[1] - 1):
                a, b, c = find_abc(i_x_2, i_x_y, i_y_2, i, j, h, fl, new_img_file_name, param_h)
                m = np.array([[a, c], [c, b]])
                det_m = np.linalg.det(m)
                trace_m = a + b
                r[i - 1, j - 1] = det_m / (trace_m + 1e-20)
    else:
        for i in range(1, i_x_2.shape[0] - 1):
            for j in range(1, i_x_2.shape[1] - 1):
                if obj_map[i - 1, j - 1]:
                    a, b, c = find_abc(i_x_2, i_x_y, i_y_2, i, j, h, fl, new_img_file_name, param_h)
                    m = np.array([[a, c], [c, b]])
                    det_m = np.linalg.det(m)
                    trace_m = a + b
                    r[i - 1, j - 1] = det_m / (trace_m + 1e-20)
                else:
                    r[i - 1, j - 1] = 0
    return r


# отсекает значения, ниже порогового значения
# входные данные: параметр p, список откликов; список координат углов corners
# выходные данные: новый список откликов и список координат предварительных углов
def cout_small(p, list_r):
    new_list_r = []
    func_corners = []
    max_r = list_r.max()
    for i in range(list_r.shape[0]):
        for j in range(list_r.shape[1]):
            if list_r[i][j] > p * max_r:
                new_list_r.append(list_r[i][j])
                func_corners.append([i, j])
    return new_list_r, func_corners


# находит локальные максимумы
# входные данные: списки значений откликов и координат углов
# выходные данные: списки новых координат углов и откликов
def find_local_max(list_r, func_corners):
    new_corners = []
    new_r = []
    near_r = []
    result_r = []
    near_corners = []
    result_corners = []
    # print(func_corners)
    list_r.insert(0, 0)
    list_r.append(0)
    peaks = find_peaks(list_r)
    # print(peaks[0])
    for i in range(len(peaks[0])):
        new_corners.append(func_corners[peaks[0][i] - 1])
        new_r.append(list_r[peaks[0][i] - 1])
    for i in range(0, len(new_corners)):
        if i > 0:
            for n in[-1, 0, 1]:
                for m in[-1, 0, 1]:
                    if [new_corners[i][0] + n, new_corners[i][1] + m] in new_corners:
                        near_corners.append([new_corners[i][0] + n, new_corners[i][1] + m])
                        ind = new_corners.index([new_corners[i][0] + n, new_corners[i][1] + m])
                        near_r.append(new_r[ind])
        else:
            for n in[0, 1, 2]:
                for m in[0, 1, 2]:
                    if [new_corners[i][0] + n, new_corners[i][1] + m] in new_corners:
                        near_corners.append([new_corners[i][0] + n, new_corners[i][1] + m])
                        ind = new_corners.index([new_corners[i][0] + n, new_corners[i][1] + m])
                        near_r.append(new_r[ind])
        if near_corners[near_r.index(max(near_r))] not in result_corners:
            result_corners.append(near_corners[near_r.index(max(near_r))])
            result_r.append(max(near_r))
        near_r = []
        near_corners = []

    return result_corners, result_r


# осуществляет многомасштабное гауссово сглаживание
# входные данные: список откликов; параметр сигма
# выходные данные: список углов
'''def mult_gaus_smooth(r, sig):
    new_corners = []
    C = 0
    i = 1
    cornerness2 = r.sum()
    while C < sig:
    # for i in range(1, sig):
        cornerness1 = cornerness2
        r = gaussian_filter(r, sigma=i)
        i += 1
        cornerness2 = r.sum()
        C = cornerness2 / cornerness1
        print(cornerness1)
        print(cornerness2)
        print(C)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if r[i, j] > 0:
                new_corners.append([i, j])
            if r[i, j] < 0:
                print('!')
    return new_corners'''


# помечает углы на изображении
# входные данные: имя изображения; список координат найденных углов
# выходные данные: имя изображения с помеченными углами
def show_corners(file_name, func_corners, n):
    img = Image.open(file_name)
    # img2 = img.copy()
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            try:
                if func_corners.index([i - 1, j - 1]) >= 0:
                    img.paste(255, (i - 1, j - 1, i + 1, j + 1))
            except ValueError:
                continue
    img2 = img.crop([n, n, img.size[0] - n, img.size[1] - n])
    new_im_name = "result.png"
    img2.save(new_im_name)
    return new_im_name


def find_similar(i, j, t):
    img = Image.open(new_img_file_name)
    ch = 0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            if (x == 0) and (y == 0):
                continue
            delta = abs(img.getpixel((i, j)) - img.getpixel((i + x, j + y)))
            if delta <= t:
                ch += 1
    return ch


'''def find_suspect_corners(ch, i, j, flag, other_d_fl, b_th, file_name, param_h, response_fl, k):
    img = Image.open(new_img_file_name)
    arr_i_x_2 = np.zeros((3, 3))
    arr_i_y_2 = np.zeros((3, 3))
    arr_i_x_y = np.zeros((3, 3))
    if (ch >= 1) and (ch <= 6):
        corners.append([i - 1, j - 1])
        for n in (- 1, 0, 1):
            for m in (- 1, 0, 1):
                intensity = [0, 0, 0, 0, 0]
                intensity[0] = img.getpixel((i + n, j + m))
                intensity[1] = img.getpixel((i + n + 1, j + m))
                intensity[2] = img.getpixel((i + n - 1, j + m))
                intensity[3] = img.getpixel((i + n, j + m + 1))
                intensity[4] = img.getpixel((i + n, j + m - 1))
                derivatives = find_derivatives(other_d_fl, b_th, intensity)
                arr_i_x_2[m + 1, n + 1] = derivatives[0]
                arr_i_x_y[m + 1, n + 1] = derivatives[1]
                arr_i_y_2[m + 1, n + 1] = derivatives[2]
        a, b, c = find_abc(arr_i_x_2, arr_i_x_y, arr_i_y_2, 1, 1, filter_h, flag, file_name, param_h)
        if response_fl == 0:
            m = np.array([[a, c], [c, b]])
            det_m = np.linalg.det(m)
            trace_m = a + b
            r_list.append(det_m - k * trace_m * trace_m)
        elif response_fl == 1:
            m = np.array([[a, c], [c, b]])
            det_m = np.linalg.det(m)
            trace_m = a + b
            if trace_m != 0:
                r_list.append(det_m / (trace_m + 1e-20))
    return r_list, corners'''


# отсекает лишние точки
# входные данные: имя изображения; пороговое значение t
def cut_points(file_name, t, filter_flag, other_d_fl, b_th, response_fl, k, epsilon, param_h, obj_map):
    repeat_border_pixels(file_name)
    img = Image.open(new_img_file_name)
    r_list = []
    corners = []
    list_ch = []
    arr_i_x_2 = np.zeros((3, 3))
    arr_i_y_2 = np.zeros((3, 3))
    arr_i_x_y = np.zeros((3, 3))
    flag = False
    # выбор фильтра
    filter_h, flag = choose_filter(filter_flag, epsilon, flag)
    if obj_map.size == 0:
        for i in range(1, img.size[0] - 2):
            for j in range(1, img.size[1] - 2):
                ch = 0
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if (x == 0) and (y == 0):
                            continue
                        delta = abs(img.getpixel((i, j)) - img.getpixel((i + x, j + y)))
                        if delta <= t:
                            ch += 1
                if (ch >= 1) and (ch <= 6):
                    list_ch.append([i, j])
                    corners.append([i - 1, j - 1])
                    for n in (- 1, 0, 1):
                        for m in (- 1, 0, 1):
                            intensity = [0, 0, 0, 0, 0]
                            intensity[0] = img.getpixel((i + n, j + m))
                            intensity[1] = img.getpixel((i + n + 1, j + m))
                            intensity[2] = img.getpixel((i + n - 1, j + m))
                            intensity[3] = img.getpixel((i + n, j + m + 1))
                            intensity[4] = img.getpixel((i + n, j + m - 1))
                            derivatives = find_derivatives(other_d_fl, b_th, intensity)
                            arr_i_x_2[m + 1, n + 1] = derivatives[0]
                            arr_i_x_y[m + 1, n + 1] = derivatives[1]
                            arr_i_y_2[m + 1, n + 1] = derivatives[2]
                    a, b, c = find_abc(arr_i_x_2, arr_i_x_y, arr_i_y_2, 1, 1, filter_h, flag, file_name, param_h)
                    if response_fl == 0:
                        m = np.array([[a, c], [c, b]])
                        det_m = np.linalg.det(m)
                        trace_m = a + b
                        r_list.append(det_m - k * trace_m * trace_m)
                    elif response_fl == 1:
                        m = np.array([[a, c], [c, b]])
                        det_m = np.linalg.det(m)
                        trace_m = a + b
                        r_list.append(det_m / (trace_m + 1e-20))
    else:
        for i in range(1, img.size[0] - 2):
            for j in range(1, img.size[1] - 2):
                if obj_map[i, j]:
                    ch = 0
                    for x in [-1, 0, 1]:
                        for y in [-1, 0, 1]:
                            if (x == 0) and (y == 0):
                                continue
                            delta = abs(img.getpixel((i, j)) - img.getpixel((i + x, j + y)))
                            if delta <= t:
                                ch += 1
                    if (ch >= 1) and (ch <= 6):
                        list_ch.append([i, j])
                    corners.append([i - 1, j - 1])
                    for n in (- 1, 0, 1):
                        for m in (- 1, 0, 1):
                            intensity = [0, 0, 0, 0, 0]
                            intensity[0] = img.getpixel((i + n, j + m))
                            intensity[1] = img.getpixel((i + n + 1, j + m))
                            intensity[2] = img.getpixel((i + n - 1, j + m))
                            intensity[3] = img.getpixel((i + n, j + m + 1))
                            intensity[4] = img.getpixel((i + n, j + m - 1))
                            derivatives = find_derivatives(other_d_fl, b_th, intensity)
                            arr_i_x_2[m + 1, n + 1] = derivatives[0]
                            arr_i_x_y[m + 1, n + 1] = derivatives[1]
                            arr_i_y_2[m + 1, n + 1] = derivatives[2]
                    a, b, c = find_abc(arr_i_x_2, arr_i_x_y, arr_i_y_2, 1, 1, filter_h, flag, file_name, param_h)
                    if response_fl == 0:
                        m = np.array([[a, c], [c, b]])
                        det_m = np.linalg.det(m)
                        trace_m = a + b
                        r_list.append(det_m - k * trace_m * trace_m)
                    elif response_fl == 1:
                        m = np.array([[a, c], [c, b]])
                        det_m = np.linalg.det(m)
                        trace_m = a + b
                        r_list.append(det_m / (trace_m + 1e-20))
    return r_list, corners


def convert_to_polar(x, y):
    magnitude = np.zeros(x.shape)
    angle = np.zeros(x.shape)
    # print(type(magnitude))
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            magnitude[i, j] = (x[i, j] * x[i, j] + y[i, j] * y[i, j]) ** 0.5
            angle[i, j] = math.atan2(y[i, j], x[i, j])
    return magnitude, angle


def convert_to_decart(magnitude, angle):
    x = np.zeros(magnitude.shape)
    y = np.zeros(magnitude.shape)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            x[i, j] = magnitude[i, j] * np.cos(angle[i, j])
            y[i, j] = magnitude[i, j] * np.sin(angle[i, j])
    return x, y


# находит значимый регион
def find_significant_region(file_name):
    # filter_h = np.array([[0.04, 0.12, 0.04], [0.12, 0.36, 0.12], [0.04, 0.12, 0.04]])
    img = Image.open(file_name)
    channel = np.zeros(img.size)
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            channel[i, j] = img.getpixel((i, j))
    f_i = np.fft.fft2(channel)
    # print(f_i[0, 0])
    # print(f_i.ndim)
    magnitude, angle = convert_to_polar(np.real(f_i), np.imag(f_i))
    # print(magnitude)
    # print(angle)
    l_f = np.log10(magnitude.clip(min=1e-9))
    # print(l_f)
    h = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    h_l = convolve(l_f, h)
    # print(h_l)
    r_p = np.exp(l_f - h_l)
    # print(r_p)
    imag_part, real_part = convert_to_decart(r_p, angle)
    # print(imag_part)
    # print(real_part)
    img_combined = np.fft.ifft2(real_part + 1j * imag_part)
    s_f, _ = convert_to_polar(np.real(img_combined), np.imag(img_combined))
    s_f = gaussian_gradient_magnitude(s_f, (8, 0))
    # img2 = img.copy()
    # print(s_f)
    s_f = s_f ** 2
    s_f = np.float32(s_f) / np.max(s_f)
    s_f = np.flipud(s_f)
    s_f = np.fliplr(s_f)
    th = np.mean(s_f)
    o_f = np.zeros(s_f.shape)
    for i in range(s_f.shape[0]):
        for j in range(s_f.shape[1]):
            if s_f[i, j] > th:
                o_f[i, j] = 1
            else:
                o_f[i, j] = 0
    o_f = scipy.ndimage.binary_erosion(o_f).astype(o_f.dtype)
    o_f = scipy.ndimage.binary_dilation(o_f).astype(o_f.dtype)
    '''for i in range(s_f.shape[0]):
        for j in range(s_f.shape[1]):
            if o_f[i, j] == 0:
                img2.putpixel((i, j), 0)
            else:
                img2.putpixel((i, j), 255)
    img2.save(new_img_file_name)'''
    return o_f


def image_block(r, list_r):
    i = 0
    new_list_r = []
    new_corners = []
    while i < list_r.shape[0]:
        j = 0
        while j < list_r.shape[1]:
            block = list_r[i:i + r, j:j + r]
            x, y = np.unravel_index(block.argmax(), block.shape)
            if block[x, y] != 0:
                new_list_r.append(block[x, y])
                x += i
                y += j
                new_corners.append([x, y])
            else:
                x += i
                y += j
            j += r
        i += r
    return new_corners, new_list_r


def gaussian_blurring(sig):
    img = Image.open(new_img_file_name)
    if sig != 0:
        img_mass = np.zeros((img.size[0] - 2, img.size[1] - 2))
        for i in range(1, img.size[0] - 1):
            for j in range(1, img.size[1] - 1):
                summa = 0
                for n in (-1, 0, 1):
                    for m in (-1, 0, 1):
                        d2 = n ** 2 + m ** 2
                        summa += \
                            img.getpixel((i + n, j + m)) * np.exp(-d2 / (2 * sig * sig)) / (np.sqrt(2 * np.pi) * sig)
                img_mass[i - 1, j - 1] = summa
    else:
        img_mass = np.asarray(img)
    return img_mass


# находит углы (основная функция)
# входные параметры: имя изображения; флаг, отвечающий за отсечение лишних точек; пороговое значение t;
# флаг, отвечающий за ввод "других" вторых производных; параметр b для "других" вторых производных;
# флаг, отвечающий за выбор фильтра; флаг, отвечающий за выбор отклика; пороговое значение k;
# пороговое значение p;
def find_corners(file_name, cut_flag, t, significant, other_derivatives_fl, multi_scale, b,
                 filter_flag, epsilon, response_flag, k, p, param_h, sig, threshold, r):
    list_r = []
    obj_map = np.zeros(0)
    if significant:
        obj_map = find_significant_region(file_name)
    if cut_flag:
        list_r, corners = cut_points(file_name, t, filter_flag, other_derivatives_fl, b,
                                     response_flag, k, epsilon, param_h, obj_map)
        new_corners = []
        new_list_r = []
        max_r = max(list_r)
        # print(list_r)
        # print(max(list_r))
        '''if multi_scale:
            r = gaussian_filter(new_list_r, sigma=sig)
            for i in range(len(r)):
                if r[i] > 0:
                    corners.append(new_corners[i])'''
        if threshold == 0:
            for i in range(len(list_r)):
                if list_r[i] > p * max_r:
                    new_corners.append(corners[i])
                    new_list_r.append(list_r[i])
            corners, new_list_r = find_local_max(new_list_r, new_corners)
        elif threshold == 1:
            img = Image.open(file_name)
            block_r_lst = np.zeros(img.size)
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    if [i, j] in corners:
                        block_r_lst[i, j] = list_r[corners.index([i, j])]
            corners, new_list_r = image_block(r, block_r_lst)
            corners, new_list_r = find_local_max(new_list_r, corners)
        else:
            img = Image.open(file_name)
            block_r_lst = np.zeros(img.size)
            if obj_map.size == 0:
                for i in range(img.size[0]):
                    for j in range(img.size[1]):
                        if [i, j] in corners:
                            block_r_lst[i, j] = list_r[corners.index([i, j])]
                        else:
                            block_r_lst[i, j] = 0
            else:
                for i in range(img.size[0]):
                    for j in range(img.size[1]):
                        if obj_map[i, j]:
                            if [i, j] in corners:
                                block_r_lst[i, j] = list_r[corners.index([i, j])]
                            else:
                                block_r_lst[i, j] = 0
            corners, new_list_r = image_block(r, block_r_lst)
            r_max = max(new_list_r)
            real_list_r = []
            new_corners = []
            for i in range(len(new_list_r)):
                if new_list_r[i] > p * r_max:
                    real_list_r.append(new_list_r[i])
                    new_corners.append(corners[i])
            corners, new_list_r = find_local_max(real_list_r, new_corners)
    else:
        # img_mass = np.zeros((img.size[0], img.size[1]))
        # формирование нового изображения с "поддельными" краевыми пикселями
        repeat_border_pixels(file_name)
        repeat_border_pixels(new_img_file_name)
        i_x_2, i_x_y, i_y_2 = find_derivatives_arrays(new_img_file_name, other_derivatives_fl, b)

        # выбор фильтра
        filt_flag = False
        filter_h, filt_flag = choose_filter(filter_flag, epsilon, filt_flag)

        # расчёт меры отклика для всех пикселей изображения
        if response_flag == 0:
            list_r = harris_response(i_x_2, i_x_y, i_y_2, k, filter_h, filt_flag, file_name, param_h, obj_map)
        elif response_flag == 1:
            list_r = forstner_response(i_x_2, i_x_y, i_y_2, filter_h, filt_flag, file_name, param_h, obj_map)

            '''if multi_scale:
                corners = mult_gaus_smooth(list_r, sig)
            else:'''
        if threshold == 0:
            new_list_r, corners = cout_small(p, list_r)
            corners, new_list_r = find_local_max(new_list_r, corners)
        elif threshold == 1:
            corners, new_list_r = image_block(r, list_r)
            corners, new_list_r = find_local_max(new_list_r, corners)
        else:
             corners, new_list_r = image_block(r, list_r)
             r_max = max(new_list_r)
             real_list_r = []
             new_corners = []

             for i in range(len(new_list_r)):
                if new_list_r[i] > p * r_max:
                    real_list_r.append(new_list_r[i])
                    new_corners.append(corners[i])
             corners, new_list_r = find_local_max(real_list_r, new_corners)

            # new_name = show_corners(new_img_file_name, corners, 2)
        '''else:
            c_th = 0
            cornerness2 = 0
            siggma = 0.4
            while c_th < sig:
                repeat_border_pixels(file_name)
                repeat_border_pixels(new_img_file_name)
                cornerness1 = cornerness2
                siggma += 0.1
                img_mass = gaussian_blurring(siggma)
                i_x_2, i_x_y, i_y_2 = find_derivatives_arrays(new_img_file_name, other_derivatives_fl, b, img_mass)

                # выбор фильтра
                filt_flag = False
                filter_h, filt_flag = choose_filter(filter_flag, epsilon, filt_flag)

                # расчёт меры отклика для всех пикселей изображения
                if response_flag == 0:
                    list_r = harris_response(i_x_2, i_x_y, i_y_2, k, filter_h, filt_flag, file_name, param_h, obj_map)
                elif response_flag == 1:
                    list_r = forstner_response(i_x_2, i_x_y, i_y_2, filter_h, filt_flag, file_name, param_h, obj_map)
                if threshold == 0:
                    new_list_r, corners = cout_small(p, list_r)
                    corners, new_list_r = find_local_max(new_list_r, corners)
                elif threshold == 1:
                    corners, new_list_r = image_block(r, list_r)
                    corners, new_list_r = find_local_max(new_list_r, corners)
                else:
                    corners, new_list_r = image_block(r, list_r)
                    r_max = max(new_list_r)
                    real_list_r = []
                    new_corners = []
                    for i in range(len(new_list_r)):
                        if new_list_r[i] > p * r_max:
                            real_list_r.append(new_list_r[i])
                            new_corners.append(corners[i])
                    corners, new_list_r = find_local_max(real_list_r, new_corners)
                cornerness2 = np.mean(new_list_r)
                c_th = cornerness1 / cornerness2
                print(cornerness1)
                print(cornerness2)
                print(len(new_list_r))
                print(c_th)'''
    return corners  # new_name, corners
