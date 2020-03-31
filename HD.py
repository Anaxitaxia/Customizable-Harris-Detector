import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_gradient_magnitude
import scipy.special
from numpy import linalg as la
import time


class HarrisDetector:
    # im --- массив numpy
    def __init__(self, im, wind_func='gauss', sigma=1, wind_size=3, response='harris',
                 k=0.04, th_politic='adapt', p=0.005, non_maxima_fl=True, maxima_wind_size=3, log_log=False, b=0.1,
                 cut_fl=False, cut_th=200, significant_fl=False):
        super().__init__()
        self.im = im
        self.wind_func = wind_func
        if self.wind_func == 'gauss':
            self.sigma = sigma
        self.wind_size = wind_size
        self.response = response
        if self.response == 'harris':
            self.k = k
        self.th_politic = th_politic
        self.p = p
        self.non_maxima_fl = non_maxima_fl
        if self.non_maxima_fl:
            self.n = maxima_wind_size
        self.log_log = log_log
        if self.log_log:
            self.b = b
        self.cut_fl = cut_fl
        if self.cut_fl:
            self.cut_th = cut_th
        self.significant_fl = significant_fl

    def create_windows(self):
        if self.wind_func == 'gauss':
            x_prewitt = np.ones((self.wind_size, self.wind_size))
            x_prewitt[:, 0] -= 2
            x_prewitt[:, 1] -= 1
            y_prewitt = np.ones((self.wind_size, self.wind_size))
            y_prewitt[0, :] -= 2
            y_prewitt[1, :] -= 1
            gaussian_wind = np.exp(-(x_prewitt ** 2 + y_prewitt ** 2) / (2 * self.sigma ** 2))
            gaussian_x = x_prewitt * np.exp(-(x_prewitt ** 2 + y_prewitt ** 2) / (2 * self.sigma ** 2))
            gaussian_y = y_prewitt * np.exp(-(x_prewitt ** 2 + y_prewitt ** 2) / (2 * self.sigma ** 2))
            return gaussian_wind, gaussian_x, gaussian_y
        elif self.wind_func == 'b-spline':
            spline = 1 / 36 * np.array([[1, 4, 1], [4, 16, 4], [1, 4, 1]])
            m_x = 1 / 12 * np.array([[1, 4, 1], [0, 0, 0], [-1, -4, -1]])
            m_y = 1 / 12 * np.array([[1, 0, -1], [4, 0, -4], [1, 0, -1]])
            return spline, m_x, m_y

    def find_derivatives(self, d_x, d_y):
        i_x = convolve(self.im, d_x, mode='constant', cval=0.0)
        i_y = convolve(self.im, d_y, mode='constant', cval=0.0)
        i_x2 = i_x ** 2
        i_y2 = i_y ** 2
        if self.log_log:
            i_x = (i_x2 + i_y2 + 1e-5) ** (0.5 * (self.b - 1)) * i_x
            i_y = (i_x2 + i_y2 + 1e-5) ** (0.5 * (self.b - 1)) * i_y
            i_x2 = i_x ** 2
            i_y2 = i_y ** 2
        i_xy = i_x * i_y
        return i_x2, i_y2, i_xy

    def find_local_max(self, r_map):
        # напрасное вычисление максимума, если пороговое значение не адаптивное
        max_matrix = np.copy(r_map)
        maximum = -1000

        for i in range(0, r_map.shape[0], self.n):
            for j in range(0, r_map.shape[1], self.n):
                block = r_map[i:i + self.n, j:j + self.n]
                sub = np.reshape(block, block.size)
                max_matrix[i:i + self.n, j:j + self.n] = max(sub)
                if max(sub) > maximum:
                    maximum = max(sub)

        corner_map = max_matrix == r_map
        return corner_map, maximum

    def cut_points(self):
        def find_diff(item):
            if abs(int(item) - int(block[1, 1])) < self.cut_th:
                return 1
            else:
                return 0

        new_im = np.zeros(self.im.shape)
        for i in range(0, self.im.shape[0], 3):
            for j in range(0, self.im.shape[1], 3):
                block = self.im[i:i + 3, j:j + 3]
                block_vect = np.reshape(block, block.size)
                n_similar = sum(list(map(find_diff, block_vect)))
                if 1 <= n_similar <= 6:
                    new_im[i:i + 3, j:j + 3] = block
        self.im = new_im

    def find_significant(self):
        def convert_to_polar(x_coord, y_coord):
            magnitude = np.sqrt(x_coord ** 2 + y_coord ** 2)
            angle = np.arctan2(x_coord, y_coord)
            return magnitude, angle

        def convert_to_decart(magnitude, angle):
            x_coord = magnitude * np.cos(angle)
            y_coord = magnitude * np.sin(angle)
            return x_coord, y_coord

        f_i = np.fft.fft2(self.im)
        rho, phi = convert_to_polar(np.real(f_i), np.imag(f_i))
        l_f = np.log10(rho.clip(min=1e-9))
        h = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        h_l = convolve(l_f, h)
        r_p = np.exp(l_f - h_l)
        imag_part, real_part = convert_to_decart(r_p, phi)
        img_combined = np.fft.ifft2(real_part + 1j * imag_part)
        s_f, _ = convert_to_polar(np.real(img_combined), np.imag(img_combined))
        s_f = gaussian_gradient_magnitude(s_f, (8, 0))
        s_f = s_f ** 2
        s_f = np.float32(s_f) / np.max(s_f)
        # s_f = np.flipud(s_f)
        # s_f = np.fliplr(s_f)
        th = 3 * np.mean(s_f)
        o_f = np.where(s_f > th, 1, 0)
        o_f = scipy.ndimage.binary_dilation(o_f).astype(o_f.dtype)
        o_f = scipy.ndimage.binary_erosion(o_f).astype(o_f.dtype)
        return o_f

    def find_corners(self):
        def find_matrix_values(conv_wind, wind_x2, wind_y2, wind_xy):
            a = convolve(wind_x2, conv_wind, mode='constant', cval=0.0)
            b = convolve(wind_y2, conv_wind, mode='constant', cval=0.0)
            c = convolve(wind_xy, conv_wind, mode='constant', cval=0.0)
            return a, b, c

        def create_matrix(a, b, c):
            m = np.array([[a, c], [c, b]])
            _, v = la.eig(m)
            return np.array(v)

        def find_harris_response(a, b, c):
            return float(a) * float(b) - float(c) ** 2 - self.k * (float(a) + float(b)) ** 2

        def find_forstner_response(m):
            return la.det(m) / (np.trace(m) + 1e-5)

        d_xy, d_x, d_y = self.create_windows()
        if self.cut_fl:
            self.cut_points()
        if self.significant_fl:
            obj_map = self.find_significant()
            self.im = self.im * obj_map
        i_x2, i_y2, i_xy = self.find_derivatives(d_x, d_y)
        mas_a, mas_b, mas_c = find_matrix_values(d_xy, i_x2, i_y2, i_xy)
        mas_a = mas_a.reshape(mas_a.size)
        mas_b = mas_b.reshape(mas_a.size)
        mas_c = mas_c.reshape(mas_a.size)
        response_map = np.array([])
        if self.response == 'harris':
            response_map = np.array(list(map(find_harris_response, mas_a, mas_b, mas_c))).reshape(self.im.shape)
        elif self.response == 'forstner':
            matrix = np.array(list(map(create_matrix, mas_a, mas_b, mas_c)))
            response_map = np.array(list(map(find_forstner_response, matrix))).reshape(self.im.shape)

        corner_map = np.ones(self.im.shape)
        if self.non_maxima_fl:
            tstart = time.time()
            corner_map, maximum = self.find_local_max(response_map)
            # HD_cy.find_local_max(response_map, self.n)  # self.find_local_max(response_map)
            tfinish = time.time()
            print('Поиск локальных максимумов, сек: ', tfinish - tstart)
            if self.th_politic == 'adapt':
                response_map = np.where(response_map > self.p * maximum, response_map, 0)
            else:
                response_map = np.where(response_map > self.p, response_map, 0)
        else:
            if self.th_politic == 'adapt':
                maximum = np.amax(response_map)
                response_map = np.where(response_map > self.p * maximum, response_map, 0)
            else:
                response_map = np.where(response_map > self.p, response_map, 0)
        corner_map = corner_map * response_map
        x_corner, y_corner = np.where(corner_map != 0)
        return x_corner, y_corner, corner_map


'''
Пример использования
image = Image.open('C://Users/Настя/PycharmProjects/FigCreator_v02/2 прав целиком/2019-8-17_8_39_7_4.png').convert('L')
image.load()
img = np.array(image)

HD = HarrisDetector(img, p=0.015,)
tstart = time.time()
y, x, c_map = HD.find_corners()
tfinish = time.time()
print('Время на поиск углов, сек.: ', tfinish - tstart)

for index in range(x.shape[0]):
    cv.circle(img, (x[index], y[index]), 5, 0, 2)
image = Image.fromarray(img.astype('uint8'), 'L')
image.show()
'''
