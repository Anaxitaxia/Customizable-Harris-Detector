import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import time
import cv2 as cv


class HarrisDetector:
    # im --- массив numpy
    def __init__(self, im, wind_func='gauss', sigma=1, wind_size=3, response='harris',
                 k=0.04, th_politic='adapt', p=0.005, non_maxima_fl=True, maxima_wind_size=3, log_log=False, b=None,
                 cut_fl=False, cut_th=None):
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
            if not self.b:
                print(f"{self.b} is not correct; number is expected")
                raise NameError
        self.cut_fl = cut_fl
        if self.cut_fl:
            self.cut_th = cut_th
            if not self.cut_th:
                print(f"{self.cut_th} is not correct; number is expected")
                raise NameError

    def create_windows(self):
        if self.wind_func == 'gauss':
            # а надо быстрее?
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
                if 1 <= n_similar <= 7:
                    new_im[i:i + 3, j:j + 3] = block
        self.im = new_im

    def find_corners(self):
        def find_matrix_values(conv_wind, wind_x2, wind_y2, wind_xy):
            a = convolve(wind_x2, conv_wind, mode='constant', cval=0.0)
            b = convolve(wind_y2, conv_wind, mode='constant', cval=0.0)
            c = convolve(wind_xy, conv_wind, mode='constant', cval=0.0)
            return a, b, c

        def create_matrix(a, b, c):
            return np.array([[a, c], [c, b]])

        def find_response(m):
            return np.linalg.det(m) - self.k * (np.trace(m) ** 2)

        d_xy, d_x, d_y = self.create_windows()

        if self.cut_fl:
            self.cut_points()
        i_x2, i_y2, i_xy = self.find_derivatives(d_x, d_y)
        response_map = np.ones(self.im.shape)
        if self.response == 'harris':
            mas_a, mas_b, mas_c = find_matrix_values(d_xy, i_x2, i_y2, i_xy)
            mas_a = mas_a.reshape(mas_a.size)
            mas_b = mas_b.reshape(mas_a.size)
            mas_c = mas_c.reshape(mas_a.size)
            matrix = np.array(list(map(create_matrix, mas_a, mas_b, mas_c)))
            response_map = np.array(list(map(find_response, matrix))).reshape(self.im.shape)
        maximum = 1
        corner_map = np.array([])
        if self.non_maxima_fl:
            corner_map, maximum = self.find_local_max(response_map)
        response_map = np.where(response_map > self.p * maximum, response_map, 0)
        corner_map = corner_map * response_map
        x_corner, y_corner = np.where(corner_map != 0)
        return x_corner, y_corner, corner_map


image = Image.open('C://Users/Настя/PycharmProjects/FigCreator_v02/2 прав целиком/2019-8-17_8_39_7_4.png').convert('L')
image.load()
img = np.array(image)

HD = HarrisDetector(img, k=0.06, p=0.015, maxima_wind_size=7, cut_fl=True, cut_th=50)
tstart = time.time()
y, x, c_map = HD.find_corners()
tfinish = time.time()
print('Время на поиск углов, сек.: ', tfinish - tstart)

for index in range(x.shape[0]):
    cv.circle(img, (x[index], y[index]), 5, 0, 2)
image = Image.fromarray(img.astype('uint8'), 'L')
image.show()
