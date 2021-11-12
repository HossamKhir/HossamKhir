#! /usr/bin/env python3
"""
"""

import numpy as np


class Image:
    """
    A class that could be convoluted,
    """

    def __init__(self, *args, **kwargs) -> None:
        """ """
        if type(args[0]) == np.ndarray:
            self.matrix = args[0]
        elif type(args[0]) == list:
            self.matrix = np.array(args[0], **kwargs)
        else:
            raise ValueError
        self.stride = kwargs.get("stride", (1, 1))
        self.padding = kwargs.get("padding", (0, 0))
        if self.matrix.ndim == 2:
            self.matrix = self.matrix.reshape((1, *self.matrix.shape))

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return repr(self.matrix)

    @property
    def stride(self):
        """ """
        return self._stride

    @property
    def padding(self):
        """ """
        return self._padding

    @property
    def shape(self):
        """ """
        return self.matrix.shape

    @stride.setter
    def stride(self, stride):
        if type(stride) == int:
            stride = (stride, stride)
        self._stride = stride

    @padding.setter
    def padding(self, pad):
        if type(pad) == int:
            pad = (pad, pad)
        self._padding = pad

    def convolute(self, kernel):
        """
        general convolution method
        """

        def step_convolute(matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            """ """
            return np.sum(matrix * kernel, axis=(1, 2))

        c, ih, iw = self.matrix.shape
        _, kh, kw = kernel.matrix.shape
        ph, pw = kernel.padding
        sh, sw = kernel.stride
        h = (ih - kh + ph + sh) // sh
        w = (iw - kw + pw + sw) // sw
        result = np.zeros((c, h, w))
        if ph or pw:
            img = np.zeros((c, ih + 2 * ph, iw + 2 * pw))
            img[:, ph:-ph, pw:-pw] = self.matrix.copy()
        else:
            img = self.matrix.copy()
        for row in range(0, h):
            for col in range(0, w):
                stepr, stepc = row * sh, col * sw
                step_matrix = img[:, stepr : stepr + kh, stepc : stepc + kw]
                result[:, row, col] = step_convolute(step_matrix, kernel.matrix)
        return result

    def __mul__(self, other):
        """ """
        return self.convolute(other)

    def numpy(self):
        """
        return the image as a numpy.ndarray
        """
        return self.matrix

    def vect_convolution(self, kernel):
        """ """

        def zero_pad2d(x: np.ndarray, shape: tuple = None) -> np.ndarray:
            """ """
            if not shape:
                return x.copy()
            result = np.zeros(shape)
            r, c = x.shape
            result[:r, :c] = x
            return result

        rotate = lambda x, c=1: np.hstack((x[-c:], x[:-c]))

        c, ih, iw = self.matrix.shape
        _, kh, kw = kernel.matrix.shape
        ph, pw = kernel.padding
        sh, sw = kernel.stride
        h = (ih - kh + ph + sh) // sh
        w = (iw - kw + pw + sw) // sw
        result = np.empty((c, h, w))
        for channel in range(c):
            k = kernel.matrix[channel]
            kf = zero_pad2d(k, (ih, iw)).flatten()
            if ph or pw:
                img = np.zeros((ih + 2 * ph, iw + 2 * pw))
                img[ph:-ph, pw:-pw] = self.matrix[channel].copy()
            else:
                img = self.matrix[channel].copy()
            res = np.vstack(
                [rotate(kf, i + j) for j in range(0, ih * h, ih) for i in range(w)]
            )
            result[channel] = (res @ img.flatten()).reshape(h, w)
        return result


if __name__ == "__main__":
    pass
