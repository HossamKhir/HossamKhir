#! /usr/bin/env python3
"""
"""
import numpy as np
from convolution.convolution import Image


if __name__ == "__main__":
    img = Image(np.ones((5, 5)))
    kernel = Image(np.eye(3), stride=1)
    print(img * kernel)
    img3c = Image(np.ones((3, 5, 5)))
    I = np.eye(3)
    k3c = Image(np.array([I, -I, I / 2]))
    print(img3c * k3c)
