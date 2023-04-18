# -*- coding:utf-8 -*-
# @FileName :rnn.py
# @Time :2023/4/18 15:00
# @Author :Xiaofeng
import numpy as np

from assignment3.cs231n.data_utils import rel_error
from assignment3.cs231n.rnn_layers import rnn_forward


def test_forward():
    N, T, D, H = 2, 3, 4, 5

    x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
    h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
    b = np.linspace(-0.7, 0.1, num=H)

    h, _ = rnn_forward(x, h0, Wx, Wh, b)
    expected_h = np.asarray([
        [
            [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
            [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
            [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
        ],
        [
            [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
            [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
            [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043]]])
    print('h error: ', rel_error(expected_h, h))


if __name__ == '__main__':
    test_forward()
