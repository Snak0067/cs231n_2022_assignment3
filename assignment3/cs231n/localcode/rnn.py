# -*- coding:utf-8 -*-
# @FileName :rnn.py
# @Time :2023/4/18 15:00
# @Author :Xiaofeng
import numpy as np
from matplotlib import pyplot as plt

from assignment3.cs231n.captioning_solver import CaptioningSolver
from assignment3.cs231n.classifiers.rnn import CaptioningRNN
from assignment3.cs231n.coco_utils import load_coco_data
from assignment3.cs231n.data_utils import rel_error
from assignment3.cs231n.rnn_layers import rnn_forward

data = load_coco_data(pca_features=True)


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


def test_rnn():
    N, D, W, H = 10, 20, 30, 40
    word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
    V = len(word_to_idx)
    T = 13

    model = CaptioningRNN(
        word_to_idx,
        input_dim=D,
        wordvec_dim=W,
        hidden_dim=H,
        cell_type='rnn',
        dtype=np.float64
    )

    # Set all model parameters to fixed values
    for k, v in model.params.items():
        model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

    features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
    captions = (np.arange(N * T) % V).reshape(N, T)

    loss, grads = model.loss(features, captions)
    expected_loss = 9.83235591003

    print('loss: ', loss)
    print('expected loss: ', expected_loss)
    print('difference: ', abs(loss - expected_loss))


def overfit_smalldata_rnn():
    np.random.seed(231)

    small_data = load_coco_data(max_train=50)

    small_rnn_model = CaptioningRNN(
        cell_type='rnn',
        word_to_idx=data['word_to_idx'],
        input_dim=data['train_features'].shape[1],
        hidden_dim=512,
        wordvec_dim=256,
    )

    small_rnn_solver = CaptioningSolver(
        small_rnn_model, small_data,
        update_rule='adam',
        num_epochs=50,
        batch_size=25,
        optim_config={
            'learning_rate': 5e-3,
        },
        lr_decay=0.95,
        verbose=True, print_every=10,
    )

    small_rnn_solver.train()

    # Plot the training losses.
    plt.plot(small_rnn_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()


if __name__ == '__main__':
    overfit_smalldata_rnn()
