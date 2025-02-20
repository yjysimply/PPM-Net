import numpy as np
import scipy as sp
import torch
from scipy.optimize import nnls

def FCLSU(M, Y, sigma=1):
    '''

    :param M: (L, P) 通道数 x 端元数
    :param Y: (L, N) 通道数 x 像素
    :return: A_hat: (P, N) 端元数 x 像素
    '''

    P = M.shape[1]
    N = Y.shape[1]
    M = sp.vstack((sigma * M, sp.ones((1, P)) ))  # 改变相关系数，将 NNLS 方法变成 FCLS
    Y = sp.vstack((sigma * Y, sp.ones((1, N)) ))
    A_hat = np.zeros((P, N))

    for i in np.arange(N):
        A_hat[:, i], res = nnls(M, Y[:, i])
    A_hat = torch.tensor(A_hat)

    return A_hat
