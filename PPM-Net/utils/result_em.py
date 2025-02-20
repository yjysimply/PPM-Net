import numpy as np
import scipy.io as scio

def result_em(EM_hat, M, A_hat, A_true, Y, Y_hat):

    EM_hat = np.mean(EM_hat, axis=2)
    norm_EM_GT = np.sqrt(np.sum(M ** 2, 0))
    norm_EM_hat = np.sqrt(np.sum(EM_hat ** 2, 0))
    inner_prod = np.sum(M * EM_hat, 0)
    em_sad = np.arccos(inner_prod / norm_EM_GT / norm_EM_hat)

    # 结果数据
    asad_em = np.mean(em_sad)
    armse_em = np.mean(np.sqrt(np.mean((M - EM_hat) ** 2, axis=0)))

    norm_y = np.sqrt(np.sum(Y ** 2, 1))
    norm_y_hat = np.sqrt(np.sum(Y_hat ** 2, 1))

    # 结果数据
    armse_a = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))
    armse_y = np.mean(np.sqrt(np.mean((Y_hat - Y) ** 2, axis=1)))
    asad_y = np.mean(np.arccos(np.sum(Y_hat * Y, 1) / norm_y / norm_y_hat))


    return armse_y, asad_y, armse_a, armse_em, asad_em