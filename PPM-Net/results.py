import numpy as np

def results(Y, y_hat, M, em_tensor, A_true, A_hat, P, Channel):
    EM_hat = em_tensor.data.cpu().numpy()
    A_true = A_true.reshape([P, -1])
    dev = np.zeros([P, P])
    for i in range(P):
        for j in range(P):
            dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
    pos = np.argmin(dev, axis=0)  # 求最小值的索引

    A_hat = A_hat[pos, :]
    EM_hat = EM_hat[:, pos, :]
    EM_hat = np.mean(EM_hat, axis=0).T

    Y_hat = y_hat.cpu().numpy().reshape(-1, Channel)
    Y = Y.reshape(-1, Channel)
    norm_y = np.sqrt(np.sum(Y ** 2, 1))
    norm_y_hat = np.sqrt(np.sum(Y_hat ** 2, 1))
    armse_y = np.mean(np.sqrt(np.mean((Y_hat - Y) ** 2, axis=1)))
    asad_y = np.mean(np.arccos(np.sum(Y_hat * Y, 1) / norm_y / norm_y_hat))

    armse_a = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))
    armse_em = np.mean(np.sqrt(np.mean((M - EM_hat) ** 2, axis=0)))

    norm_EM_GT = np.sqrt(np.sum(M ** 2, 0))
    norm_EM_hat = np.sqrt(np.sum(EM_hat ** 2, 0))
    inner_prod = np.sum(M * EM_hat, 0)
    em_sad = np.arccos(inner_prod / norm_EM_GT / norm_EM_hat)

    asad_em = np.mean(em_sad)

    return armse_y, asad_y, armse_a, armse_em, asad_em