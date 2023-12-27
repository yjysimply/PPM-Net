import os
import numpy as np
import torch
import scipy as sp
import scipy.io as scio
import torch.utils
import torch.utils.data
from torch import nn
from time import time
from tqdm import tqdm
from pp_model import PPM
from scipy.optimize import nnls
from torch.utils.data import Dataset

def FCLSU(M, Y, sigma=1):
    P = M.shape[1]
    N = Y.shape[1]
    M = sp.vstack((sigma * M, sp.ones((1, P)) ))
    Y = sp.vstack((sigma * Y, sp.ones((1, N)) ))
    A_hat = np.zeros((P, N))

    for i in np.arange(N):
        A_hat[:, i], res = nnls(M, Y[:, i])
    A_hat = torch.tensor(A_hat)

    return A_hat

def pca(X, d):
    N = np.shape(X)[1]
    xMean = np.mean(X, axis=1, keepdims=True)
    XZeroMean = X - xMean

    [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
    Ud = U[:, 0:d]
    return Ud

def hyperVca(M, q):
    L, N = np.shape(M)
    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean

    U, S, V = np.linalg.svd(RZeroMean @  RZeroMean.T / N)
    Ud = U[:, 0:q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    SNRth = 18 + 10 * np.log(q)
    if SNR > SNRth:
        d = q
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        Y = Xd /  np.sum(Xd * u , axis=0, keepdims=True)

    else:
        d = q - 1
        r_bar = np.mean(M.T, axis=0, keepdims=True).T
        Ud = pca(M, d)

        R_zeroMean = M - r_bar
        Xd = Ud.T @ R_zeroMean
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([q, 1])
    e_u[q - 1, 0] = 1
    A = np.zeros([q, q])
    A[:, 0] = e_u[0]
    I = np.eye(q)
    k = np.zeros([N, 1])

    indicies = np.zeros([q, 1])
    for i in range(q):
        w = np.random.random([q, 1])

        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        f = tmpNumerator / np.linalg.norm(tmpNumerator)

        v = f.T @ Y
        k = np.abs(v)

        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    if (SNR > SNRth):
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U, indicies, snrEstimate

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)

class CustomDataset(Dataset):
    def __init__(self, train_db, fcls_a_true):
        self.train_db = train_db
        self.fcls_a_true = fcls_a_true

    def __len__(self):
        return len(self.train_db)

    def __getitem__(self, idx):
        return self.train_db[idx], self.fcls_a_true[:, idx]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

case = ['ex2']

file = './dataset/data_ex2.mat'
data = scio.loadmat(file)
Y = data['r']
A_true = data['alphas']
M = data['Mvs']
P = A_true.shape[0]
Y = Y.astype(np.float32)
A_true = A_true.astype(np.float32)

z_dim = 4
Channel, N = Y.shape
batchsz = N // 10
epochs = 200

lr = 1e-2
lambda_kl = 0.6
lambda_sad = 0.8
lambda_vol = 0.3
lambda_a = 6

Y = np.transpose(Y)
vca_em, indicies, snrEstimate = hyperVca(Y.T, P)
M0 = np.reshape(vca_em, [1, vca_em.shape[0], vca_em.shape[1]]).astype('float32')
M0 = torch.tensor(M0).to(device)
print('SNR:', snrEstimate)

fcls_a_true = FCLSU(vca_em, Y.T, 0.01)

train_db = torch.tensor(Y)
train_db = CustomDataset(train_db, fcls_a_true)
train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsz, shuffle=True)

model = PPM(P, Channel, z_dim, M0).to(device)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

tic = time()
for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_db:
        y, fcls_a = batch
        y = y.to(device)
        fcls_a = fcls_a.to(device).T

        y_hat, em_hat, a_hat, mu_s, mu_d, var_s, var_d = model(y)

        loss_rec = ((y - y_hat) ** 2).sum() / y.shape[0]

        loss_kl = -0.5 * (var_s + var_d + 2 - mu_s ** 2 - mu_d ** 2 - var_s.exp() - var_d.exp())
        loss_kl = loss_kl.sum() / y.shape[0]
        loss_kl = torch.max(loss_kl, torch.tensor(0.2).to(device))

        if epoch < 100:
            a_hat = torch.squeeze(a_hat)
            loss_a = (a_hat.T - fcls_a).square().sum() / y.shape[0]
            loss = loss_rec + lambda_kl * loss_kl + lambda_a * loss_a
        else:
            em_bar = em_hat.mean(dim=1, keepdim=True)
            loss_vol = ((em_hat - em_bar) ** 2).sum() / y.shape[0] / P / Channel

            em_bar = em_hat.mean(dim=0, keepdim=True)
            aa = (em_hat * em_bar).sum(dim=2)
            em_bar_norm = em_bar.square().sum(dim=2).sqrt()
            em_tensor_norm = em_hat.square().sum(dim=2).sqrt()
            sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
            loss_sad = sad.sum() / y.shape[0] / P

            loss = loss_rec + lambda_kl * loss_kl + lambda_vol * loss_vol + lambda_sad * loss_sad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

toc = time()
model.eval()
with torch.no_grad():
    y_hat, em_hat, a_hat, mu_s, mu_d, var_s, var_d = model(torch.tensor(Y).to(device))

    Y_hat = y_hat.cpu().numpy()
    EM_hat = em_hat.data.cpu().numpy().T
    A_hat = a_hat.cpu().numpy().T

    dev = np.zeros([P, P])
    for i in range(P):
        for j in range(P):
            dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
    pos = np.argmin(dev, axis=0)

    A_hat = A_hat[pos, :]
    EM_hat = EM_hat[:, pos, :]

    norm_EM_GT = np.sqrt(np.sum(M ** 2, 0))
    norm_EM_hat = np.sqrt(np.sum(EM_hat ** 2, 0))
    inner_prod = np.sum(M * EM_hat, 0)
    em_sad = np.arccos(inner_prod / norm_EM_GT / norm_EM_hat)

    asad_em = np.mean(em_sad)

    Mvs = np.reshape(M, [Channel, P * N])
    EM_hat = np.reshape(EM_hat, [Channel, P * N])
    armse_em = np.mean(np.sqrt(np.mean((Mvs - EM_hat) ** 2, axis=0)))

    norm_y = np.sqrt(np.sum(Y ** 2, 1))
    norm_y_hat = np.sqrt(np.sum(Y_hat ** 2, 1))

    armse_a = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))
    armse_y = np.mean(np.sqrt(np.mean((Y_hat - Y) ** 2, axis=1)))
    asad_y = np.mean(np.arccos(np.sum(Y_hat * Y, 1) / norm_y / norm_y_hat))

    print('RESULTS:')
    print('aRMSE_Y:', armse_y)
    print('aSAD_Y:', asad_y)
    print('aRMSE_a:', armse_a)
    print('aRMSE_M', armse_em)
    print('aSAD_em', asad_em)