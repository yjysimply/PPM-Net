import os
import numpy as np
import torch
import scipy.io as scio
import torch.utils
import torch.utils.data
from torch import nn
from time import time
from tqdm import tqdm
from model.pp_model import PP_Net
from utils.hyperVca import hyperVca
from utils.loadhsi import loadhsi
from utils.result_em import result_em
from utils.FCLSU import FCLSU
from utils.loadparameter import loadparameter
from torch.utils.data import Dataset

model_weights = './PP_weight/'
output_path = './PP_out/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(model_weights):
    os.makedirs(model_weights)
model_weights += 'PP.pt'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class CustomDataset(Dataset):
    def __init__(self, train_db, fcls_a_true):
        self.train_db = train_db
        self.fcls_a_true = fcls_a_true

    def __len__(self):
        return len(self.train_db)

    def __getitem__(self, idx):
        return self.train_db[idx], self.fcls_a_true[:, idx]

def train(case, K = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on', device)

    Y, A_true, P, M = loadhsi(case)
    lr, lambda_kl, lambda_sad, lambda_vol, lambda_a = loadparameter(case)

    z_dim = 4
    Channel, N = Y.shape
    batchsz = N // K
    epochs = 200

    Y = np.transpose(Y)
    vca_em, indicies, snrEstimate = hyperVca(Y.T, P)
    M0 = np.reshape(vca_em, [1, vca_em.shape[0], vca_em.shape[1]]).astype('float32')
    M0 = torch.tensor(M0).to(device)
    print('SNR:', snrEstimate)

    fcls_a_true = FCLSU(vca_em, Y.T, 0.01)

    train_db = torch.tensor(Y)
    train_db = CustomDataset(train_db, fcls_a_true)
    train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsz, shuffle=True)

    model = PP_Net(P, Channel, z_dim, M0).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tic = time()
    losses = []
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

        losses.append(loss.detach().cpu().numpy())
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), model_weights)
            scio.savemat(output_path + 'loss.mat', {'loss': losses})

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

        scio.savemat(output_path + 'results.mat', {'EM': EM_hat.T,
                                                   'A': A_hat.T,
                                                   'Y_hat': y_hat.cpu().numpy()})

        armse_y, asad_y, armse_a, armse_em, asad_em = result_em(EM_hat, M, A_hat, A_true, Y, Y_hat)

        return armse_y, asad_y, armse_a, armse_em, asad_em, toc - tic

if __name__=='__main__':
    cases = ['ex2', 'ridge', 'urban', 'houston', 'synthetic']  # 8.4
    case = cases[0]
    K = 10
    armse_y, asad_y, armse_a, armse_em, asad_em, tim = train(case, K)

    print('*' * 70)
    print('time elapsed:', tim)
    print('RESULTS:')
    print('aRMSE_Y:', armse_y)
    print('aSAD_Y:', asad_y)
    print('aRMSE_a:', armse_a)
    print('aRMSE_M', armse_em)
    print('aSAD_em', asad_em)