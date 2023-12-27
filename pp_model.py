import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPM(nn.Module):
    def __init__(self, P, Channel, z_dim, M0):
        super().__init__()
        self.P = P
        self.Channel = Channel
        self.M0 = M0

        self.fc1 = nn.Linear(Channel, 32 * P)
        self.bn1 = nn.BatchNorm1d(32 * P)

        self.fc2 = nn.Linear(32 * P, 16 * P)
        self.bn2 = nn.BatchNorm1d(16 * P)

        self.fc3 = nn.Linear(16 * P, 4 * P)
        self.bn3 = nn.BatchNorm1d(4 * P)

        self.fc4 = nn.Linear(4 * P, z_dim)
        self.fc5 = nn.Linear(4 * P, z_dim)

        self.fc6 = nn.Linear(Channel, 32 * P)
        self.bn6 = nn.BatchNorm1d(32 * P)

        self.fc7 = nn.Linear(32 * P, 16 * P)
        self.bn7 = nn.BatchNorm1d(16 * P)

        self.fc8 = nn.Linear(16 * P, 4 * P)
        self.bn8 = nn.BatchNorm1d(4 * P)

        self.fc9 = nn.Linear(4 * P, z_dim)
        self.fc10 = nn.Linear(4 * P, z_dim)

        self.fc11 = nn.Sequential(
            nn.Linear(Channel, 32 * P),
            nn.BatchNorm1d(32 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(32 * P, 16 * P),
            nn.BatchNorm1d(16 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(16 * P, 4 * P),
            nn.BatchNorm1d(4 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(4 * P, 4 * P),
            nn.BatchNorm1d(4 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(4 * P, P),
            nn.Softmax(dim=1)
        )

        self.fc12 = nn.Linear(z_dim, P * 2)
        self.bn12 = nn.BatchNorm1d(P * 2)

        self.fc13 = nn.Linear(P * 2, P * P)
        self.bn13 = nn.BatchNorm1d(P * P)

        self.fc17 = nn.Linear(P * P, P * P)

        self.fc14 = nn.Linear(z_dim, P * 4)
        self.bn14 = nn.BatchNorm1d(P * 4)

        self.fc15 = nn.Linear(P * 4, 64 * P)
        self.bn15 = nn.BatchNorm1d(64 * P)

        self.fc16 = nn.Linear(64 * P, Channel * P)

    def encoder_s(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h11 = F.leaky_relu(h1, 0.00)

        h1 = self.fc3(h11)
        h1 = self.bn3(h1)
        h1 = F.leaky_relu(h1, 0.00)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_d(self, x):
        h1 = self.fc6(x)
        h1 = self.bn6(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h11 = F.leaky_relu(h1, 0.00)

        h1 = self.fc8(h11)
        h1 = self.bn8(h1)
        h1 = F.leaky_relu(h1, 0.00)

        mu = self.fc9(h1)
        log_var = self.fc10(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)

        return mu + eps * std

    def decoder_s(self, s):
        h1 = self.fc12(s)
        h1 = self.bn12(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc13(h1)
        h1 = self.bn13(h1)
        h1 = F.leaky_relu(h1, 0.00)

        psi = self.fc17(h1)

        return psi

    def deocer_d(self, d):
        h1 = self.fc14(d)
        h1 = self.bn14(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc15(h1)
        h1 = self.bn15(h1)
        h1 = F.leaky_relu(h1, 0.00)

        dM = self.fc16(h1)
        return dM

    def decoder_em(self, psi, dM):
        M0 = (self.M0).repeat(psi.shape[0], 1, 1)
        em = M0 @ psi + dM
        em = torch.sigmoid(em)
        return em

    def forward(self, inputs):
        mu_s, var_s = self.encoder_s(inputs)
        mu_d, var_d = self.encoder_d(inputs)
        a = self.fc11(inputs)

        s = self.reparameterize(mu_s, var_s)
        d = self.reparameterize(mu_d, var_d)

        psi = self.decoder_s(s)
        dM = self.deocer_d(d)

        psi_tensor = psi.view([-1, self.P, self.P])
        dM_tensor = dM.view([-1, self.Channel, self.P])

        em_tensor = self.decoder_em(psi_tensor, dM_tensor)
        em_tensor = em_tensor.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)

        return y_hat, em_tensor, a, mu_s, mu_d, var_s, var_d
