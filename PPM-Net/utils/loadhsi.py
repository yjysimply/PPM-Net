import numpy as np
import scipy.io as scio

def loadhsi(case):
    '''
    :input: case: for different datasets,
                 'toy' and 'usgs' are synthetic datasets
    :return: Y : HSI data of size [Bands,N]
             A_ture : Ground Truth of abundance map of size [P,N]
             P : nums of endmembers signature
    '''
    if case == 'ridge':
        file = './dataset/JasperRidge2_R198.mat'
        data = scio.loadmat(file)
        Y = data['Y']
        Y = np.reshape(Y,[198,100,100])
        for i,y in enumerate(Y):
            Y[i]=y.T
        Y = np.reshape(Y, [198, 10000])

        GT_file = './dataset/JasperRidge2_end4.mat'
        A_true = scio.loadmat(GT_file)['A']
        M = scio.loadmat(GT_file)['M']
        A_true = np.reshape(A_true, (4, 100, 100))
        for i,A in enumerate(A_true):
            A_true[i]=A.T
        A_true = np.reshape(A_true, (4, 10000))
        if np.max(Y) > 1:
            Y = Y / np.max(Y)

    elif case == 'synthetic':
        file = './dataset/synthetic_data.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # Y_var, Y_nlin, Y_both
        # Y = Y.T
        A_true = data['A']
        M = data['M']  # L,P

    elif case == 'ex2':
        file = './dataset/data_ex2.mat'
        data = scio.loadmat(file)
        Y = data['r']  # (224, 2500)
        # Y = Y.T
        A_true = data['alphas']
        M = data['M']

    elif case == 'urban':
        file = './dataset/Urban_R162.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h) (162, 307*307)

        GT_file = './dataset/Urban_end4.mat'
        A_true = scio.loadmat(GT_file)['A']
        M = scio.loadmat(GT_file)['M']
        if np.max(Y) > 1:
            Y = Y / np.max(Y)

    elif case == 'houston':
        file = './dataset/Houston.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h) (156, 95*95)
        A_true = data['A']
        M = data['M']

    P = A_true.shape[0]

    Y = Y.astype(np.float32)  # 数据类型转换
    A_true = A_true.astype(np.float32)
    return Y, A_true, P, M

if __name__=='__main__':
    cases = ['ex2','ridge']
    case = cases[0]

    Y, A_true, P = loadhsi(case)
    Channel = Y.shape[0]
    N = Y.shape[1]
    print(case)
    print('nums of EM:',P)
    print('Channel :',Channel, ' pixels :',N)

    GT_file = '../dataset/Urban_end4.mat'
    M = scio.loadmat(GT_file)['M']
    print(M.shape)

    from matplotlib import pyplot as plt
    plt.plot(M)
    plt.show()

