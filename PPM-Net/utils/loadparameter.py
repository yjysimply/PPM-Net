def loadparameter(case):
    if case == 'ridge':
        lr = 1e-3
        lambda_kl = 0.04
        lambda_sad = 8
        lambda_vol = 7
        lambda_a = 6

    elif case == 'ex2':
        lr = 1e-2
        lambda_kl = 0.6
        lambda_sad = 0.8
        lambda_vol = 0.3
        lambda_a = 6

    elif case == 'urban':
        lr = 1e-3
        lambda_kl = 0.001
        lambda_sad = 4
        lambda_vol = 7
        lambda_a = 0.0001

    elif case == 'houston':
        lr = 1e-3
        lambda_kl = 0.3
        lambda_sad = 0.5
        lambda_vol = 6
        lambda_a = 1

    elif case == 'synthetic':
        lr = 1e-3
        lambda_kl = 9e-3
        lambda_sad = 0.2
        lambda_vol = 0.4
        lambda_a = 7

    return lr, lambda_kl, lambda_sad, lambda_vol, lambda_a