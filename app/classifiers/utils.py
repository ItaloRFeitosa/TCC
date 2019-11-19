def normalizar(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x - x_mean)/x_std
    return x