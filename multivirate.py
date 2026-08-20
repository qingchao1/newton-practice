import numpy as np
def pderiv(f,x,eps=1e-5):
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    fx=f(x)
    for _ in range(x.shape[0]):
        d = np.zeros_like(x)
        d[_] = eps
        res[_] = (f(x + d) - fx)/eps
    return res[:,:,np.new_axis]
def pderiv2(f, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    res = np.zeros((n, n), dtype=float)
    fx = f(x)

    for i in range(n):
        ei = np.zeros_like(x)
        ei[i] = eps
        res[i, i] = (
            f(x + ei) - 2 * fx + f(x - ei)
        ) / eps**2

        for j in range(i + 1, n):
            ej = np.zeros_like(x)
            ej[j] = eps
            value = (
                f(x + ei + ej)
                - f(x + ei - ej)
                - f(x - ei + ej)
                + f(x - ei - ej)
            ) / (4 * eps**2)

            res[i, j] = value
            res[j, i] = value

    return res
def multi_iter(f,x,eps):
    d=pderiv(f,x,eps)
    H=pderiv2(f,x,eps)
    return x-np.linalg.inv(H)@d

def optimize(f,x,min_dif=1e-5,eps=1e-5):
    x0=x.copy()+1
    x1=x.copy()
    i=0
    while np.abs(x1-x0)>min_dif:
        x0=x1
        x1=multi_iter(f,x0,eps)
        print(i,x1,f(x1))
    