import numpy as np


def LaplaceWavelet(p):
    A = 0.08
    eps = np.random.uniform(0.01, 0.05)  # 0.03, 该参数可变，敏感
    tau = 0.1  # 与时移因子 b效果一样，效果没有提升
    # taus = np.array([0.1, 0.3, 0.5])
    # tau = np.random.choice(taus, 1)
    # f = np.random.randint(10, 40)  # 50-100
    # f = 20  # 10, 20, 20-best on SQ
    f = 50  # default: 50, f = 60 better
    w = 2 * np.pi * f
    q = 1 - pow(eps, 2)

    y = A * np.exp((-eps / (np.sqrt(q))) * (w * (p - tau))) * (-np.sin(w * (p - tau)))
    return y


def Morlet(p):
    # You can see: scipy.signal.morlet2
    C = 0.75112554  # pi**(-0.25), pow(np.pi, -0.25)
    f = np.random.uniform(10, 100)  # 10-100 best
    w = 2 * np.pi * f

    # if np.random.rand()>0.5:
    y = C * np.exp(-np.power(p, 2) / 2) * np.sin(w * p)  # sq-snr0, ~98.44
    # else:
    # y = C * np.exp(-np.power(p, 2) / 2) * np.cos(w * p)

    return y


# @njit("float64[:](float64[:])")
def HermitianRe(p):  # Mexican hat
    y = (1 - np.power(p, 2)) * np.exp(-np.power(p, 2) / 2)

    return y


# @njit("float64[:](float64[:])")
def HermitianIm(p):
    y = p * np.exp(-np.power(p, 2) / 2)
    return y


# @njit("float64[:](float64[:])")
def HarmonicRe(t):
    return (np.sin(4 * np.pi * t) - np.sin(2 * np.pi * t)) / (2 * np.pi * t)


# @njit("float64[:](float64[:])")
def HarmonicIm(t):
    return (np.cos(2 * np.pi * t) - np.cos(4 * np.pi * t)) / (2 * np.pi * t)
