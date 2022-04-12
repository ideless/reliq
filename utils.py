import numpy as np
import scipy.signal as sig


def mix(*args):
    l = np.max([len(a) for a in args])
    ret = np.zeros(l)
    for a in args:
        for i in range(len(a)):
            ret[i] += a[i]
    ret = np.trim_zeros(ret, 'b')
    return ret


def dot(*args):
    l = np.min([len(a) for a in args])
    ret = np.ones(l)
    for a in args:
        for i in range(l):
            ret[i] *= a[i]
    ret = np.trim_zeros(ret, 'b')
    return ret


def convolve(*args):
    ret = [1]
    for a in args:
        ret = sig.convolve(ret, a)
    return ret


def convolve_avg(pdfs, avgs):
    if len(pdfs) != len(avgs):
        raise
    ret = []
    for i in range(len(pdfs)):
        tmp = [1]
        for j in range(len(pdfs)):
            if i == j:
                tmp = sig.convolve(tmp, dot(pdfs[j], avgs[j]))
            else:
                tmp = sig.convolve(tmp, pdfs[j])
        ret = mix(ret, tmp)
    return ret


if __name__ == '__main__':
    # a = mix([1, 2], [0, 0, 3], [0, 0, 0, 4])
    # print(a)
    # a = convolve([1, 2], [3, 4])
    # print(a)
    # a = convolve([1, 2], [3, 4, 5], [1])
    # print(a)
    a = convolve_avg([[1], [.5, .5]], [[7], [8, 9]])
    print(a)
