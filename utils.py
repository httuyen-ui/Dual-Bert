import math
from math import sqrt
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import numpy as np

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean())
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean()
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def MAE(y, f):
    rs = sklearn.metrics.mean_absolute_error(y, f)
    return rs


def ci(y, f):
    ind = np.argsort(y)  # indices that would sort y ascending
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def draw_loss(train_losses, test_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_loss(train_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_mse(mse, rmse, title, result_folder):
    plt.figure()
    plt.plot(mse, label='test MSE')
    plt.plot(rmse, label='test rMSE')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def evaluate_others(M, Tr_neg, Te, positions=[1, 5, 10, 15]):
    """
    :param M: prediction scores
    :param Tr_neg: dict containing Te-related negatives
    :param Te: dict
    :param positions:
    :return:
    """
    prec = np.zeros(len(positions))
    rec = np.zeros(len(positions))
    map_value, auc_value, ndcg = 0.0, 0.0, 0.0

    val = M
    inx = np.array(Tr_neg)
    A = set(Te)
    B = set(inx) - A
    # compute precision and recall
    ii = np.argsort(val[inx])[::-1][:max(positions)]
    prec += precision(Te, inx[ii], positions)
    rec += recall(Te, inx[ii], positions)
    ndcg_user = nDCG(Te, inx[ii], 10)
    # compute map and AUC
    pos_inx = np.array(list(A))
    neg_inx = np.array(list(B))
    map_user, auc_user = map_auc(pos_inx, neg_inx, val)
    ndcg += ndcg_user
    map_value += map_user
    auc_value += auc_user
    return map_value / len(Te.keys()), auc_value / len(Te.keys()), ndcg / len(Te.keys()), prec / len(
    Te.keys()), rec / len(Te.keys())


def precision(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def nDCG(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1/math.log(i+2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i]/math.log(i+2, 2) for i in range(num)])
    if idcg > 0:
        return dcg/idcg
    else:
        return idcg


def overlap_at_top_percent(labels, preds, percents=(1, 5, 10, 20)):
    """
    Overlap@N% theo Eq.20 paper HSTrans (Neural Networks 2025):

        Overlap@N% = TP / (T x N%)

    T   = tong so mau trong test set (ca positive lan negative label=0)
    N%  = ti le, vi du 0.01 (1%), 0.05 (5%), ...
    TP  = so positive that su (label > 0) nam trong top (T x N%) score cao nhat

    Dam bao monotonicity: O@1% >= O@5% >= O@10% >= O@20%

    Parameters
    ----------
    labels   : numpy array, toan bo T mau (ca label=0 lan label 1-5)
    preds    : numpy array, predicted score tuong ung
    percents : tuple cac nguong N (don vi %, vi du 1 = top 1%)

    Returns
    -------
    dict {1: float, 5: float, 10: float, 20: float}
    """
    labels = np.array(labels).flatten()
    preds  = np.array(preds).flatten()
    T      = len(labels)
    if T == 0:
        return {p: 0.0 for p in percents}
    ranked_idx = np.argsort(preds)[::-1]   # giam dan theo score
    result = {}
    for n in percents:
        top_k   = max(1, int(T * n / 100.0))
        top_idx = ranked_idx[:top_k]
        TP      = np.sum(labels[top_idx] > 0)
        result[n] = float(TP) / float(T * n / 100.0)
    return result


def map_auc(pos_inx, neg_inx, val):
    map = 0.0
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    auc_num = 0.0
    for i,pos in enumerate(pos_sort):
        num = 0.0
        for neg in neg_sort:
            if pos<=neg:
                num+=1
            else:
                auc_num+=1
        map += (i+1)/(i+num+1)
    return map/len(pos_inx), auc_num/(len(pos_inx)*len(neg_inx))

