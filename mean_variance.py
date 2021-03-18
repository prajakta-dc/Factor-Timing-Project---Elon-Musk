import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize


def mu_(x):
    x = x[x.isin([np.nan, np.inf, -np.inf]).any(1)]
    return x.mean()
def cov_(x):
    return x.cov()
def mean_var(rtn):    
    m = mu_(rtn)
    e = np.ones((len(rtn.columns),1))
    inv = np.linalg.inv(cov_(rtn))
    return (inv @ m) / (e.T @ inv @ m)

def mean_var_multi(rtn, window = 52):
    dates = rtn.index
    w_ = []
    for d in tqdm(range(len(dates)-window)):
        date_idx = dates[d:d+window+1]
        try:
            rtn_t = rtn.loc[date_idx]
            res = mean_var(rtn_t)
            w_.append(res)
        except:
            continue
    w_ = np.array(w_).squeeze()
    final_w = pd.DataFrame(w_,columns = rtn.columns, index = dates[-w_.shape[0]:])
    return final_w


def min_var(rtn):
    e = np.ones((len(rtn.columns),1))
    inv = np.linalg.inv(cov_(rtn))
    return (inv @ e) / (e.T @ inv @ e)

def min_var_multi(rtn, window = 52):
    dates = rtn.index
    w_ = []
    for d in tqdm(range(len(dates)-window)):
        date_idx = dates[d:d+window+1]
        try:
            rtn_t = rtn.loc[date_idx]
            res = min_var(rtn_t)
            w_.append(res)
        except:
            continue
    w_ = np.array(w_).squeeze()
    final_w = pd.DataFrame(w_,columns = rtn.columns, index = dates[-w_.shape[0]:])
    return final_w