import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression


def eval_division(val, m=5): 
    if val%m == 0:
        return True
    else:
        return False
    
def mask_idx(idx, m=5): 
    nulls = (
        np.array(
            [eval_division(x, m=m) for _, x in enumerate(idx)],
            dtype=bool
        )
    )
    return nulls

def linear_regression(x, y, thresh=0.05):
    reg = LinearRegression()
    reg.fit(x.values.reshape(-1,1), y.values)
    print(f'Intercept: {reg.intercept_}')
    print(f'Coefficient: {reg.coef_[0]}')
    y_pred = x * reg.coef_ + reg.intercept_
    residuals = y - y_pred
    adf, p_value = check_residuals(residuals)
    if p_value < thresh:
        return reg, y_pred, residuals
    else:
        raise Exception('Residual is not stationary!')

def check_residuals(residuals): 
    adf = adfuller(residuals)[0]
    p_value = adfuller(residuals)[1]
    print(f'ADF Statistic: {adf}')
    print(f'p-value: {p_value}')
    return adf, p_value

def zscore(x):
    mean = np.mean(x)
    std = np.std(x)
    z_score = (x - mean) / std
    return z_score

def mod_zscore(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    mod_zscore = (0.6745)*(x - median)/mad
    return mod_zscore
