import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression


class Cointegrate(): 
    
    def __init__(self, x, y): 
        self.x = x
        self.y = y
        
    def regression(self): 
        
        self.reg, self.y_pred, self.residuals = (
            self.linear_regression(self.x, self.y)
        )

        self.limits = self._mod_zscore(self.residuals)
        
        self.ub = np.mean(self.limits) + np.std(self.limits)
        self.lb = np.mean(self.limits) - np.std(self.limits)
        
        self.signals = (
            np.select(condlist=[self.limits >= self.ub,
                                self.limits <= self.lb], 
                      choicelist=[-1, 1], 
                      default=0)
        )

    @staticmethod
    def linear_regression(x, y, thresh=0.05):
        
        def _check_residuals(residuals): 
            adf = adfuller(residuals)[0]
            p_value = adfuller(residuals)[1]
            print(f'ADF Statistic: {adf}')
            print(f'p-value: {p_value}')
            return adf, p_value
        
        reg = LinearRegression()
        reg.fit(x.values.reshape(-1,1), y.values)
        print(f'Intercept: {reg.intercept_}')
        print(f'Coefficient: {reg.coef_[0]}')
        y_pred = x * reg.coef_ + reg.intercept_
        residuals = y - y_pred
        adf, p_value = _check_residuals(residuals)
        if p_value < thresh:
            return reg, y_pred, residuals
        else:
            raise Exception('Residual is not stationary!')

    @staticmethod  
    def _mod_zscore(x):
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        mod_zscore = (0.6745)*(x - median)/mad
        return mod_zscore
