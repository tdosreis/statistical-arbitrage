import numpy as np
from utility_functions import *


class Cointegrate(): 
    
    def __init__(self, x, y): 
        self.x = x
        self.y = y
        
    def regression(self): 
        
        self.reg, self.y_pred, self.residuals = (
            linear_regression(self.x, self.y)
        )

        self.limits = mod_zscore(self.residuals)
        
        self.ub = np.mean(self.limits) + np.std(self.limits)
        self.lb = np.mean(self.limits) - np.std(self.limits)
        
        self.signals = (
            np.select(condlist=[self.limits >= self.ub,
                                self.limits <= self.lb], 
                      choicelist=[-1, 1], 
                      default=0)
        )