# Import Required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm


class Selector(object):
    # TODO: Need to work on forward selection process, Currently supports Backward-Elimination only.

    def __init__(self, X, y, significance_level=0.05, add_constant=True):
        """
        :param X: input matrix of X independent variables
        :param significance_level: Significance level of the model to keep variable in model
        :param add_constant: if True, will add random constant on initialization
        """
        self.X = X
        self.y = y
        self.SL = significance_level
        self.add_constant = add_constant
        self.X_new = self.apply_constant()

    # Function to add constant to X and return new matrix
    def apply_constant(self):
        """
        Adds constant and return new features of matrix X
        :return: new Matrix of features of X with constant values prepended
        """
        return sm.add_constant(self.X)

    # Function will Give new matrix based on Backward Elimination
    def backward_elimination(self, features):
        """
        Apply backward elimination process to remove multi-collinear and identifies best suitable matrix of features of X
        :return: new Matrix of features of X
        """
        x = self.X_new
        y = self.y
        sl = self.SL
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_ols = sm.OLS(y, x).fit()
            max_var = max(regressor_ols.pvalues)
            if max_var > sl:
                for j in range(0, numVars - i):
                    if regressor_ols.pvalues[j].astype(float) == max_var:
                        print("Removing feature --> {}".format(features[j]))
                        x = np.delete(x, j, 1)
        #print(regressor_ols.summary())
        return x