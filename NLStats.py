#! /usr/bin/env python
"""
Non-Linear Regression Stats Module
Modified from code by Nathan Lemoine
(https://climateecology.wordpress.com/2013/08/26/r-vs-python-practical-data-analysis/)
Jeremy Smith
2016-07-22

"""

import numpy as np
import scipy.stats as spst
from scipy import linalg
from scipy.optimize import least_squares


class NLS:
    """This provides a wrapper for scipy.optimize.least_squares to get the relevant output for nonlinear least squares.
    Although scipy provides curve_fit for that reason, curve_fit only returns parameter estimates and covariances.
    This wrapper returns numerous statistics and diagnostics"""

    def __init__(self, func, p0, xdata, ydata, bounds=None, loss='soft_l1'):
        # Check the data
        if len(xdata) != len(ydata):
            msg = "The number of observations does not match the number of rows for the predictors"
            raise ValueError(msg)

        # Check parameter estimates
        if type(p0) != dict:
            msg = "Initial parameter estimates (p0) must be a dictionry of form p0={'a':1, 'b':2, etc}"
            raise ValueError(msg)
        if 'order' not in p0.keys():
            msg = "Initial parameter estimates (p0) must contain and 'order' list"
            raise ValueError(msg)

        self.func = func
        self.parmNames = p0['order']
        self.inits = np.array([p0[name] for name in self.parmNames])
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.nobs = len(ydata)
        self.nparm = len(self.inits)
        self.bounds = bounds
        self.loss = loss

        # Truncate parameter names to 8 characters
        for i, name in enumerate(self.parmNames):
            if len(name) > 8:
                self.parmNames[i] = self.parmNames[i][0:8]

    def fit(self):
        # Run the model
        if self.bounds == None:
            self.mod1 = least_squares(self.func, self.inits, method='trf', loss=self.loss, args=(self.xdata, self.ydata))
        else:
            self.mod1 = least_squares(self.func, self.inits, method='trf', bounds=self.bounds, loss=self.loss, args=(self.xdata, self.ydata))

        if not self.mod1.success:
            raise RuntimeError("Optimal parameters not found: {:s}".format(self.mod1.message))

        # Get the fitted parameters
        self.parmEsts = np.round(self.mod1.x, 6)

        # Calculate the Variances
        self.SS_err = np.sum(self.mod1.fun**2)
        self.SS_tot = np.sum((self.ydata - np.mean(self.ydata))**2)
        self.SS_par = self.SS_tot - self.SS_err

        # Calculate Degrees of Freedom
        self.df_par = self.nparm - 1
        self.df_tot = self.nobs - 1
        self.df_err = self.df_tot - self.df_par

        # Calculate Mean Sum Squares and RMSs
        self.MSE_err = self.SS_err / self.df_err
        self.MSE_tot = self.SS_tot / self.df_tot
        self.MSE_par = self.SS_par / self.df_par
        self.RMSE_err = np.sqrt(self.MSE_err)

        # Get the covariance matrix by inverting Jacobian
        u, s, vh = linalg.svd(self.mod1.jac, full_matrices=False)
        self.cov = self.MSE_err * np.dot(vh.T / s**2, vh)
 
        # Get parameter standard errors
        self.parmSE = np.sqrt(np.diag(self.cov))

        # Calculate the t-values and their p-values for parameters
        self.tvals = self.parmEsts / self.parmSE
        self.pvals = 2 * (1 - spst.t.cdf(np.abs(self.tvals), self.df_err))

        # Calculate F-value and its p-value
        self.fvalue = self.MSE_par / self.MSE_err
        self.pvalue = (1 - spst.f.cdf(self.fvalue, self.df_par, self.df_err))
 
    # Get AIC. Add 1 to the number of parameters to account for estimation of standard error
    def aic(self):
        # Get biased variance (MLE) and calculate log-likelihood
        self.s2b = self.SS_err / self.nobs
        self.logLik = -0.5 * self.nobs * np.log(2*np.pi) - 0.5 * self.nobs * np.log(self.s2b) - 1/(2*self.s2b) * self.SS_err
        return 2 * (self.nparm + 1) - 2 * self.logLik
 
    # Print the summary
    def summary(self):
        print "\n==============================================================="
        print "Non-linear least squares regression"
        print "Model: '{:s}'".format(self.func.func_name)
        print "Parameters:"
        print "  Factor       Estimate       Std Error      t-value    P(>|t|)"
        for i, name in enumerate(self.parmNames):
                print "  {:10s}  {: .6e}  {: .6e}  {: 8.5f}  {:8.5f}".format(name, self.parmEsts[i], self.parmSE[i], self.tvals[i], self.pvals[i])
        print
        print "Residual Standard Error: {:8.5f}".format(self.RMSE_err)
        print
        print "Analysis of Variance:"
        print "  Source     DF   SS        MS         F-value   P(>F)"
        print "  Model     {:3d}  {:8.5f}  {:8.5f}  {:9.5f}  {:8.5f}".format(self.df_par, self.SS_par, self.MSE_par, self.fvalue, self.pvalue)
        print "  Error     {:3d}  {:8.5f}  {:8.5f}".format(self.df_err, self.SS_err, self.MSE_err)
        print "  Total     {:3d}  {:8.5f}".format(self.df_tot, self.SS_tot)
        print "===============================================================\n"
