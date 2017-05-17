#! /usr/bin/env python
"""
Non-Linear Regression Stats Module
Modified from code by Nathan Lemoine
(https://climateecology.wordpress.com/2013/08/26/r-vs-python-practical-data-analysis/)
Jeremy Smith
2016-07-22

"""

import os
import sys
import numpy as np
import scipy.stats as spst
from scipy import linalg
from scipy.optimize import least_squares

infodict = {'trf':"Trust Region Reflective Algorithm", 'lm':"Levenberg-Marquardt Algorithm",
            'linear':"Linear Loss Function", 'soft_l1':"Soft L1 Loss Function", 'huber':"Huber Loss Function",
            'cauchy':"Cauchy Loss Function", 'arctan':"Arctangent Loss Function"}

class NLS:
    """This provides a wrapper for scipy.optimize.least_squares to get the relevant output for nonlinear least squares.
    Although scipy provides curve_fit for that reason, curve_fit only returns parameter estimates and covariances.
    This wrapper returns numerous statistics and diagnostics"""

    def __init__(self, func, p0, xdata, ydata, bounds=None, method='trf', loss='soft_l1'):
        # Check the data
        if len(xdata) != len(ydata):
            msg = "The number of observations does not match the number of rows for the predictors"
            raise ValueError(msg)

        # Check parameter estimates
        if type(p0) != dict:
            msg = "Initial parameter estimates (p0) must be a dictionry of form p0={'a':1, 'b':2, etc}"
            raise ValueError(msg)
        if 'order' not in p0.keys():
            msg = "Initial parameter estimates (p0) must contain an 'order' list"
            raise ValueError(msg)

        # Check method and loss
        if method not in infodict.keys():
            msg = "Unknown method name"
            raise ValueError(msg)
        if loss not in infodict.keys():
            msg = "Unknown loss function name"
            raise ValueError(msg)

        self.func = func
        self.parmNames = p0['order']
        self.inits = np.array([p0[name] for name in self.parmNames])
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.nobs = len(ydata)
        self.nparm = len(self.inits)
        self.bounds = bounds
        self.method = method
        self.loss = loss
        self.fitted = False

        # Set loss to linear for LM fitting
        if (self.method == 'lm'):
            self.loss = 'linear'

        # Truncate parameter names to 8 characters
        for i, name in enumerate(self.parmNames):
            if len(name) > 8:
                self.parmNames[i] = self.parmNames[i][0:8]

    def fit(self):
        # Run the model     
        if (self.bounds == None) or (self.method == 'lm'):
            self.mod1 = least_squares(self.func, self.inits, method=self.method, loss=self.loss, args=(self.xdata, self.ydata))
        else:
            self.mod1 = least_squares(self.func, self.inits, method=self.method, bounds=self.bounds, loss=self.loss, args=(self.xdata, self.ydata))

        if self.mod1.success:
            self.fitted = True
        else:
            raise RuntimeError("Optimal parameters not found: {:s}".format(self.mod1.message))

        # Modified Jacobian matrix at the solution
        self.jac = self.mod1.jac

        # Vector of residuals at the solution
        self.fun = self.mod1.fun

        # Get the fitted parameter estimates
        self.parmEsts = np.round(self.mod1.x, 6)

        # Calculate the Variances
        self.SS_err = np.sum(self.fun**2)
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
        u, s, vh = linalg.svd(self.jac, full_matrices=False)
        self.cov = self.MSE_err * np.dot(vh.T / s**2, vh)
 
        # Get parameter standard errors
        self.parmSE = np.sqrt(np.diag(self.cov))

        # Calculate the t-values and their p-values for parameters
        self.tvals = self.parmEsts / self.parmSE
        self.pvals = 2 * (1 - spst.t.cdf(np.abs(self.tvals), self.df_err))

        # Calculate F-value and its p-value
        self.fvalue = self.MSE_par / self.MSE_err
        self.pvalue = (1 - spst.f.cdf(self.fvalue, self.df_par, self.df_err))
        return
 
    # Get AIC or BIC. Add 1 to the number of parameters to account for estimation of standard error
    def ic(self, typ='a'):
        # Get biased variance (MLE) and calculate log-likelihood
        n = self.nparm + 1
        s2b = self.SS_err / self.nobs
        self.logLik = -0.5 * self.nobs * np.log(2*np.pi) - 0.5 * self.nobs * np.log(s2b) - 1/(2*s2b) * self.SS_err
        if typ == 'a':
            return 2 * n - 2 * self.logLik
        elif typ == 'b':
            return np.log(self.nobs) * n - 2 * self.logLik
        else:
            msg = "Type must be 'a' or 'b'"
            raise ValueError(msg)

    # Print the summary to stdout
    def tout(self):
        return self._summary_output(sys.stdout.write)

    # Print the summary to file
    def fout(self, outfile):
        with open(outfile, 'w') as f:
            self._summary_output(f.write)
        return
 
    def _summary_output(self, pf):
        if self.fitted:
            pf("\n===============================================================\n")
            pf("Non-linear least squares regression\n")
            pf("Model: '{:s}'\n".format(self.func.func_name))
            pf("{:s}\n".format(infodict[self.method]))
            pf("Info: {:s}\n".format(self.mod1.message))
            pf("Parameters:\n")
            pf("  Factor       Estimate       Std Error      t-value    P(>|t|)\n")
            for i, name in enumerate(self.parmNames):
                    pf("  {:10s}  {: .6e}  {: .6e}  {: 8.5f}  {:8.5f}\n".format(name, self.parmEsts[i], self.parmSE[i], self.tvals[i], self.pvals[i]))
            pf("\nResidual Standard Error: {:8.5f}\n".format(self.RMSE_err))
            pf("                    AIC: {:8.5f}\n".format(self.ic(typ='a')))
            pf("                    BIC: {:8.5f}\n\n".format(self.ic(typ='b')))
            pf("Analysis of Variance:\n")
            pf("  Source     DF   SS        MS         F-value   P(>F)\n")
            pf("  Model     {:3d}  {:8.5f}  {:8.5f}  {:9.5f}  {:8.5f}\n".format(self.df_par, self.SS_par, self.MSE_par, self.fvalue, self.pvalue))
            pf("  Error     {:3d}  {:8.5f}  {:8.5f}\n".format(self.df_err, self.SS_err, self.MSE_err))
            pf("  Total     {:3d}  {:8.5f}\n".format(self.df_tot, self.SS_tot))
            pf("===============================================================\n\n")
        else:
            pf("\n===============================================================\n")
            pf("Non-linear least squares regression\n")
            pf("Model: '{:s}'\n".format(self.func.func_name))
            pf("RUN FIT FOR OUTPUT\n")
            pf("===============================================================\n\n")
        return
