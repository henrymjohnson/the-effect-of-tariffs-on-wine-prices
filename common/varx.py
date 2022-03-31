import pandas as pd
import numpy as np

class VARX:
    def __init__(self, df, endog, exog, lag, x_lag):
        df = self.df
        endog = self.endog
        exog = self.exog
        lag = self.lag
        x_lag = self.x_lag

    
    def fit(self, include_mean=True, fixed=np.NaN, output=True):
        x, y = self.separate_endog_and_exog(self.df)
        p = self.lag
        m = self.x_lag

        # calculate dimensions for exogenous variables
        if x.shape[1] < 1:
            m = -1
            k_x = 0
        else:
            k_x = x.shape[1]
        # validate lags are non-negative
        if p < 0:
            p = 0

        ist = np.max([p, m])
        n_t = y.shape[0]
        k = y.shape[1]
        y_t = y[ist:n_t, ]

        xmtx = np.NaN

        if include_mean:
            xmtx = np.ones(n_t - ist + 1)

        # let's define the columns so that we can keep track of which values are for which columns
        cols = self.endog
        cols_exog = self.exog
        
        # add in the lags for endog variables
        if p > 0:
            for i in range(p):
                xmtx = np.concatenate((xmtx, y[(ist-i-1):(n_t-i-1),]), axis=1)
                if i > 0:
                    new_cols = [c + '_l' + str(i) for c in self.endog]
                    # update list of column names
                    cols = np.concatenate([new_cols, cols])
        # add in the lags for exog variables
        if m > -1:
            for i in range(m+1):
                xmtx = np.concatenate((xmtx, x[(ist-i):(n_t-i),]), axis=1)
                # update exog column names
                if i > 0:
                    new_cols_exog = [c + '_l' + str(i) for c in self.exog]
                    cols_exog = np.concatenate([new_cols_exog, cols_exog])

        # update entire set of columns
        cols = np.concatenate([cols, cols_exog])
        cols = np.concatenate([['intercept'], cols])

        p_1 = xmtx.shape[1]
        nobe = xmtx.shape[0]

        beta = np.zeros((p_1, k))
        se_beta = np.ones((p_1, k))
        resi = y_t
        n_par = 0

        if (np.isnan(fixed)):
            xpx = np.dot(xmtx.T, xmtx)
            xpx = xpx.astype(float)
            xpy = np.dot(xmtx.T, y_t)
            xpxi = np.linalg.solve(xpx)
            beta = np.dot(xpxi, xpy)
            resi = y_t - np.dot(xmtx, beta)
            sig = np.matmul(resi.T, resi) / nobe
            co = np.kron(sig, xpxi)
            co_diag = np.diag(co)
            se = co_diag**(1/2)
            se_beta = se.reshape((beta.shape[0], k), order='F')
            n_par = beta.shape[0] * k
            sig = sig.astype(float)
            sig_det = np.linalg.det(sig)
            d_1 = np.log(sig_det)
            aic = d_1 + 2 * n_par / nobe
            bic = d_1 + np.log(nobe) * n_par / nobe
        else:
            beta = np.zeros((p_1, k))
            se_beta = np.ones((p_1, k))
            resi = y_t
            n_par = 0
            for i in range(k):
                id_x = np.argwhere(np.any(fixed[:, i] == 1))
                n_par = n_par + len(id_x)
                if (len(id_x) > 0):
                    x_m = xmtx[:, id_x]
                    y_1 = y_t[:, i].reshape((nobe, 1))
                    xpx = np.dot(x_m.T, x_m)
                    xpy = np.dot(x_m.T, y_1)
                    xpx = xpx.astype(float)
                    xpxi = np.linalg.solve(xpx)
                    beta_1 = np.dot(xpxi, xpy)
                    res = y_1 - np.dot(x_m, beta_1)
                    sig_1 = np.sum(res**2) / nobe
                    sig_1 = sig_1.astype(float)
                    diag_sig_1 = np.diag(xpxi) * sig_1
                    se = diag_sig_1**(1/2)
                    beta[id_x, i] = beta_1
                    se_beta[id_x, i] = se
                    resi[:, i] = res
            sig = np.matmul(resi.T, resi) / nobe
            sig = sig.astype(float)
            d_1 = np.log(np.linalg.det(sig))
            aic = d_1 + 2 * n_par / nobe
            bic = d_1 + np.log(nobe) * n_par / nobe
        
        ph0 = np.NaN
        icnt = 0

        if (include_mean):
            ph0 = beta[0,:]
            icnt = icnt + 1
            print('Constant Term:')
            print('est:', [np.round(i, 4) for i in ph0])
            print('se:', [np.round(i, 4) for i in se_beta[0,:]])
        
        phi = np.NaN

        if (p > 0):
            phi = beta[(icnt):(icnt+k*p),].T
            se_phi = se_beta[(icnt):(icnt+k*p),].T
            for i in range(1, p+1):
                print('AR(' + str(i) + ') matrix:')
                jcnt = (i-1) * k
                print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in phi[:,jcnt:(jcnt+k)]]))
                print('standard errors:')
                print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in se_phi[:,jcnt:(jcnt+k)]]))
            icnt = icnt + k * p

        if (m > -1):
            print('coefficients of exogenous variables:')
            beta_exog = beta[icnt:icnt+(m+1)*k_x,:].T
            se_beta_exog = se_beta[icnt:icnt+(m+1)*k_x,:].T
            if (k_x == 1):
                beta_exog = beta_exog.T
                se_beta_exog = se_beta_exog.T
            for i in range(m):
                jdx = i * k_x
                print('l' + str(i), 'coefficient matrix')
                print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in beta_exog[:,jdx:(jdx+k_x)]]))
                print('standard errors:')
                print('\n'.join([' '.join(['{:.3f}'.format(item) for item in row]) for row in se_beta_exog[:,jdx:(jdx+k_x)]]))

        print('Residual Covariance Matrix:')
        print('\n'.join([' '.join(['{:.5f}'.format(item) for item in row]) for row in sig]))
        print('Information Criteria:')
        print('AIC:', '{:.3f}'.format(aic))
        print('BIC:', '{:.3f}'.format(bic))

        self.coef = beta
        self.se_coef = se_beta
        self.residuals = resi
        self.sigma = sig
        self.beta_exog = beta_exog
        self.ph0 = ph0
        self.phi = phi

        return beta, se_beta, resi, sig, beta_exog, ph0, phi

    
    def transform(self):
        y_pred = []

        return y_pred

    
    def irf(self):
        print('')

    
    def separate_endog_and_exog(self):
        endog_data = self.df[self.endog]
        exog_data = self.df[self.exog]

        # convert dfs to matrices
        x = exog_data.to_numpy()
        y = endog_data.to_numpy()

        return x, y