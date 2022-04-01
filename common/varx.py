import pandas as pd
import numpy as np

class VARX:
    def __init__(self, df, endog, exog, lag, x_lag):
        df = self.df
        endog = self.endog
        exog = self.exog
        lag = self.lag
        x_lag = self.x_lag

    
    def fit(self, include_mean=True, fixed=np.NaN):
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
        self.x = x
        self.y = y
        self.ran_fit = True

        return y, x, beta, se_beta, resi, sig, beta_exog, ph0, phi

    
    def transform(self, step, new_x=np.NaN, origin=0):
        if (np.isnan(self.ran_fit)):
            print('You need to fit the model')
            return

        y = self.y
        x = self.x
        n_t = self.y.shape[0]
        k = self.y.shape[1]
        dim_x = self.x.shape[1]
        se = np.NaN
        ph0 = self.ph0.reshape((3, 1))

        if (len(ph0) < 1):
            ph0 = np.zeros(k, 1)
        if (step < 1):
            step = 1
        if (origin < 1):
            origin = n_t
        
        if (~np.isNaN(new_x)):
            # estimate values
            step = np.min([new_x.shape[0], step])
            y_new = self.y[0:(origin)]
            if (dim_x > 1):
                x = np.concatenate((self.x[0:(origin)], new_x), axis=0)

            for i in range(step):
                t_p = ph0
                t_i = origin + i
                for i in range(1, self.lag):
                    id_x = (i-1) * self.lag
                    t_p = t_p + np.dot(self.phi[:, (id_x+1):(id_x+k)], y_new[t_i - i, :].reshape(dim_x, 1))
                if (self.x_lag > -1):
                    for i in range(self.x_lag):
                        id_x = i * dim_x
                        t_p = t_p + np.dot(self.beta[:, (id_x+1):(id_x+dim_x)], x[t_i - i, :].reshape(dim_x, 1))
                y_new = np.concatenate((y_new, t_p), axis=0)

            # standard errors of predictions
            weights = self.psi_weights(self.phi, step)
            se = np.diag(self.sig)**(1/2)
            se = se.reshape(1, k)
            if step > 1:
                for i in range(2, step):
                    id_x = (i-1) * k
                    wk = weights[:, (id_x):(id_x+k)]
                    si = si + np.linalg.multi_dot([wk, self.sig, wk.T])
                    se1 = np.diag(si)**(1/2)
                    se1 = se1.reshape(1, k)
                    se = np.concatenate((se, se1), axis=0)

            print('Prediction at Origin:', origin)
            print('Point forecasts:')
            print('\n'.join([' '.join(['{:.4f}'.format(item) for item in row]) for row in y_new[:, (origin+1):(origin+step)]]))
            print('Standard errors:')
            print('\n'.join([' '.join(['{:.4f}'.format(item) for item in row]) for row in se[0:step,:]]))
        
        self.pred_errors = se

    
    def irf(self):
        print('')

    
    def separate_endog_and_exog(self):
        endog_data = self.df[self.endog]
        exog_data = self.df[self.exog]

        # convert dfs to matrices
        x = exog_data.to_numpy()
        y = endog_data.to_numpy()

        return x, y


    def psi_weights(phi, lag):
        k = phi.shape[0]
        m = phi.shape[1]
        p = np.floor(m/k)
        si = np.zeros((k, k))
        np.fill_diagonal(si, 1)
        if (p < 1):
            p = 1
        if (lag < 1):
            lag = 1
        
        for i in range(1, (lag+1)):
            if (i < (p+1)):
                id_x = (i-1) * k
                t_p = phi[:, (id_x):(id_x+k)]
            else:
                t_p = np.zeros((k, k))
            jj = i-1
            jp = np.minimum(jj, p).astype(int)
            if (jp > 0):
                for j in range(1, (jp+1)):
                    jd_x = (j-1) * k
                    id_x = (i-j) * k
                    w1 = phi[:, (jd_x):(jd_x+k)]
                    w2 = si[:, (id_x):(id_x+k)]
                    t_p = t_p + np.dot(w1, w2)
            si = np.concatenate((si, t_p), axis=1)
        
        return si
