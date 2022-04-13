import pandas as pd
import numpy as np

class vector_ar:
    def __init__(self, df, endog, exog, lag):
        df = self.df
        data = self.build_lagged_data(df)
        endog = self.endog
        exog = self.exog
        lag = self.lag

    
    def fit(self):
        x, y = self.separate_endog_and_exog(self)
        A = []
        for i in range(y.shape[1]):
            y_endog = y[:, i]
            y_exog = y[:, ~i]
            x_complete = x.concatenate(y_exog)

            betas = self.estimate_betas(y_endog, x_complete)
            A[i] = betas
        
        return A

    
    def transform(self, A):
        x, y = self.separate_endog_and_exog(self)
        pred_endog = []

        for i in range(A.shape[1]):
            y_endog = y[:, i]
            y_exog = y[:, ~i]
            x_complete = x.concatenate(y_exog)

            pred_endog[i] = np.matmul(x_complete, A[i])

        return pred_endog


    def fit_transform(self):
        A = self.fit(self)
        pred_endog = self.transform(self, A)

        return pred_endog


    def build_lagged_data(self):
        data = pd.DataFrame(self.data)
        data.set_index('month')

        # build dataset with lags
        for i in range(1, self.lag+1):
            for j in data.columns:
                # add lag i of feature j to the dataframe
                data[f'{j}_lag_{i}'] = data[j].shift(i)
        data.dropna(inplace=True)

        return data


    def separate_endog_and_exog(self):
        endog_data = self.data[self.endog]
        exog_data = self.data[self.exog]
        # add intercept
        exog_data.insert(0, 'intercept', 1)

        # convert dfs to matrices
        x = exog_data.to_numpy()
        y = endog_data.to_numpy()

        return x, y

    
    def estimate_betas(y, x):
        xTx = np.matmul(x.T, x)
        xTy = np.matmul(x.T, y)
        xTx_i = np.linalg.inv(xTx)
        
        return np.matmul(xTx_i, xTy)


