seeds = list(range(30))
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import seaborn as sns
#from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
#from sklearn import linear_model
#from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
#from sklearn.metrics import classification_report,confusion_matrix
#from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
#from  statsmodels.api import OLS
#import statistics as stat
from joblib import Parallel, delayed
import time
import warnings
import datetime
warnings.filterwarnings('ignore')


class LOAN_base:
    def __init__(self, subnet_struct, initial="OLS", fit_intercept=True, optimizer="LocalLinear_y",
                 ModelSelection='best_val'):
        """
        Initialize the model object
        :param subnet_struct: list, specify the hidden layers structure using brackets, e.g. [k, l, m] three hidden layers
        with first layer has k nodes, second layer has l nodes and third layer has m nodes
        :param initial: string, initializer for weights and biases, OLS: ordinary least squares, SIR: sliced inverse regression
        :param fit_intercept: bool, if True, an intercept will be included in fitting linear regression and False otherwise
        :param optimizer: string, LocalLinear_r, LocalLinear_y, Adam, Adamax
        """
        self.n_subnets = len(subnet_struct)
        self.subnet_struct = subnet_struct
        self.yhat = []
        self.b = []  # beta value for initial linear regression
        self.logistic = False
        self.initial = initial
        self.fit_intercept = fit_intercept

        self.max_iter = 100
        self.epsilon = 1e-5
        self.optimizer = optimizer
        self.ModelSelection = ModelSelection
        # if self.n_subnets == 1: self.ModelSelection= [ModelSelection]
        # else: self.ModelSelection = [ModelSelection] * (self.n_subnets - 1) + ['model_averaging']

        self.weight = {}
        self.bias = {}
        self.beta = {}
        self.intercept = {}
        self.history = {}
        self.weight_record = None
        self.bias_record = None
        self.beta_record = None
        self.train_yhat = [[] for _ in range(self.n_subnets)]
        self.val_yhat = [[] for _ in range(self.n_subnets)]
        self.test_yhat = [[] for _ in range(self.n_subnets)]
        self.mse_old = 0.0
        self.train_loss = []
        self.val_loss = []

        self.train_nu = None
        self.val_nu = None
        if self.initial == "SIR":
            self.sir = SlicedInverseRegression(n_directions=1)

        if self.n_subnets == 0:
            raise ValueError("Please specified at least one hidden layer.")

    def lm(self, _x, _y, _nu=None, fit_intercept = True):
        _n = len(_y)
        if fit_intercept:
            _x = np.c_[np.ones((_n, 1)), _x]
        if _nu is None:
            xT = np.transpose(_x) / _n
        else:
            xT = np.transpose(np.multiply(_nu, _x))
        xTx = np.dot(xT, _x)
        # eignvalues = np.linalg.eig(xx_xx)[0]
        # print(max(eignvalues)/min(eignvalues))
        xx_inv = np.linalg.pinv(xTx)
        corr = np.dot(xT, _y)
        temp = np.dot(xx_inv, corr)

        _yhat = np.dot(_x, temp)
        if fit_intercept:
            return {'coef_': temp[1:], 'intercept_': temp[0], 'yhat': _yhat}
        else:
            return {'coef_': temp, 'intercept_': 0.0, 'yhat': _yhat}

    def __init_bias(self, nodes, _yhat, _y=None):
        tau = np.quantile(_yhat, [i / (nodes + 1) for i in range(1, (nodes + 1))])
        if self.initial in ["OLS", "SIR"]:
            return -tau.reshape(-1, 1)
        else:
            new_x = self.transform_x(_yhat, tau=tau)
            self.b = abs(self.lm(new_x, _y, _nu=self.train_nu)['coef_'])
            tau = np.array(tau).reshape(self.b.shape)
            return -np.multiply(self.b, tau)

    def __init_weight(self, fit0, J):
        if self.initial == "SIR":
            ALPHA = np.repeat(self.sir.directions_.reshape(1, -1), J, axis=0)
            return ALPHA
        else:
            ALPHA = np.repeat(fit0['coef_'].reshape(1, -1), J, axis=0)
            if self.initial == "OLS":
                return ALPHA
            else:
                return np.multiply(self.b, ALPHA)

    def Node(self, x, W, b, beta=None):
        _b = np.repeat(b.reshape(1, -1), x.shape[0], axis=0)
        _sigma = np.maximum((np.dot(x, W.T) + _b), 0)
        if beta is None:
            return _sigma
        else:
            if len(beta) == len(b):  # check dimension
                _beta = np.repeat(beta.reshape(1, -1), x.shape[0], axis=0)
                return _beta * _sigma
            else:
                raise ValueError('Dimension of beta not match')

    def transform_x(self, X, tau, main_effect=False, V=False):
        nknots = len(tau)

        # generate x1 and x2
        new_x1 = np.maximum(X - tau, 0)
        new_x2 = -1 * ((X - tau) > 0)
        knots = [str(i) for i in range(nknots)]

        if V:  # Indicator part of approximation
            sname = ['U', 'V']
            splines = np.c_[new_x1, new_x2]
            new_x = pd.DataFrame(splines, columns=[x + y for x in sname for y in knots])
        else:
            new_x = pd.DataFrame(new_x1, columns=["U" + y for y in knots])
        if main_effect:
            new_x['X'] = X.values
            # varname = X.columns.values.tolist() + [x + y for x in sname  for y in knots]
            # new_x = pd.DataFrame(np.c_[X, splines], columns=varname)

        # if any(new_x.T.duplicated()):
        #    print("Exist duplicated variables in new_X")
        return new_x

    def __get_initial_values(self, _x, _y, _J, _nu=None):
        # Get initial value of bias and weight: conduct a linear regress y on x and
        # obtain a rough estimate of \hat{\alpha}
        fit0 = self.lm(_x, _y, _nu)
        if self.initial == "SIR":
            self.sir.fit(_x, np.ravel(_y))
            _yhat = self.sir.transform(_x)
        else:
            _yhat = fit0['yhat']
        b0 = self.__init_bias(_J, _yhat, _y)  # initial value for bias
        w0 = self.__init_weight(fit0, _J)  # initial value for weight
        return w0, b0, _yhat

    def initial_values(self, _x, _y, _J):
        # Get initial value of bias and weight:
        if isinstance(_J, int):
            _wt, _bi, _ = self.__get_initial_values(_x, _y, _J)
            return _wt, _bi
        else:
            # Use LocalLinear_r to initialize on J[0] and broadcast to J[1]
            _wt, _bi, _yhat = self.__get_initial_values(_x, _y, _J[0])
            _beta = np.zeros(_J[0])
            n, p = _x.shape
            t = 0
            D = 1a
            mse_hist = [mean_squared_error(_y, _yhat)]
            _weight = _wt  # Only record the best result
            _bias = _bi
            while (t < self.max_iter) & (D > 1e-2):
                t = t + 1
                Z, m_h = self.__get_Z(_x, _wt, _bi, _beta)
                r_c = _y - m_h
                # s = np.maximum(np.power(start_from, (t - 1)), to_min)  #############Least Sq. Alg. 1
                s = np.maximum(np.power(0.99, t), 0.5)  # if layer ==0 else 0.01 #*0.05
                fit0 = self.lm(Z, r_c, _nu=self.train_nu)
                _theta = s * np.ravel(fit0['coef_'])  # s * np.ravel(self.lm(Z, r_c))

                theta1 = _theta[:_J[0]]
                gamma = _theta[_J[0]:(2 * _J[0])]
                eta = _theta[(2 * _J[0]):]
                eta = eta.reshape(-1, p)  # J*p
                _beta = theta1.reshape(_beta.shape) + _beta
                for i, val in enumerate(_beta):
                    if abs(val) > 0.001:
                        _bi[i] += gamma[i] / _beta[i]  # J*1
                        # Update weight
                        _delta = eta[i, :] / _beta[i]
                        _wt[i, :] = _wt[i, :] + _delta  # J*p

                train_mh = self.Node(_x, _wt, _bi)
                refit = self.lm(train_mh, _y, _nu=self.train_nu)
                train_yhat = refit['yhat']
                train_mse = mean_squared_error(_y, train_yhat)
                D = abs(train_mse - mse_hist[-1]) / mse_hist[-1]
                mse_hist.append(train_mse)
                if train_mse <= min(mse_hist):
                    # store best results
                    _weight = _wt
                    _bias = _bi

            _rep = round(_J[1] / _J[0])
            J_new = _J[0] * _rep
            if _J[1] % _J[0] != 0:
                print("Warning: the number of knots equals:", J_new)
                _J[1] = J_new
            for _j in range(_J[0]):
                _yhat = np.dot(_x, _weight[_j, :])
                _b_temp = self.__init_bias(_rep, _yhat)
                _bi1 = _b_temp if _j == 0 else np.append(_bi1, _b_temp)
            _wt1 = np.repeat(_weight, _rep, axis=0)
            return _wt1, _bi1

    def __get_Z(self, _X, _W, _bias, _beta=None):
        z1 = self.Node(_X, _W, _bias)  # Calculate hidden-layer
        z2 = z1.copy()  # Indicator for hidden layer
        z2[z2 > 0] = 1  # If the output of hiddenlayer >0 then set as 1
        n, J = z2.shape
        z3 = np.empty((n, 0))  # n * (p * J)
        for j in range(J):  # Generate the design matrix for eta node by node
            z3 = np.c_[z3, np.multiply(_X, z2[:, j].reshape(-1, 1))]
        Z = np.c_[z1, z2, z3]
        if _beta is None:
            return Z
        else:
            if len(_beta) == z1.shape[1]:
                mh = np.dot(z1, _beta).reshape(-1, 1)
                return Z, mh
            else:
                raise ValueError("beta dimension not match with z1")

    def __record(self, _W, _bias, _beta, _train_mse):
        self.bias_record = np.append(self.bias_record, [_bias.flatten()], axis=0)
        self.weight_record = np.append(self.weight_record, [_W.flatten()], axis=0)
        self.beta_record = np.append(self.beta_record, [_beta.flatten()], axis=0)
        self.train_loss.append(_train_mse)

    def __val_results(self, _val_X, _val_y, _W, _bias, _beta=None):
        val_mh = self.Node(_val_X, _W, _bias, _beta)
        val_yhat = np.dot(val_mh, self.refit_mod['coef_']) + self.refit_mod['intercept_']
        val_mse = mean_squared_error(_val_y, val_yhat)
        self.val_loss.append(val_mse)
        return val_mh

    def LocalLinear_r(self, x, y, _wt, _bi, _beta, layer, J, val_x=None, val_y=None, learn_rate=None, LSE = True):
        n, p = x.shape
        t = 0
        D = 1
        # start_from = 0.5**layer if layer > 0 else 0.99
        # to_min = 0.001 if layer >0 else 0.5
        while (t < self.max_iter) & (D > self.epsilon):
            t = t + 1
            Z, m_h = self.__get_Z(x, _wt, _bi, _beta)
            r_c = y - m_h.reshape(y.shape)
            # s = np.maximum(np.power(start_from, (t - 1)), to_min)  #############Least Sq. Alg. 1
            if learn_rate is None:
                learn_rate = np.maximum(np.power(0.99, t), 0.5)  # step size
            if LSE:
                # Use full least square estimator (X)
                _theta = learn_rate * np.ravel(self.lm(Z, r_c, _nu=self.train_nu, fit_intercept=self.fit_intercept)['coef_'])
            else:
                _theta = np.zeros(J*(p+2))
                for j in range(J):
                    j_idx = range(j, J*(p+2), J)
                    if self.train_nu is None:
                        _theta[j_idx] = learn_rate * np.ravel(self.lm(Z[:, j_idx], r_c, fit_intercept=False)['coef_'])
                    else:
                        _theta[j_idx] = learn_rate * np.ravel(
                            self.lm(Z[:, j_idx], r_c, _nu=self.train_nu[j_idx], fit_intercept=False)['coef_'])

            theta1 = _theta[:J]
            gamma = _theta[J:(2 * J)]
            eta = _theta[(2 * J):]
            eta = eta.reshape(-1, p)  # J*p
            _beta = theta1.reshape(_beta.shape) + _beta
            for i, val in enumerate(_beta):
                if abs(val) > 0.001:
                    _bi[i] += gamma[i] / _beta[i]  # J*1
                    # Update weight
                    _delta = eta[i, :] / _beta[i]
                    _wt[i, :] = _wt[i, :] + _delta  # J*p

            train_mh = self.Node(x, _wt, _bi)
            self.refit_mod = self.lm(train_mh, y, _nu=self.train_nu)
            train_yhat = self.refit_mod['yhat']
            train_mse = mean_squared_error(y, train_yhat)
            self.__record(_wt, _bi, _beta, train_mse)
            if val_x is not None:
                val_mh = self.__val_results(val_x, val_y, _wt, _bi)

            _yhat = np.dot(train_mh, _beta)
            _mse = mean_squared_error(y, _yhat)
            D = abs(_mse - self.mse_old) / self.mse_old
            self.mse_old = _mse

        if val_x is None: val_mh = None
        return train_mh, val_mh, _wt, _bi, self.refit_mod['coef_'], t

    def LocalLinear_y(self, x, y, _wt, _bi, J, learn_rate=None, val_x=None, val_y=None, LSE = True):
        n, p = x.shape
        D = 10
        n_iter = 0
        val_mh = None
        if learn_rate is None:
            learn_rate = np.maximum(np.power(0.99, n_iter), 0.5)  # step size
        while (D > self.epsilon) & (n_iter < self.max_iter):
            n_iter += 1  # record the number of current iteration
            Z = self.__get_Z(x, _wt, _bi)
            if LSE:
                # Use full least square estimator (X)
                subnet_coef = learn_rate * np.ravel(self.lm(Z, y, _nu=self.train_nu, fit_intercept=self.fit_intercept)['coef_'])
            else:
                subnet_coef = np.zeros(J*(p+2))
                for j in range(J):
                    j_idx = range(j, J*(p+2), J)
                    if self.train_nu is None:
                        temp = learn_rate * np.ravel(self.lm(Z[:, j_idx], y, fit_intercept=False)['coef_'])
                        subnet_coef[j_idx] = temp
                    else:
                        subnet_coef[j_idx] = learn_rate * np.ravel(
                            self.lm(Z[:, j_idx], y, _nu=self.train_nu[j_idx], fit_intercept=False)['coef_'])

            #subnet_coef = self.lm(Z, y, _nu=self.train_nu)['coef_']  # m_{h_1}(x)
            _beta = subnet_coef[:J]  # J*1
            # beta_sign = np.sign(bb)
            # beta = (np.abs(bb) + 0.001) * beta_sign
            gamma = subnet_coef[J:(2 * J)]  # J*1
            eta = subnet_coef[(2 * J):]
            eta = eta.reshape(-1, p)  # J*p
            if self.optimizer == "LocalLinear_sgn_y":
                _beta = np.sign(_beta)
                _bi = _bi + _beta * gamma  # update bias
                _wt = _wt + eta * _beta  # Update weight

                train_mh = self.Node(x, _wt, _bi, _beta)
                self.refit_mod = self.lm(train_mh, y, _nu=self.train_nu)
                self.yhat = self.refit_mod['yhat']  # Refit
                # yhat2 = np.sum(train_mh, axis=1)  # Without refit

                if val_x is not None:
                    val_mh = self.__val_results(val_x, val_y, _wt, _bi, _beta)

            elif self.optimizer == "LocalLinear_y":
                # Update bias
                for _i, _v in enumerate(_beta):
                    if abs(_v) > 0.001:
                        _bi[_i] += gamma[_i] / _v   # J*1
                        # Update weight
                        _delta = eta[_i, :] / _v
                        _wt[_i, :] = _wt[_i, :] + _delta   # J*p

                # Refit
                train_mh = self.Node(x, _wt, _bi)
                self.refit_mod = self.lm(train_mh, y, _nu=self.train_nu)
                self.yhat = self.refit_mod['yhat']
                _beta = self.refit_mod['coef_']
                if val_x is not None:
                    val_mh = self.__val_results(val_x, val_y, _wt, _bi)

            mse_new = mean_squared_error(y, self.yhat)
            D = abs(mse_new - self.mse_old) / self.mse_old
            self.mse_old = mse_new
            self.__record(_wt, _bi, _beta, mse_new)
        return train_mh, val_mh, _wt, _bi, _beta, n_iter

    def ADAM(self, x, y, w0, b0, beta0, J, val_x=None, val_y=None,
             learn_rate=0.001, epsilon0=1e-8, nu_1=0.9, nu_2=0.999, u=0, v=0, norm_l2=False):
        '''
        If Adam:
        initialize $m_0$ as 1st moment vector
        initialize $v_0$ as 2nd moment vector

        The update rule for $\theta$ with gradient $g$ uses an optimization
        described at the end of section 2 of the paper:

        $$lr_t = \mathrm{learning\_rate} *
        \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
        $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
        $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
        $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

        If Adamax:
        initialize $m_0$ as 1st moment vector
        initialize $v_0$ as 2nd moment vector
        initialize $\hat{v}_0$ as 2nd moment vector

        The update rule for $\theta$ with gradient $g$ uses an optimization
        described at the end of section 2 of the paper:

        $$lr_t = \mathrm{learning\_rate} *
        \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$

        $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
        $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
        $$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$
        $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{\hat{v}_t} + \epsilon)$$
        '''
        n, p = x.shape
        if norm_l2:  # Assume bias^2 + ||weight||^2 = 1
            bi_l2 = b0 ** 2
            wt_l2 = np.sum(w0 ** 2, axis=1)
            l2 = np.sqrt(wt_l2.reshape(bi_l2.shape) + bi_l2)
            _bi = b0 / l2
            _wt = w0 / l2.reshape(-1, 1)
            _beta = -l2
        else:
            _bi = b0.copy()
            _wt = w0.copy()
            _beta = beta0.copy()
        self.beta_record = np.array([_beta.flatten()])

        gamma = _bi * _beta.reshape(-1, 1)
        eta = _wt * _beta.reshape(-1, 1)
        temp = np.append(_beta, gamma)
        _theta = np.append(temp, eta.flatten())

        t = 0  # number of iteration
        D = 1
        while (t < self.max_iter) & (D > self.epsilon):
            t = t + 1
            Z, m_h = self.__get_Z(x, _wt, _bi, _beta)  # calculate Z and m_h of current layer
            r_c = y - m_h
            # delta = self.lm(Z, r_c).reshape(_theta.shape)
            # _theta = delta + _theta
            Z_T = Z.T  # / n
            D = np.dot(Z_T, Z)

            g_c = np.dot(Z_T, r_c)
            g_c2 = np.multiply(g_c, g_c)
            # g_c2 = np.diag(D).reshape(-1, 1)
            # step = np.maximum(np.power(learn_rate, t), 0.01)
            step = learn_rate
            u_new = nu_1 * u + (1 - nu_1) * g_c
            u_hat = u_new / (1 - nu_1 ** t)
            if self.optimizer == "Adamax":  # Use Adamax
                g_c_tilda = np.abs(g_c)  # + epsilon0
                # g_c_tilda = np.sqrt(g_c2)
                v_new = np.maximum((nu_2 * v), g_c_tilda)
                _theta = 2 * step * u_hat / v_new  # step = 2* learn_rate

            else:  # original Adam
                v_new = nu_2 * v + (1 - nu_2) * g_c2
                v_hat = v_new / (1 - nu_2 ** t)
                _theta = step * u_hat / (np.sqrt(v_hat) + epsilon0)
                # print('u_hat:',u_hat[:5])

            u = u_new
            v = v_new
            _theta1 = _theta[:J]
            gamma = _theta[J:(2 * J)]
            eta = _theta[(2 * J):]
            eta = eta.reshape(-1, p)  # J*p
            _beta = _theta1.reshape(_beta.shape) + _beta

            for i, val in enumerate(_beta):
                if abs(val) > 0.001:
                    _bi[i] += gamma[i] / val  # J*1
                    # Update weight
                    delta = eta[i, :] / val
                    _wt[i, :] = _wt[i, :] + delta  # J*p
            # print("weight:",_wt[1,:5])
            if norm_l2:
                # Assume bias^2 + ||weight||^2 = 1
                sgn_beta = np.sign(_beta)
                bi_l2 = _bi ** 2
                wt_l2 = np.sum(_wt ** 2, axis=1)
                l2 = np.sqrt(bi_l2 + wt_l2)
                _bi = _bi / l2
                _wt = _wt / l2.reshape(-1, 1)
                _beta = sgn_beta * l2

            # Refit linear model without intercept
            train_mh = self.Node(x, _wt, _bi)
            self.refit_mod.fit(train_mh, y)
            train_yhat = self.refit_mod.predict(train_mh)
            train_mse = mean_squared_error(y, train_yhat)
            self.__record(_wt, _bi, _beta, train_mse)

            if val_x is not None:
                val_mh = self.__val_results(_val_X=val_x, _val_y=val_y, _W=_wt, _bias=_bi)

            # Calculate stopping criteria without refit
            _yhat = np.dot(train_mh, _beta)
            _mse = mean_squared_error(y, _yhat)
            D = abs(_mse - self.mse_old) / self.mse_old
            self.mse_old = _mse

        if val_x is None:
            val_mh = None
        return train_mh, val_mh, _wt, _bi, _beta, t

    def build_subnets(self, layer, J, train_X_tilde, train_y, learn_rate, val_X_tilde=None, val_y=None,
                      test_X_tilde=None, test_y=None, LSE = True):
        n, p = train_X_tilde.shape
        # Get initial value for weight, bias and beta
        w0, b0 = self.initial_values(train_X_tilde, train_y, J)
        if isinstance(J, list):  # Specify the true number of nodes in this layer
            J = J[1]
        beta0 = np.zeros(J)

        train_mh = self.Node(train_X_tilde, w0, b0)
        self.init_mod = self.lm(train_mh, train_y, _nu=self.train_nu)
        self.mse_old = mean_squared_error(train_y, self.init_mod['yhat'])

        # Record beta, bias, weight and train loss
        self.beta_record = np.array([beta0])
        self.bias_record = np.array([b0.flatten()])
        self.weight_record = np.array([w0.flatten()])
        self.train_loss = [self.mse_old]

        if val_X_tilde is not None:  # Record test loss
            val_mh = self.Node(val_X_tilde, w0, b0)
            val_yhat = np.dot(val_mh, self.init_mod['coef_']) + self.init_mod['intercept_']
            val_mse_old = mean_squared_error(val_y, val_yhat)
            self.val_loss = [val_mse_old]

        if self.optimizer == "LocalLinear_r":
            train_mh, val_mh, _wt, _bi, _beta, n_iter = self.LocalLinear_r(train_X_tilde, train_y, w0, b0, beta0, J,
                                                                           val_x=val_X_tilde, val_y=val_y,
                                                                           learn_rate=learn_rate, LSE = LSE)


        elif 'Adam' in self.optimizer:
            train_mh, val_mh, _wt, _bi, _beta, n_iter = self.ADAM(train_X_tilde, train_y, w0, b0, beta0, J,
                                                                  val_x=val_X_tilde, val_y=val_y)

        else:
            train_mh, val_mh, _wt, _bi, _beta, n_iter = self.LocalLinear_y(train_X_tilde, train_y, w0, b0, J,
                                                                           learn_rate=learn_rate,
                                                                           val_x=val_X_tilde, val_y=val_y, LSE = LSE)

        # Clean up trained layer result
        hist_key = 'subnet' + str(layer)

        if self.ModelSelection == 'best_train':  # use the best training result
            best_idx = np.argmin(self.train_loss)
        elif self.ModelSelection == 'best_val':  # use the best validation result
            best_idx = np.argmin(self.val_loss)
        elif self.ModelSelection == 'last':
            best_idx = n_iter

        if best_idx != n_iter:
            _wt = self.weight_record[best_idx, :].reshape(J, p)
            _bi = self.bias_record[best_idx, :]
            train_mh = self.Node(train_X_tilde, _wt, _bi)
            self.refit_mod = self.lm(train_mh, train_y, _nu=self.train_nu)
            _beta = self.refit_mod['coef_']
            if val_X_tilde is not None:
                val_mh = self.Node(val_X_tilde, _wt, _bi)

        self.weight.update({hist_key: _wt})
        self.bias.update({hist_key: _bi})
        self.beta.update({hist_key: _beta.flatten()})
        self.intercept.update({hist_key: self.refit_mod['intercept_']})
        # print(hist_key, "train MSE:", self.train_loss[best_idx])
        self.train_yhat[layer] = self.refit_mod['yhat']
        if val_X_tilde is not None:
            # print(hist_key, "validation MSE:", self.val_loss[best_idx])
            self.val_yhat[layer] = np.dot(val_mh, _beta) + self.refit_mod['intercept_']

        if test_X_tilde is not None:
            test_mh = self.Node(test_X_tilde, _wt, _bi)
            self.test_yhat[layer] = np.dot(test_mh, _beta) + self.refit_mod['intercept_']
            # print(hist_key, "test MSE:", mean_squared_error(test_y,  self.test_yhat[layer]))
        else:
            test_mh = None

        layer_hst = {'loss': self.train_loss, 'n_iter': n_iter, 'bias_hst': self.bias_record,
                     'weight_hst': self.weight_record, 'beta_hst': self.beta_record}
        if val_X_tilde is not None:
            layer_hst.update({'val_loss': self.val_loss})

        self.history.update({hist_key: layer_hst})
        # if n_iter == self.max_iter:
        #    print(f"Failed to converge before max number of iteration is reached at hidden layer {layer}.")

        return train_mh, val_mh, test_mh

    def weighted(self, _x, _y, _nu):
        _y = _y.reshape(-1, 1)
        if _nu is None:
            return _x, _y
        elif len(_nu) != len(_y):
            raise ValueError("Weight dimension not match! len(weight) != len(y)")
        else:
            _nu = np.sqrt(_nu).reshape(-1, 1)
            return np.multiply(_nu, _x), np.multiply(_nu, _y)

    def fit(self, train_X, train_y, val_X=None, val_y=None, test_X=None, test_y=None, epsilon=1e-6, learn_rate=None,
            max_iter=100, train_nu=None, LSE = True):
        '''
        Fit hidden layers
        :param train_X, val_X: array-like or sparse matrix, shape (n_samples, n_features) Training data
        :param train_y, val_y: array_like, shape (n_samples, 1)
         Target values. Will be cast to X's dtype if necessary
        :param epsilon: control the stopping of the algorithm
        :param learn_rate: float, learning rate. If it is None, use max(0.99^t, 0.5), where t is the number of current iteration.
        This parameter only takes effect when optimzer is either LocalLinear_r or LocalLinear_y
        :param max_iter: integer, the maximum number of iteration
        '''
        self.max_iter = max_iter
        self.epsilon = epsilon
        train_X_tilde = train_X
        val_X_tilde = val_X
        test_X_tilde = test_X
        if train_nu is not None:
            self.train_nu = train_nu.reshape(-1, 1)

        for layer, J in enumerate(self.subnet_struct):
            train_X_tilde, val_X_tilde, test_X_tilde = self.build_subnets(layer, J, train_X_tilde, train_y, learn_rate,
                                                                          val_X_tilde=val_X_tilde, val_y=val_y,
                                                                          test_X_tilde=test_X_tilde, test_y=test_y, LSE = LSE)

        # print('----------------------')

    def predict(self, X, y=None):
        _x = X
        for layer, key in enumerate(self.weight.keys()):
            _w = self.weight[key]
            _b = self.bias[key]
            _h = self.Node(_x, _w, _b)
            _beta = self.beta[key]
            _intcpt = self.intercept[key]
            _x = _h

        _yhat = np.dot(_x, _beta) + _intcpt
        if y is not None:
            print('Test loss (MSE):', mean_squared_error(y, _yhat))
        return _yhat

    def get_yhat(self):
        return np.ravel(self.train_yhat[-1]), np.ravel(self.val_yhat[-1]), np.ravel(self.test_yhat[-1])


class ResNet(LOAN_base):
    def fit(self, train_X, train_y, val_X=None, val_y=None, test_X=None, test_y=None, epsilon=1e-6, learn_rate=None,
            max_iter=100, train_nu=None, LSE = False):
        '''
        Fit hidden layers
        :param train_X, val_X: array-like or sparse matrix, shape (n_samples, n_features) Training data
        :param train_y, val_y: array_like, shape (n_samples, 1)
         Target values. Will be cast to X's dtype if necessary
        :param epsilon: control the stopping of the algorithm
        :param learn_rate: float, learning rate. If it is None, use max(0.99^t, 0.5), where t is the number of current iteration.
        This parameter only takes effect when optimzer is either LocalLinear_r or LocalLinear_y
        :param max_iter: integer, the maximum number of iteration
        '''
        self.max_iter = max_iter
        self.epsilon = epsilon
        train_X_tilde = train_X
        val_X_tilde = val_X
        test_X_tilde = test_X
        if train_nu is not None:
            self.train_nu = train_nu.reshape(-1, 1)

        for layer, J in enumerate(self.subnet_struct):
            if layer == 0:
                train_X_tilde, val_X_tilde, test_X_tilde = self.build_subnets(layer, J, train_X_tilde, train_y,
                                                                              learn_rate,
                                                                              val_X_tilde=val_X_tilde, val_y=val_y,
                                                                              test_X_tilde=test_X_tilde, test_y=test_y, LSE = LSE)
            else:
                train_res = train_y - self.train_yhat[layer - 1]
                val_res = val_y - self.val_yhat[layer - 1] if val_y is not None else None
                test_res = test_y - self.test_yhat[layer - 1] if test_y is not None else None
                train_X_tilde, val_X_tilde, test_X_tilde = self.build_subnets(layer, J, train_X_tilde, train_res,
                                                                              learn_rate,
                                                                              val_X_tilde=val_X_tilde, val_y=val_res,
                                                                              test_X_tilde=test_X_tilde,
                                                                              test_y=test_res, LSE = LSE)

        # print('----------------------')

    def predict(self, X, y=None):
        _x = X
        for layer, key in enumerate(self.weight.keys()):
            _w = self.weight[key]
            _b = self.bias[key]
            _h = self.Node(_x, _w, _b)
            _beta = self.beta[key]
            _intcpt = self.intercept[key]
            if layer == 0:
                _yhat = np.dot(_h, _beta) + _intcpt
            else:
                _yhat += np.dot(_h, _beta) + _intcpt
            _x = _h

        if y is not None:
            print('Test loss (MSE):', mean_squared_error(y, _yhat))
        return _yhat


class LOANSampleSplit:
    def __init__(self, subnet_struct, n_split, weighted_method='linear', fit_intercept=True, ModelSelection='last',
                 delete_ratio=0.5):
        """
        Sample splitting for distributed learning
        :param subnet_struct: model structure of each neural network
        :param n_split: the number of sample splitting
        :param weighted_method: used of calculating alpha, options: linear, quadratic and exponential
        :param fit_intercept: Whether contain intercept in the model
        :param ModelSelection: select what set of weights and bias during updates, options: 'last', 'best_train', 'beat_val'.
        :param delete_ratio: the proportion of sample spliting submodels will be removed from producing the final result.
        """
        self.n_split = n_split
        self.submods = [[] for _ in range(n_split)]
        self.alpha = np.zeros((n_split, 1))
        self.train_yhat = None
        self.val_yhat = None
        self.test_yhat = None
        self.yhat = None
        self.train_mse_o = np.Inf
        self.val_mse_o = np.Inf
        self.test_mse_o = np.Inf
        self.subnet_struct = subnet_struct
        self.weighted_method = weighted_method
        self.fit_intercept = fit_intercept
        self.ModelSelection = ModelSelection
        self.delete_ratio = delete_ratio
        self.keep_split = [True] * n_split

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

        self.Hat = []

    def linear_weight(self, r):
        return np.min(1.0, abs(r))

    def quadratic_weight(self, r):
        return np.min(1.0, r ** 2)

    def logitistic_weight(self, r):
        exp_r = np.exp(abs(r))
        return (exp_r - 1) / (exp_r + 1)

    def get_alpha(self, _y, _yhat):
        _res = np.abs(_y - _yhat.reshape(_y.shape))  # absolute value of residuals
        sigma_hat = np.median(_res) / 0.6745  # robust estimate of std of residuals
        # sigma_hat = np.std(_y - _yhat.reshape(_y.shape))
        _res = _res / sigma_hat
        if self.weighted_method == "linear":
            temp = np.minimum(1.0, _res)
        elif self.weighted_method == "quadratic":
            temp = np.minimum(1.0, _res ** 2)
        else:
            exp_r = np.exp(_res)
            temp = (exp_r - 1) / (exp_r + 1)

        err = np.mean(temp)
        # err = np.sum(np.multiply(_nu, temp)) / np.sum(_nu)
        alpha_m = (1 - err) / err
        # new_nu = np.multiply(_nu, np.exp(alpha_m * temp))
        return alpha_m

    def weighted_predict(self, newX):  # Method1
        if np.prod(newX.shape) == len(newX):
            _n = 1
        else:
            _n = newX.shape[0]
        theta_hat = np.zeros(_n)
        _weights = np.zeros((_n, self.n_split))
        for k, XXinv in enumerate(self.Hat):
            if self.keep_split[k]:
                if _n > 1:
                    xHx = np.array([np.dot(np.dot(newX[i, :], XXinv), newX[i, :]) for i in range(_n)])
                else:
                    xHx = np.dot(np.dot(newX, XXinv), newX)
                xHx_inv = 1 / xHx
                theta_k_hat = self.submods[k].predict(newX)  # x^T \beta

                temp_k = np.multiply(xHx_inv, theta_k_hat)  # (x^T (X_k^T X_k)^-1 x)^-1 x^T \beta

                theta_hat += temp_k
                _weights[:, k] = xHx_inv

        W = np.sum(_weights, axis=1)
        W_inv = 1 / W
        W_inv2 = [1 / w if w != 0 else 0 for w in W]
        temp = np.multiply(W_inv, theta_hat)
        return temp

    def fit(self, train_X, train_y, val_X=None, val_y=None, test_X=None, test_y=None, epsilon=1e-6, learn_rate=None,
            max_iter=100, combine_method='average', leave_out=True, residual_net=False, LSE = True):
        train_n, p = train_X.shape
        self.train_yhat = np.zeros((train_n, self.n_split))
        if val_X is not None:
            val_n = val_X.shape[0]
            self.val_yhat = np.empty((val_n, self.n_split))
        if test_X is not None:
            test_n = test_X.shape[0]
            self.test_yhat = np.empty((test_n, self.n_split))

        # Split train data
        kfold = [list(range(train_n))[i::self.n_split] for i in range(self.n_split)]
        # kfold = GroupKFold(n_splits=self.n_split)
        # kfold.get_n_splits(train_X, train_y)
        if isinstance(train_X, pd.DataFrame):
            var_names = train_X.columns.tolist()
            train_X = train_X.values

        for k, idxs in enumerate(kfold):
            if leave_out:
                train_X_k = np.delete(train_X, idxs, axis=0)
                train_y_k = np.delete(train_y, idxs)
            else:
                train_X_k = train_X[idxs, :]
                train_y_k = train_y[idxs]
            if residual_net:
                submod_k = ResNet(subnet_struct=self.subnet_struct, fit_intercept=self.fit_intercept,
                                  ModelSelection=self.ModelSelection)
            else:
                submod_k = LOAN_base(subnet_struct=self.subnet_struct, fit_intercept=self.fit_intercept,
                                     ModelSelection=self.ModelSelection)
            submod_k.fit(train_X_k, train_y_k, val_X=val_X, val_y=val_y, test_X=test_X, test_y=test_y, epsilon=epsilon,
                         learn_rate=learn_rate, max_iter=max_iter, LSE = LSE)
            train_yhat_k, val_yhat_k, test_yhat_k = submod_k.get_yhat()
            self.alpha[k] = self.get_alpha(train_y_k, train_yhat_k)
            if combine_method == 'weighted':
                Hat_k = np.linalg.pinv(np.dot(train_X_k.T, train_X_k))
                self.Hat.append(Hat_k)
            self.submods[k] = submod_k

        # Get final weighted sum of each sub-models
        for k, mod in enumerate(self.submods):
            self.train_yhat[:, k] = mod.predict(train_X)
            self.train_loss.append(mean_squared_error(train_y, self.train_yhat[:, k]))
            if val_X is not None:
                self.val_yhat[:, k] = mod.predict(val_X)
                self.val_loss.append(mean_squared_error(val_y, self.val_yhat[:, k]))
            #    print("val_", k, "_MSE:", self.val_loss[-1])
            if test_X is not None:
                self.test_yhat[:, k] = mod.predict(test_X)
                self.test_loss.append(mean_squared_error(test_y, self.test_yhat[:, k]))
            #    print("test_", k, "_MSE:", mean_squared_error(test_y, self.test_yhat[:, k]))

        # Only keep top int(n_split * (1 - delete_ratio)) splitting results to constitute final result
        if self.delete_ratio > 0:
            n_keep = int(self.n_split * (1 - self.delete_ratio))
            if val_X is not None:
                idxes = np.argpartition(self.val_loss, n_keep)[:n_keep]
            else:
                idxes = np.argpartition(self.train_loss, n_keep)[:n_keep]

            self.keep_split = [True if i in idxes else False for i in range(self.n_split)]
            self.alpha = np.multiply(self.alpha, np.array(self.keep_split, dtype=int).reshape(self.alpha.shape))
        else:
            n_keep = self.n_split

        if combine_method == 'average':
            self.train_yhat_o = np.dot(self.train_yhat, self.keep_split) / n_keep
        if combine_method == 'boosting':
            self.train_yhat_o = np.dot(self.train_yhat, self.alpha) / sum(self.alpha)
        if combine_method == 'weighted':
            self.train_yhat_o = self.weighted_predict(train_X)
        self.train_mse_o = mean_squared_error(train_y, self.train_yhat_o)
        print("Train MSE:", self.train_mse_o)
        if val_X is not None:
            if combine_method == 'average':
                self.val_yhat_o = np.dot(self.val_yhat, self.keep_split) / n_keep
            if combine_method == 'boosting':
                self.val_yhat_o = np.dot(self.val_yhat, self.alpha) / sum(self.alpha)
            if combine_method == 'weighted':
                self.val_yhat_o = self.weighted_predict(val_X)
            self.val_mse_o = mean_squared_error(val_y, self.val_yhat_o)
            print("Val MSE:", self.val_mse_o)
        if test_X is not None:
            if combine_method == 'average':
                self.test_yhat_o = np.dot(self.test_yhat, self.keep_split) / n_keep
            if combine_method == 'boosting':
                self.test_yhat_o = np.dot(self.test_yhat, self.alpha) / sum(self.alpha)
            if combine_method == 'weighted':
                self.test_yhat_o = self.weighted_predict(test_X)
            self.test_mse_o = mean_squared_error(test_y, self.test_yhat_o)
            print("Test MSE:", self.test_mse_o)

    def get_mse(self):
        return self.train_mse_o, self.val_mse_o, self.test_mse_o


def other_regressor(mod, x_train, x_test, y_train, y_test):
    # Wrapper for different ML model training and testing
    mod.fit(x_train, y_train)
    train_yaht = mod.predict(x_train)
    test_yaht = mod.predict(x_test)
    return mean_squared_error(y_train, train_yaht), mean_squared_error(y_test, test_yaht)


def Split_utils(X, y, test_size, subnet_struct, n_split, ModelSelection='last', delete_ratio=0.25, max_iter=50,
                seed=0, compare_other_models = False):
    # Help to slite training and testing data set and compare with other models
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Block LSE
    start_time = datetime.datetime.now()
    BlockLSE = LOANSampleSplit(subnet_struct=subnet_struct, n_split=n_split, ModelSelection=ModelSelection,
                               delete_ratio=delete_ratio)
    BlockLSE.fit(x_train, y_train, val_X=x_train, val_y=y_train, test_X=x_test, test_y=y_test, learn_rate=1,
                 max_iter=max_iter, epsilon=1e-6, combine_method='average', leave_out=True, LSE=False)
    end_block = datetime.datetime.now()
    SplineMod = LOANSampleSplit(subnet_struct=subnet_struct, n_split=n_split, ModelSelection=ModelSelection,
                                delete_ratio=delete_ratio)
    SplineMod.fit(x_train, y_train, val_X=x_train, val_y=y_train, test_X=x_test, test_y=y_test, learn_rate=1,
                  max_iter=max_iter,
                  epsilon=1e-6, combine_method='average', leave_out=True, LSE = True)
    end_full = datetime.datetime.now()
    a = end_full - end_block
    b = end_block - start_time
    time_diff = a-b
    print("Full LSE is slower than block LSE: {}".format(time_diff))

    if compare_other_models:
        Ada_mod = AdaBoostRegressor()
        Ada_train_mse, Ada_test_mse = other_regressor(Ada_mod, x_train, x_test, y_train, y_test)

        Mlp_mod = MLPRegressor()
        Mlp_train_mse, Mlp_test_mse = other_regressor(Mlp_mod, x_train, x_test, y_train, y_test)

        GB_mod = GradientBoostingRegressor()
        GB_train_mse, GB_test_mse = other_regressor(GB_mod, x_train, x_test, y_train, y_test)

        RF_mod = RandomForestRegressor()
        RF_train_mse, RF_test_mse = other_regressor(RF_mod, x_train, x_test, y_train, y_test)

        xgb_mod = XGBRegressor()
        xgb_train_mse, xgb_test_mse = other_regressor(xgb_mod, x_train, x_test, y_train, y_test)

        return seed, SplineMod.train_mse_o, SplineMod.test_mse_o, BlockLSE.train_mse_o, BlockLSE.test_mse_o, \
               Ada_train_mse, Ada_test_mse, Mlp_train_mse, Mlp_test_mse, \
               GB_train_mse, GB_test_mse, RF_train_mse, RF_test_mse, xgb_train_mse, xgb_test_mse
    else:
        table = pd.DataFrame({'ID': [str(seed)+'_'+str(i) for i in range(len(SplineMod.train_loss))],
                              'LOAN train': SplineMod.train_loss,
                              'LOAN test': SplineMod.test_loss,
                              'Block train': BlockLSE.train_loss,
                              'Block test': BlockLSE.test_loss})
        return (seed, SplineMod.train_mse_o, SplineMod.test_mse_o, BlockLSE.train_mse_o, BlockLSE.test_mse_o, time_diff), table

#import data
#x = np.mat(pd.read_csv('calx.csv', header=None))
#Y = np.ravel(np.squeeze(pd.read_csv('caly.csv', header=None)))
#n, p = x.shape
#scaler = StandardScaler()
#scaler.fit(np.c_[x, Y])
#xy_scale = scaler.transform(np.c_[x, Y])
#X = xy_scale[:, :p]
#y = xy_scale[:, p]
XY = np.mat(pd.read_csv("airfoilX.csv"))
scaler = StandardScaler()
scaler.fit(XY)
dataset = scaler.transform(XY)
nrow,ncol = np.shape(dataset)
X = dataset[:, 0:ncol-1]
y = dataset[:,ncol-1]



# Set the testing data size ratio to all data
test_size = 0.2

# Number of splits for the training data to fit sub-LOAN models
n_split = 20
# When avaraging sub-LOAN models delete last bad results
delete_ratio = 0.25

# Maximum number of ephoch
max_iter = 100
# Choose sub-LOAN models epoch's weights or biases
ModelSelection = 'last'
# Hidden-layer network structure, the number stands for number of nodes
subnet_struct = [60]
#results = Parallel(n_jobs=-1, backend='loky', batch_size='auto')(
#            delayed(Split_utils)(X, y, test_size, subnet_struct, n_split, ModelSelection, delete_ratio, max_iter,seed)
#            for seed in seeds)

results = []
detail_results = pd.DataFrame()
for seed in seeds:
    summary, detail = Split_utils(X, y, test_size, subnet_struct, n_split, ModelSelection, delete_ratio, max_iter, seed)

    results.append(summary)
    detail_results = pd.concat([detail_results, detail], axis = 0,join='outer')

if len(results[0]) > 6:
    colname = ['seed','LOANN train', 'LOANN test', 'ResNet LOANN train', 'ResNet LOANN test',
           'Ada train', 'Ada test','MLP train', 'MLP test','GB train', 'GB test','RF train','RF test', 'xgb train', 'xgb test']
else:
    colname = ['seed', 'LOANN train', 'LOANN test', 'Block LSE LOANN train', 'Block LSE LOANN test', 'time_diff']

Result = pd.DataFrame(results, columns=colname)
fileName = "CHP_struct"+str(subnet_struct) + "_n_split"+ \
           str(n_split) +'_delete_ratio'+ str(delete_ratio) +"_maxiter" + str(max_iter) +".csv"
try:
    temp = pd.read_csv(fileName)
    Result = pd.concat([temp, Result], axis=0, join='outer')
except:
    pass
Result.to_csv(fileName, index= False)

fileName2 = "CHP_struct"+str(subnet_struct) + "_n_split"+ \
           str(n_split) +'_delete_ratio'+ str(delete_ratio) +"_maxiter" + str(max_iter) +"detail.csv"
try:
    temp = pd.read_csv(fileName2)
    detail_results = pd.concat([temp, detail_results], axis=0, join='outer')
except:
    pass

detail_results.to_csv(fileName2, index= False)