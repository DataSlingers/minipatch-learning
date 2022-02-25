import numbers

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ...common import BaseLearner


class ThresholdedOLS(BaseLearner):
    """Feature selection with the Thresholded OLS selector.

    This class is designed to be used as a base feature selector on
    the minipatches with the `minipatch_feature_selection.AdaSTAMPS` class.

    Parameters
    ----------
    num_features_to_select : int or float, default=None
        The number of features to select from the m features in a minipatch.

        - If `None`, it employs the Bonferroni procedure as
          described in [1] to automatically decide the number
          of features to select on the minipatch.
        - If positive integer, it is the absolute number of
          features to select on a minipatch.
        - If float in the interval (0.0, 1.0], it is the percentage
          of the m features in a minipatch to select.

    screening_thresh : float, default=None
        This is ignored if the minipatch has more observations n
        than features m. For high-dimensional minipatches (n<m),
        `screening_thresh` should be a float in the interval (0.0, 1.0),
        which will first applying an efficient screening rule to reduce
        the number of features in the minipatch to `round(screening_thresh * n)`.

    Attributes
    ----------
    selection_indicator_ : array of shape (m or `round(screening_thresh * n)`, )
        A binary selection indicator for the features in the minipatch
        (1 for selected features and 0 for unselected features). If low-dimensional
        minipatch (n>m), the shape is m. Otherwise, the shape is `round(screening_thresh * n)`.

    Fk_ : array of shape (m or `round(screening_thresh * n)`, )
        The corresponding feature indices of the features in `selection_indicator_`.
        Note that these indices correspond to these features' column indices
        in the full data X_full.

    References
    ----------
    .. [1] Giurcanu, M. . "Thresholding least-squares inference in
           high-dimensional regression models." Electron. J. Statist.
           10 (2) 2124 - 2156, 2016.

    """

    def __init__(self,
                 *,
                 num_features_to_select=None,
                 screening_thresh=None):

        self.num_features_to_select = num_features_to_select
        self.screening_thresh = screening_thresh

    def fit(self, X, y, Fk):
        """Fit the thresholded OLS base selector to a minipatch.

        Parameters
        ----------
        X : ndarray of shape (n, m)
            The data matrix corresponding to the minipatch (n observations and m features).

        y : ndarray of shape (n,)
            The target values corresponding to the minipatch.

        Fk : array of shape (m, )
            The feature indices of the features in the minipatch.
            Note that these indices correspond to these features' column indices
            in the full data X_full. For example, `X = X_full[:, F_k]`.

        Returns
        -------
        self : object
            Fitted estimator.

        """

        n, m = X.shape

        # If the minipatch has more features than observations, apply the screening rule to keep only (prop_to_keep * n) features
        if n < m:

            if self.screening_thresh is None:
                raise ValueError('Because n < m. Please specify screening_thresh to be a float in (0.0, 1.0).')

            # Compute componentwise least squared estimate
            y_n_m = np.tile(y, (m, 1)).transpose()
            beta_tilde = np.sum((y_n_m * X), axis=0) / np.sum((X ** 2), axis=0)

            beta_tilde_n_m = np.tile(beta_tilde, (n, 1))
            sigma_tilde_squared = (1. / (n * (n - 1))) * np.sum(np.square((y_n_m - (X * beta_tilde_n_m))), axis=0)

            gamma_tilde = np.absolute(beta_tilde) / np.sqrt(sigma_tilde_squared)

            sorted_gamma_tilde_descend = np.sort(gamma_tilde)[::-1]
            q = np.int(self.screening_thresh * n)
            gamma_tilde_thres = sorted_gamma_tilde_descend[q]
            Fk_to_keep_mask = (gamma_tilde > gamma_tilde_thres)
            X = X[:, Fk_to_keep_mask]
            n, m = X.shape
            Fk = Fk[Fk_to_keep_mask]

        lm = LinearRegression(fit_intercept=False).fit(X, y)
        beta_hat_ols = lm.coef_

        beta_hat_ols_thresholded = beta_hat_ols.copy()

        if self.num_features_to_select is None:

            t_alpha_quantile = 1. / (2 * np.log(n))
            t_scale = stats.t(df=(n - m)).ppf((1 - (t_alpha_quantile / m)))
            error_var_hat = (1. / (n - m)) * (np.linalg.norm(y - np.matmul(X, beta_hat_ols)) ** 2)
            try:
                Omega_inv = np.linalg.inv(((1. / n) * np.matmul(X.transpose(), X)))
            except:
                Omega_inv = np.linalg.pinv(((1. / n) * np.matmul(X.transpose(), X)))
            sigma_bar_jj = ((1. / np.sqrt(n)) * np.sqrt(error_var_hat)) * np.sqrt(np.absolute(np.diagonal(Omega_inv)))
            beta_hat_ols_thresholded[(np.absolute(beta_hat_ols) <= (sigma_bar_jj * t_scale))] = 0
        else:
            error_msg = (
                "num_features_to_select must be either None, a "
                "positive integer representing the absolute "
                "number of features to select on a minipatch or a float in (0.0, 1.0] "
                "representing a percentage of the m features in a minipatch to select."
            )
            if self.num_features_to_select < 0:
                raise ValueError(error_msg)
            elif isinstance(self.num_features_to_select, numbers.Integral):
                num_features_to_select = self.num_features_to_select
            elif self.num_features_to_select > 1.0:
                raise ValueError(error_msg)
            else:
                num_features_to_select = int(m * self.num_features_to_select)

            sorted_beta_hat_ols = np.sort(np.absolute(beta_hat_ols))[::-1]
            beta_hat_ols_thresholded[(np.absolute(beta_hat_ols) <= sorted_beta_hat_ols[num_features_to_select])] = 0

        self.selection_indicator_ = np.absolute(np.sign(beta_hat_ols_thresholded))
        self.Fk_ = Fk

        return self


class DecisionTreeSelector(BaseLearner):
    """Feature selection with the decision tree selector.

    This class is designed to be used as a base feature selector on
    the minipatches with the `minipatch_feature_selection.AdaSTAMPS` class.
    This is a wrapper built around the DecisionTreeClassifier and the
    DecisionTreeRegressor from the sklearn package.

    Parameters
    ----------
    mode : {'classifier', 'regressor'}
        Controls the type of the decision tree model to use.

    max_depth : int, default=5
        The maximum depth of the tree. If `None`, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    criterion : {'gini', 'entropy', 'squared_error', 'friedman_mse', 'absolute_error', 'poisson'}
        The criterion to measure the quality of a split. If `mode='classifier'`, this must be
        {'gini', 'entropy'}. If `mode='regressor'`, this must be
        {'squared_error', 'friedman_mse', 'absolute_error', 'poisson'}.

    num_features_to_select : int or float, default=0.1
        The number of features to select from the m features in a minipatch.

        - If positive integer, it is the absolute number of
          features to select on a minipatch.
        - If float in the interval (0.0, 1.0], it is the percentage
          of the m features in a minipatch to select.

    random_state : int, default=123
        Controls the randomness of the decision tree model.

    Attributes
    ----------
    selection_indicator_ : array of shape (m, )
        A binary selection indicator for the features in the minipatch
        (1 for selected features and 0 for unselected features).

    Fk_ : array of shape (m, )
        The corresponding feature indices of the features in `selection_indicator_`.
        Note that these indices correspond to these features' column indices
        in the full data X_full.

    """

    def __init__(self,
                 *,
                 mode='classifier',
                 max_depth=5,
                 criterion='gini',
                 num_features_to_select=0.1,
                 random_state=123):

        self.mode = mode
        self.max_depth = max_depth
        self.criterion = criterion
        self.num_features_to_select = num_features_to_select
        self.random_state = random_state

    def fit(self, X, y, Fk):
        """Fit the decision tree base selector to a minipatch.

        Parameters
        ----------
        X : ndarray of shape (n, m)
            The data matrix corresponding to the minipatch (n observations and m features).

        y : ndarray of shape (n,)
            The target values corresponding to the minipatch.

        Fk : array of shape (m, )
            The feature indices of the features in the minipatch.
            Note that these indices correspond to these features' column indices
            in the full data X_full. For example, `X = X_full[:, F_k]`.

        Returns
        -------
        self : object
            Fitted estimator.

        """

        n, m = X.shape

        if self.mode == 'classifier':
            if self.criterion not in ['gini', 'entropy']:
                raise ValueError("criterion must be {'gini', 'entropy'} for classification.")
            estimator = DecisionTreeClassifier(criterion=self.criterion,
                                               max_depth=self.max_depth,
                                               random_state=self.random_state).fit(X, y)
        elif self.mode == 'regressor':
            if self.criterion not in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']:
                raise ValueError("criterion must be {'squared_error', 'friedman_mse', 'absolute_error', 'poisson'} for regression.")
            estimator = DecisionTreeRegressor(criterion=self.criterion,
                                              max_depth=self.max_depth,
                                              random_state=self.random_state).fit(X, y)
        else:
            raise ValueError("mode must be either 'classifier' or 'regressor'.")
        feature_importance_scores = estimator.feature_importances_

        error_msg = (
            "num_features_to_select must be either a "
            "positive integer representing the absolute "
            "number of features to select on a minipatch or a float in (0.0, 1.0] "
            "representing a percentage of the m features in a minipatch to select."
        )
        if self.num_features_to_select < 0:
            raise ValueError(error_msg)
        elif isinstance(self.num_features_to_select, numbers.Integral):
            num_features_to_select = self.num_features_to_select
        elif self.num_features_to_select > 1.0:
            raise ValueError(error_msg)
        else:
            num_features_to_select = int(m * self.num_features_to_select)

        feature_importance_scores_sort_descend_idx = np.argsort(feature_importance_scores)[::-1]

        hat_nonzero_indicator = np.zeros(m)
        hat_nonzero_indicator[feature_importance_scores_sort_descend_idx[:num_features_to_select]] = 1

        self.selection_indicator_ = hat_nonzero_indicator
        self.Fk_ = Fk

        return self





