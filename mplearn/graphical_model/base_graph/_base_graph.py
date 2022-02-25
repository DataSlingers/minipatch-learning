import sys

import numpy as np
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import sklearn.utils._testing
sys.modules['sklearn.utils.testing'] = sklearn.utils._testing
from sklearn.utils.extmath import fast_logdet
from inverse_covariance import QuicGraphicalLasso

from ...common import BaseLearner


def _hard_thresh_matrix(mat, tau_k):
    """ Hard-threshold a matrix elementwise at a threshold value.

    Parameters
    ----------
    mat : ndarray of shape (m, m)
        The initial graph estimate from a minipatch.

    tau_k : float
        Every element of `mat` whose absolute value is less than
        or equal to `tau_k` will be set to zero.

    Returns
    -------

    """

    mat_copy = mat.copy()

    mat_copy[np.absolute(mat) <= tau_k] = 0

    return mat_copy


def _compute_ebic(precision, covariance, n, m, gamma):
    """ Compute the EBIC score for Gaussian graphical model.

    Parameters
    ----------
    precision : ndarray of shape (m, m)
        The estimated precision matrix on the minipatch.

    covariance : ndarray of shape (m, m)
        The sample covariance matrix on the minipatch.

    n : int
        The number of observations in the minipatch.

    m : int
        The number of nodes in the minipatch.

    gamma : float
        The gamma parameter in the extended BIC criterion.

    Returns
    -------

    """

    ln = (n / 2.) * (fast_logdet(precision) - np.sum(covariance * precision))

    if np.isinf(ln) or np.isnan(ln):
        return 1e10

    E_card = (np.sum(np.abs(precision.flat) > np.finfo(precision.dtype).eps) - m) / 2.0

    return (-2 * ln) + (E_card * np.log(n)) + (4 * E_card * gamma * np.log(m))


class ThresholdedGraphicalLasso(BaseLearner):
    """Gaussian graphical model selection with the Thresholded Graphical Lasso estimator.

    This class is designed to be used as a base graph selector on
    the minipatches with the `minipatch_graphical_model.MPGraph` class.
    At a high level, this estimator first gets an initial graph estimate
    using the Graphical Lasso at a small amount a regularization. After that,
    it hard-threshold the initial graph estimate at a sequence of threshold values
    and then chooses the thresholded graph with the best EBIC score as the final
    graph estimate. See the original paper [1] for more details.

    Parameters
    ----------
    lambda0_scale : float, default=1
        Controls the amount of regularization for Graphical Lasso to
        get the initial graphical model estimate. Specifically, the
        Graphical Lasso is fit at regularization
        `lambda0_scale * sqrt(log(n_features)/n_samples)`.
        Hence a larger value means a sparser initial graph estimate.
        Note that `lambda0_scale` needs to be larger than 0.

    threshold_seq : ndarray, default=None
        The sequence of threshold values at which to hard-threshold
        the initial graph estimate from the Graphical Lasso estimator.
        The elements of this array should be in the interval (0.0, 1].
        If set to `None`, the program will automatically set
        `threshold_seq=numpy.linspace(0.1, 0.5, 9)`.

    ebic_gamma : float, default=0.5
        The gamma parameter in the extended BIC criterion [2]. A larger
        value encourages sparser graph estimates. Note that `ebic_gamma`
        needs to be larger than or equal to 0.

    Attributes
    ----------
    Theta_tilde_ : ndarray of shape (n_features, n_features)
        The final precision matrix estimate corresponding to the thresholded graph
        with the best EBIC score.

    References
    ----------
    .. [1] Yao, T. and Wang, M. and Allen, G. I., "Gaussian Graphical Model Selection for Huge Data
           via Minipatch Learning", arXiv:2110.12067.
    .. [2] Foygel, R. and Drton, M., "Extended Bayesian Information Criteria for Gaussian
           Graphical Models", Neural Information Processing Systems 2010.

    """

    def __init__(self,
                 *,
                 lambda0_scale=1,
                 threshold_seq=None,
                 ebic_gamma=0.5):

        if threshold_seq is None:
            threshold_seq = np.linspace(0.1, 0.5, 9)

        self.lambda0_scale = lambda0_scale
        self.threshold_seq = threshold_seq
        self.ebic_gamma = ebic_gamma

    def fit(self, X, y=None):
        """Fit the Thresholded Graphical Lasso model to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data from which to infer the graphical model structure.
        y : Ignored.
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        assert (self.ebic_gamma >= 0), "ebic_gamma needs to be larger than or equal to 0."
        assert (self.lambda0_scale > 0), "lambda0_scale needs to be a float number larger than 0."

        n, m = X.shape

        lambda0 = self.lambda0_scale * np.sqrt(np.log(m) / n)
        # get an initial graph estimate using the Graphical Lasso estimator
        glasso = QuicGraphicalLasso(lam=lambda0,
                                    mode='default',
                                    verbose=0,
                                    init_method='cov',
                                    auto_scale=False,
                                    tol=1e-3,
                                    max_iter=1000).fit(X)

        Theta_hat_glasso = glasso.precision_.astype(np.float32)

        Sigma_hat = glasso.sample_covariance_.astype(np.float32)

        del glasso
        del X

        Theta_hat_max = 1

        ebic_seq = np.zeros(self.threshold_seq.size)
        # hard-threshold the initial graph estimate at different threshold and compute corresponding EBIC score
        for i in range(self.threshold_seq.size):

            ebic_seq[i] = _compute_ebic(_hard_thresh_matrix(Theta_hat_glasso, self.threshold_seq[i] * Theta_hat_max),
                                        Sigma_hat,
                                        n,
                                        m,
                                        self.ebic_gamma)

        min_indices = np.where(np.absolute(ebic_seq - ebic_seq.min()) < 1e-10)
        # choose the thresholded graph with the best EBIC score as the final graph estimate
        self.Theta_tilde_ = _hard_thresh_matrix(Theta_hat_glasso, self.threshold_seq[np.min(min_indices)] * Theta_hat_max)

        return self






