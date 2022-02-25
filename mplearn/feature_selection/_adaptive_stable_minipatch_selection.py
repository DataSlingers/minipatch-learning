import numpy as np
from common import BaseLearner
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
import numbers
from warnings import warn
from sklearn.utils import safe_mask
from minipatch_feature_selection.utils_feature_selection import manual_sort, \
    kde_based_pi_thr, visualize_selection_frequency_versus_iteration


def _adaptive_feature_sampling_exploitation_exploration(sampling_options,
                                                        rng,
                                                        gamma,
                                                        m,
                                                        M,
                                                        G,
                                                        keep_all_iters,
                                                        max_iters_to_keep,
                                                        Pi_hat_seq,
                                                        prev_partition,
                                                        k,
                                                        verbose):
    """Exploitation & Exploration adaptive feature sampling procedure.

    Parameters
    ----------
    sampling_options : dict
        Dictionary with parameter names (`str`) as keys and specific parameter
        settings as values. See the documentation of `AdaSTAMPS` for details.

    rng : RandomState instance
        Make sure the sampling uses the same RandomState instance for all iterations.

    gamma : float
        Controls the proportion of features sampled from the active set.

    m : int
        The number of features in a minipatch.

    M : int
        The total number of input features.

    G : int
        Number of iterations for each burn-in epoch.

    keep_all_iters : bool
        See the documentation of `AdaSTAMPS` for details.

    max_iters_to_keep : int
        See the documentation of `AdaSTAMPS` for details.

    Pi_hat_seq : array of shape (n_features, num_iterations)
        Contains the selection frequency of features over previous iterations.

    prev_partition : list of arrays
        Each element contains the indices of features that are sampled into a minipatch.

    k : int
        The current iteration number.

    verbose : {0, 1, 2}
        Controls the verbosity.

    Returns
    -------
    Fk : array of shape (m, )
        Contains the integer indices of the features from X that are in the current minipatch.

    rng : RandomState instance

    gamma : float
        The gamma value from the current iteration.

    feature_set_partition : list of arrays
        This is only returned during the burn-in stage.

    """

    # Specify options for the sampling procedure
    gamma_max = sampling_options["gamma_max"]  # usually default to 1

    # Initialize array to store index for all M features
    full_feature_idx = np.arange(M)

    # Subsample features using Adaptive Feature Sampling Exploitation and Exploration
    if k <= (sampling_options["E"] * G):  # during burn-in
        if int(k % G) == 1:
            # Reshuffle the full index set and partition at the beginning of an epoch (during burn-in only)
            feature_set_partition = np.array_split(rng.permutation(full_feature_idx), G)
            if verbose == 2:
                print("Reshuffling index set at k = %.i and partitioning into G=%.i groups is completed" % (k, G))
        else:
            feature_set_partition = prev_partition

        Fk_current = feature_set_partition[np.int((k % G))]
        if Fk_current.size == m:
            Fk = Fk_current
        elif Fk_current.size < m:
            if np.int((k % G)) < (len(feature_set_partition) - 1):
                # take Fk_append from the next block
                Fk_append = rng.choice(feature_set_partition[np.int((k % G) + 1)],
                                       size=np.int(m - Fk_current.size),
                                       replace=False)
            elif np.int((k % G)) == (len(feature_set_partition) - 1):
                Fk_append = rng.choice(feature_set_partition[0],
                                       size=np.int(m - Fk_current.size),
                                       replace=False)
            Fk = np.concatenate((Fk_current, Fk_append))
    else:  # after burn-in

        if keep_all_iters:
            above_thr_mask = Pi_hat_seq[:, (k - 1)] >= sampling_options["active_set_thr"]
        else:
            above_thr_mask = Pi_hat_seq[:, np.int(((k - 2) % max_iters_to_keep))] >= sampling_options["active_set_thr"]
        active_set_above_thr = full_feature_idx[above_thr_mask]
        active_set_size = np.sum(above_thr_mask)
        if verbose == 2:
            print("The active set above threshold |A| = %.i at iteration k = %.i" % (active_set_size, k))

        # Increase the proportion of active set sampled in the subsequent minipatch
        gamma = np.minimum((sampling_options["gamma_multiplier"] * gamma), gamma_max)
        num_var_from_active = np.minimum(m, np.int(gamma * active_set_size))
        if verbose == 2:
            print("%.i variables are sampled from active set at iteration k = %.i" % (num_var_from_active, k))
        Fk_from_active = rng.choice(active_set_above_thr, size=num_var_from_active, replace=False)

        if (m - num_var_from_active) > 0:
            candidate_set_below_thr = full_feature_idx[np.logical_not(above_thr_mask)]
            Fk_from_rest = rng.choice(candidate_set_below_thr, size=np.int(m - num_var_from_active), replace=False)
            Fk = np.sort(np.concatenate((Fk_from_active, Fk_from_rest)))
        else:
            Fk = Fk_from_active

    if k <= (sampling_options["E"] * G):
        return Fk, rng, gamma, feature_set_partition
    else:
        return Fk, rng, gamma


def _adaptive_feature_sampling_probabilistic(sampling_options,
                                             rng,
                                             m,
                                             M,
                                             G,
                                             keep_all_iters,
                                             max_iters_to_keep,
                                             Pi_hat_seq,
                                             prev_partition,
                                             k,
                                             verbose):
    """Probabilistic adaptive feature sampling procedure.

    Parameters
    ----------
    sampling_options : dict
        Dictionary with parameter names (`str`) as keys and specific parameter
        settings as values. See the documentation of `AdaSTAMPS` for details.

    rng : RandomState instance
        Make sure the sampling uses the same RandomState instance for all iterations.

    m : int
        The number of features in a minipatch.

    M : int
        The total number of input features.

    G : int
        Number of iterations for each burn-in epoch.

    keep_all_iters : bool
        See the documentation of `AdaSTAMPS` for details.

    max_iters_to_keep : int
        See the documentation of `AdaSTAMPS` for details.

    Pi_hat_seq : array of shape (n_features, num_iterations)
        Contains the selection frequency of features over previous iterations.

    prev_partition : list of arrays
        Each element contains the indices of features that are sampled into a minipatch.

    k : int
        The current iteration number.

    verbose : {0, 1, 2}
        Controls the verbosity.

    Returns
    -------
    Fk : array of shape (m, )
        Contains the integer indices of the features from X that are in the current minipatch.

    rng : RandomState instance

    feature_set_partition : list of arrays
        This is only returned during the burn-in stage.

    """

    # Initialize array to store index for all M features
    full_feature_idx = np.arange(M)

    if k <= (sampling_options["E"] * G):  # during burn-in
        if int(k % G) == 1:
            # Reshuffle the full index set and partition at the beginning of an epoch (during burn-in only)
            feature_set_partition = np.array_split(rng.permutation(full_feature_idx), G)
            if verbose == 2:
                print("Reshuffling index set at k = %.i and partitioning into G=%.i groups is completed" % (k, G))
        else:
            feature_set_partition = prev_partition

        Fk_current = feature_set_partition[np.int((k % G))]
        if Fk_current.size == m:
            Fk = Fk_current
        elif Fk_current.size < m:
            if np.int((k % G)) < (len(feature_set_partition) - 1):
                # take Fk_append from the next block
                Fk_append = rng.choice(feature_set_partition[np.int((k % G) + 1)],
                                       size=np.int(m - Fk_current.size), replace=False)
            elif np.int((k % G)) == (len(feature_set_partition) - 1):
                Fk_append = rng.choice(feature_set_partition[0],
                                       size=np.int(m - Fk_current.size), replace=False)
            Fk = np.concatenate((Fk_current, Fk_append))

    else:  # after burn-in

        if keep_all_iters:
            feature_prob = Pi_hat_seq[:, (k - 1)] / np.sum(Pi_hat_seq[:, (k - 1)])
        else:
            feature_prob = (Pi_hat_seq[:, np.int(((k - 2) % max_iters_to_keep))]
                            / np.sum(Pi_hat_seq[:, np.int(((k - 2) % max_iters_to_keep))]))

        Fk = np.sort(rng.choice(M, size=m, replace=False, p=feature_prob))

    if k <= (sampling_options["E"] * G):
        return Fk, rng, feature_set_partition
    else:
        return Fk, rng


class AdaSTAMPS(BaseLearner):
    """Feature selection with Adaptive Stable Minipatch Selection.

    This is a meta-algorithm that repeatedly fits base feature selectors to many
    random or adaptively chosen subsets of both observations and features (minipatches)
    and ensembles the selection events from all these base selectors. At the end
    of the algorithm, the final selection frequency is computed for each
    input feature as the number of times it is sampled and then selected by base selectors
    divided by the number of times it is sampled into minipatches. The selection
    frequency signifies the importance of features. The algorithm eventually selects
    the set of features whose selection frequency is above a certain threshold (can be
    either user-specific or determined automatically).

    Important note: `AdaSTAMPS` assumes that all necessary pre-processing steps
    prior to feature selection have already been carried out on the input data (X, y).
    For instance, data standardization (centering and/or scaling) needs to be
    performed on the raw data prior to calling `fit()` if such data pre-processing
    is deemed necessary by the users.

    Parameters
    ----------
    base_selector : ``Selector`` instance
        A feature selector with a ``fit`` method that provides
        binary selection indicators (1 for selected features and
        0 for unselected features). See **Notes** for more details.

    minipatch_m_ratio : float, default=0.05
        The fraction of features to draw from X to train each base selector.
        Specifically, `round(minipatch_m_ratio * X.shape[1])` features are drawn
        into each minipatch. Thus, `minipatch_m_ratio` should be in the
        interval (0.0, 1.0]. See **Notes** for more details.

    minipatch_n_ratio : float, default=0.5
        The fraction of observations to draw from X to train each base selector.
        Specifically, `round(minipatch_n_ratio * X.shape[0])` observations are drawn
        into each minipatch. Thus, `minipatch_n_ratio` should be in the
        interval (0.0, 1.0]. See **Notes** for more details.

    sampling_options : dict, default=None
        Dictionary with parameter names (`str`) as keys and specific parameter
        settings as values. This specifies the randomization scheme used to sample
        features into minipatches. Unless set to `None`, `sampling_options` is
        required to have a key named `'mode'`, whose value must be one of {'ee', 'prob', 'uniform'}.
        It is recommended to set `sampling_options` to `None` for starters, which uses the
        default Exploitation & Exploration adaptive feature sampling scheme with parameter
        values set to respective recommended values as described below. See **Notes** for more
        details.

        - If `'mode'` has value 'ee', it uses the Exploitation & Exploration scheme to
          adaptively sample features into minipatches. In this case, `sampling_options`
          is required to have the following parameters as additional keys:

          - `'E'` : int. The number of burn-in epochs during which every feature is sampled
            exactly `'E'` times to get an initial guess of feature importance before starting
            the adaptive sampling of features. A value of 10 generally works well for many problems.
            Note that a larger `'E'` generally requires increase the maximum number of
            iterations (`'max_k'`) in `stopping_criteria_options`.
          - `'active_set_thr'` : float. The selection frequency threshold above which
            a feature is put into the active set during the adaptive sampling stage.
            A value of 0.1 generally works well for many problems. Note that its value
            should be in the interval (0.0, 1.0). A larger value generally means fewer
            features in the active set.
          - `'gamma_min'` : float. The minimum proportion of features in the active set to
            sample into minipatches at the beginning of the adaptive sampling stage.
            It is recommened to fix its value to 0.5. Note that its value should be in
            the interval (0.0, 1.0), and should not exceed the value of `'gamma_max'`.
          - `'gamma_max'` : float. The maximum proportion of features in the active set to
            sample into minipatches as the adaptive sampling scheme proceeds. It is recommened
            to fix its value to 1.0. Note that its value should be in the interval (0.0, 1.0).
          - `'gamma_len'` : int. The number of iterations it takes for the adaptive feature
            sampler to go from `'gamma_min'` to `'gamma_max'`. This controls the trade-off
            between exploiting the active set and exploring the remaining input feature space.
            In general, a smaller value favors exploitation while a larger value favors exploration.
            A value in the range [50, 500] generally works well for many problems.

        - If `'mode'` has value 'prob', it uses the Probabilistic scheme to adaptively sample
          features into minipatches. In this case, `sampling_options` is required to have
          the following parameters as additional keys:

          - `'E'` : Int. This is the same as the `'E'` parameter in the case of `'mode'` being 'ee'.
            See the descriptions above for details.

        - If `'mode'` has value 'uniform', it samples features uniformly at random into minipatches.
          In this case, `sampling_options` does not need to have other key-value pairs.

    stopping_criteria_options : dict, default=None
        Dictionary with parameter names (`str`) as keys and specific parameter
        settings as values. This specifies parameter values for the data-driven stopping rule,
        which stops the meta-algorithm when the rank ordering of the top features in terms of
        selection frequency remain unchanged for the past `'num_last_iterations'`.
        Unless set to `None`, `stopping_criteria_options` is required to have
        4 keys: `'tau_u'`, `'tau_l'`, `'max_k'`, and `'num_last_iterations'`. It is recommended
        to set `stopping_criteria_options` to `None` for starters, which sets the parameter values to
        the respective default as described below.

        - `'tau_u'` : int. This specifies the maximum number of top features
          whose rank orderings should be considered when assessing the stopping rule. It is
          recommened to set its value to well exceed the expected number of truly informative features.
          The default value is set to 30. Note that its value should be much smaller than the total
          number of input features.
        - `'tau_l'` : int. This specifies the minimum number of top features
          whose rank orderings should be considered when assessing the stopping rule. It is
          recommened to set its value to well exceed the expected number of truly informative features.
          The default value is set to 15. Note that its value should be much smaller than `'tau_u'`.
        - `'num_last_iterations'` : int. The algorithm stops when the rank ordering of the
          top features in terms of selection frequency remain unchanged
          for the past `'num_last_iterations'`. It is recommended to fix its value to 100. Note that
          a unreasonably large value could render the stopping rule ineffective.
        - `'max_k'` : int. The maximum number of iterations to run the meta-algorithm if the data-driven
          stopping rule has not stopped it earlier. The default value is set to 5000. If `'mode'` of
          `sampling_options` is set to {'ee', 'prob'} and `'max_k'` is set to `None`, the algorithm will
          automatically compute `'max_k'` to be 5 times the number of burn-in iterations.

    random_state : int, default=0
        Controls both the randomness of sampling observations and sampling features into minipatches.

    keep_all_iters : bool, default=True
        Whether to store and output intermediate featrue selection frequency across all iterations. It could be
        useful to visualize selection frequency of all features versus iteration number for qualitatively
        discovering informative features. However, if the number of input feature is large (e.g. hundreds of thousands),
        then it is recommended to set this to `False` to avoid consuming too much memory. If set to `False`, it is
        required to set a value for `max_iters_to_keep`.

    max_iters_to_keep : int, default=None
        This value is ignored if `keep_all_iters` is `True`. Otherwise, this specifies the
        number of iterations (counting backwards from the last iteration) for which feature
        selection frequency should be stored and output. Note that `max_iters_to_keep` should be
        at least as large as the `'num_last_iterations'` of `stopping_criteria_options` (which is the
        default when `max_iters_to_keep` is set to `None`).

    verbose : {0, 1, 2}
        Controls the verbosity: the higher, more messages are displayed.


    Attributes
    ----------
    last_k_ : int
        The total number of iterations for which the meta-algorithm has run.

    Pi_hat_last_k_ : array of shape (n_features, )
        The final selection frequency for each of the input features. Each element is in the interval [0.0, 1.0].
        A larger value indicates that the corresponding feature is more informative, vice versa.

    full_Pi_hat_seq_ : array of shape (n_features, `last_k_`) or (n_features, `max_iters_to_keep`)
        If `keep_all_iters` is `True`, then this is an array of shape (n_features, `last_k_`) containing
        the selection frequency of all input features from first iteration to the last. If `keep_all_iters`
        is `False`, then this is an array of shape (n_features, `max_iters_to_keep`) containing the
        selection frequency of all input features for the last `max_iters_to_keep` iterations.

    full_Pi_hat_k_seq_ : array of shape (`last_k_`, ) or (`max_iters_to_keep`, )
        This contains the iteration numbers corresponding to the columns of `full_Pi_hat_seq_`.

    burn_in_length_ : int
        The total number of iterations spent in the burn-in stage. If `'mode'` of `sampling_options`
        is set to 'uniform' or if `keep_all_iters` is `False`, this is set to `None`. 

    Notes
    -----
    - More details about `base_selector`: The AdaSTAMPS meta-algorithm can be employed with
      a wide variety of feature selection techniques as the base selector on minipatches.
      This package current provides two highly efficient base selector classes -
      `minipatch_feature_selection.base_selector.ThresholdedOLS` for regression
      problems and `minipatch_feature_selection.base_selector.DecisionTreeSelector` for
      both regression and classification problems. However, user-supplied selector is
      also allowed as long as the selector class follows the same structure as the two
      base selectors mentioned above (i.e. has a ``fit`` method that accepts minipatch
      feature indices and provides binary selection indicators (1 for selected features and
      0 for unselected features).
    - More details about choice of minipatch size: Suppose the data X has N observations (rows)
      and M features (columns). Following the notations of [1], a minipatch is obtained by
      subsampling n observations and m features simultaneously without replacement from X
      using some form of randomization. The parameter `minipatch_m_ratio` represents :math:`m/M`
      and `minipatch_n_ratio` represents :math:`n/N`. As demonstrated in [1], the performance
      of the meta-algorithm is robust for a sensible range of n and m values. The general rule
      of thumb is to take m to well exceed the expected number of true informative features
      (e.g. 3-10 times the expected number of true informative features) and then pick n relative to
      m such that it well exceeds the sample complexity of the base selector used.
    - We refer the users to the original paper [1] for detailed algorithms for the various
      sampling procedures and the stopping rule.

    References
    ----------
    .. [1] Yao, T. and Allen, G. I., "Feature Selection for Huge Data via Minipatch Learning",
           arXiv:2010.08529.
    Examples
    --------
    The following example shows how to retrieve the 4 truly informative
    features in the sparse regression dataset.
    >>> from sklearn.datasets import make_sparse_uncorrelated
    >>> from minipatch_feature_selection.base_selector import ThresholdedOLS
    >>> from minipatch_feature_selection import AdaSTAMPS
    >>> X, y = make_sparse_uncorrelated(n_samples=100, n_features=10, random_state=0)
    >>> thresholded_ols = ThresholdedOLS(num_features_to_select=None, screening_thresh=None)
    >>> selector = AdaSTAMPS(base_selector=thresholded_ols,
    ...                      minipatch_m_ratio=0.5,
    ...                      minipatch_n_ratio=0.5,
    ...                      random_state=123,
    ...                      verbose=0)
    >>> fitted_selector = selector.fit(X, y)
    >>> fitted_selector.get_support(indices=True, pi_thr=0.5)
    array([0, 1, 2, 3])
    >>> X_new = fitted_selector.transform(X, pi_thr=0.5)
    >>> X_new.shape
    (100, 4)
    >>> fitted_selector.visualize_selection_frequency(max_features_to_plot=None)  # doctest: +SKIP
    """

    def __init__(self,
                 base_selector,
                 *,
                 minipatch_m_ratio=0.05,
                 minipatch_n_ratio=0.5,
                 sampling_options=None,
                 stopping_criteria_options=None,
                 random_state=0,
                 keep_all_iters=True,
                 max_iters_to_keep=None,
                 verbose=0):

        if sampling_options is None:
            sampling_options = {"mode": "ee",
                                "E": 10,
                                "active_set_thr": 0.1,
                                "gamma_min": 0.5,
                                "gamma_max": 1.,
                                "gamma_len": 50}

        if stopping_criteria_options is None:
            stopping_criteria_options = {"tau_u": 30,
                                         "tau_l": 15,
                                         "max_k": 5000,
                                         "num_last_iterations": 100}

        self.base_selector = base_selector
        self.minipatch_m_ratio = minipatch_m_ratio
        self.minipatch_n_ratio = minipatch_n_ratio
        self.sampling_options = sampling_options
        self.stopping_criteria_options = stopping_criteria_options
        self.random_state = random_state
        self.keep_all_iters = keep_all_iters
        self.max_iters_to_keep = max_iters_to_keep
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the AdaSTAMPS model to data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples. Note that data frame or sparse matrix format
            are not allowed. Also, the dtype of X has to be numeric (e.g. float, int).
            The algorithm expects that all appropriate preprocessing steps on X have been completed.

        y : ndarray of shape (n_samples,)
            The target values. Note that for classification problems (categorical y),
            the input y should contain integers denoting class labels instead of actual
            class names (`str`). In other words, the dtype of y has to be numeric (e.g. float, int).

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Allows NaN/Inf in the input if the underlying estimator does as well.
        """

        # Specify minipatch size
        m_ratio = self.minipatch_m_ratio  # m/M
        n_ratio = self.minipatch_n_ratio  # n/N

        # Dimensions of data matrix
        N, M = X.shape

        n = np.int(np.round(n_ratio * N))
        m = np.int(np.round(m_ratio * M))

        # Specify the maximum number of iterations to run if the data-driven stopping criterion is not satisfied
        max_k = self.stopping_criteria_options["max_k"]
        if max_k is None:
            if self.sampling_options["mode"] != "uniform":
                max_k = np.int((np.ceil(1. / m_ratio) * self.sampling_options["E"]) * 5)
            else:
                raise Exception('Setting max_k to None is not allowed when sampling mode is uniform.')

        # Initialize parameters for the feature sampling procedure
        if self.sampling_options["mode"] != "uniform":
            G = np.int(np.ceil(1. / m_ratio))
            iter_to_start_checking_criterion = int(self.sampling_options["E"] * G)
            burn_in_length = int(self.sampling_options["E"] * G)
            prev_partition = np.array_split(np.arange(M), G)
            if self.sampling_options["mode"] == "ee":
                gamma_seq_after_burn_in = np.geomspace(self.sampling_options["gamma_min"],
                                                       self.sampling_options["gamma_max"],
                                                       num=self.sampling_options["gamma_len"])
                self.sampling_options["gamma_multiplier"] = gamma_seq_after_burn_in[1] / gamma_seq_after_burn_in[0]
                # Initialize gamma to start from the minimum value
                gamma = self.sampling_options["gamma_min"]
        else:
            iter_to_start_checking_criterion = np.minimum(500, max_k)
            burn_in_length = None

        # Specify parameter values for the data-driven stopping criteria
        num_last_iterations = self.stopping_criteria_options["num_last_iterations"]
        num_top_var_to_check_lower_bound = np.minimum(self.stopping_criteria_options["tau_l"], int(0.5*M))
        stop_criterion_matrix = np.zeros((self.stopping_criteria_options["tau_u"], num_last_iterations), dtype=np.int64)
        stop_criterion_met = False
        stop_criterion_counter = -1 * iter_to_start_checking_criterion

        # Initialize array to store feature selection frequency over iterations
        if self.keep_all_iters:
            Pi_hat_seq = np.zeros((M, (max_k + 1)))
        else:
            if self.max_iters_to_keep is None:
                self.max_iters_to_keep = self.stopping_criteria_options["num_last_iterations"]
            Pi_hat_seq = np.zeros((M, self.max_iters_to_keep))
            Pi_hat_k_seq = np.zeros(self.max_iters_to_keep)

        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)

        # Initialize iteration counter
        k = 0

        # Get a clean copy of the base selector
        base_selector = clone(self.base_selector)

        # Keep a running tally of the number of times each variable is subsampled and selected
        total_times_subsampled_selected = np.zeros(M)
        total_times_subsampled = np.zeros(M)

        continue_alg = True

        while continue_alg:

            # Increment iteration counter
            k += 1

            # Initialize temporary storage for results (cleared after each iteration)
            var_subsampled_indicator = np.zeros(M)
            var_subsampled_selected_indicator = np.zeros(M)

            # Subsample n observations from {1,...,N} uniformly at random without replacement
            Ik = np.sort(rng.choice(N, size=n, replace=False))

            if self.sampling_options["mode"] == "uniform": # use uniform sampling of features
                Fk = np.sort(rng.choice(M, size=m, replace=False))
            elif self.sampling_options["mode"] == "ee": # use Exploitation and Exploration Adaptive Feature Sampling
                if k <= burn_in_length:
                    Fk, rng, gamma, prev_partition = _adaptive_feature_sampling_exploitation_exploration(self.sampling_options,
                                                                                                         rng,
                                                                                                         gamma,
                                                                                                         m,
                                                                                                         M,
                                                                                                         G,
                                                                                                         self.keep_all_iters,
                                                                                                         self.max_iters_to_keep,
                                                                                                         Pi_hat_seq,
                                                                                                         prev_partition,
                                                                                                         k,
                                                                                                         self.verbose)
                else:
                    Fk, rng, gamma = _adaptive_feature_sampling_exploitation_exploration(self.sampling_options,
                                                                                         rng,
                                                                                         gamma,
                                                                                         m,
                                                                                         M,
                                                                                         G,
                                                                                         self.keep_all_iters,
                                                                                         self.max_iters_to_keep,
                                                                                         Pi_hat_seq,
                                                                                         prev_partition,
                                                                                         k,
                                                                                         self.verbose)

            elif self.sampling_options["mode"] == "prob": # use Probabilistic Adaptive Feature Sampling
                if k <= burn_in_length:
                    Fk, rng, prev_partition = _adaptive_feature_sampling_probabilistic(self.sampling_options,
                                                                                       rng,
                                                                                       m,
                                                                                       M,
                                                                                       G,
                                                                                       self.keep_all_iters,
                                                                                       self.max_iters_to_keep,
                                                                                       Pi_hat_seq,
                                                                                       prev_partition,
                                                                                       k,
                                                                                       self.verbose)
                else:
                    Fk, rng = _adaptive_feature_sampling_probabilistic(self.sampling_options,
                                                                       rng,
                                                                       m,
                                                                       M,
                                                                       G,
                                                                       self.keep_all_iters,
                                                                       self.max_iters_to_keep,
                                                                       Pi_hat_seq,
                                                                       prev_partition,
                                                                       k,
                                                                       self.verbose)
            else:
                raise Exception('Other sampling procedures not implemented.')

            stop_criterion_counter += 1
            var_subsampled_indicator[Fk] = 1

            base_selector_on_minipatch = clone(base_selector)
            selection_on_minipatch = base_selector_on_minipatch.fit(X[np.ix_(list(Ik), list(Fk))], y[Ik], Fk)

            # Set the j-th entry to 1 if \beta_j^{(k)} ~= 0 and j \in Fk
            var_subsampled_selected_indicator[selection_on_minipatch.Fk_] = selection_on_minipatch.selection_indicator_

            # Keep a running tally of each feature's stability score up until the kth iteration (inclusive)
            total_times_subsampled_selected = total_times_subsampled_selected + var_subsampled_selected_indicator
            total_times_subsampled = total_times_subsampled + var_subsampled_indicator

            if self.keep_all_iters:
                Pi_hat_seq[:, k] = total_times_subsampled_selected / np.maximum(1, total_times_subsampled)
                if stop_criterion_counter >= 1:
                    # estimate size of possible stable set whose selection frequency are above 0.5
                    num_var_above_tau_thr = np.sum(Pi_hat_seq[:, k] >= 0.5)
                    num_top_var_to_check = np.minimum(self.stopping_criteria_options["tau_u"],
                                                      np.maximum(num_var_above_tau_thr,
                                                                 num_top_var_to_check_lower_bound))
            else:
                Pi_hat_seq[:, np.int(((k - 1) % self.max_iters_to_keep))] = (total_times_subsampled_selected
                                                                             / np.maximum(1, total_times_subsampled))
                Pi_hat_k_seq[np.int(((k - 1) % self.max_iters_to_keep))] = k
                if stop_criterion_counter >= 1:
                    num_var_above_tau_thr = np.sum(Pi_hat_seq[:, np.int(((k - 1) % self.max_iters_to_keep))] >= 0.5)
                    num_top_var_to_check = np.minimum(self.stopping_criteria_options["tau_u"],
                                                      np.maximum(num_var_above_tau_thr,
                                                                 num_top_var_to_check_lower_bound))

            # Compute the stopping criterion at the kth iteration
            if k >= np.maximum(iter_to_start_checking_criterion + 1, 10):
                if self.keep_all_iters:
                    top_Pi_hat_descend_idx = manual_sort(Pi_hat_seq[:, k],
                                                         num_top_var_to_check,
                                                         self.stopping_criteria_options["tau_u"])
                    stop_criterion_matrix[:, np.int(((stop_criterion_counter - 1) % num_last_iterations))] = top_Pi_hat_descend_idx
                else:
                    top_Pi_hat_descend_idx = manual_sort(Pi_hat_seq[:, np.int(((k - 1) % self.max_iters_to_keep))],
                                                         num_top_var_to_check,
                                                         self.stopping_criteria_options["tau_u"])
                    stop_criterion_matrix[:, np.int(((stop_criterion_counter - 1) % num_last_iterations))] = top_Pi_hat_descend_idx

            # check the stopping criterion after fully filled out the stop_criterion_matrix for the first time
            if k >= np.maximum(iter_to_start_checking_criterion + num_last_iterations + 1, 10):
                unique_top_var_sets = np.unique(stop_criterion_matrix, axis=1)
                if unique_top_var_sets.shape[1] == 1:
                    stop_criterion_met = True
                else:
                    if self.verbose == 2:
                        print("=== Number of unique set of top var in the last %.i iterations is %.i"
                              % (num_last_iterations, unique_top_var_sets.shape[1]))

            if (k + 1) > max_k:
                continue_alg = False
                print("==== Maximum number of iteration reached. ======")
                last_k = k
            elif k >= np.maximum(iter_to_start_checking_criterion + num_last_iterations + 1, 10) and stop_criterion_met:
                continue_alg = False
                print("=== Stopping criterion reached at k = " + str(k))
                last_k = k
            else:
                if self.verbose == 2:
                    print("========= Currently finished iteration k = " + str(k) + "=========")

        if self.keep_all_iters:
            Pi_hat_last_k = Pi_hat_seq[:, last_k]
        else:
            Pi_hat_last_k = Pi_hat_seq[:, np.int(((last_k - 1) % self.max_iters_to_keep))]

        if self.keep_all_iters:
            output_Pi_hat_seq = Pi_hat_seq[:, 1:(last_k + 1)]
            output_Pi_hat_k_seq = np.arange(1, (last_k + 1))
            # clear memory
            Pi_hat_seq = None
        else:
            if last_k > self.max_iters_to_keep:
                output_Pi_hat_seq = np.concatenate((Pi_hat_seq[:, np.int((last_k % self.max_iters_to_keep)):],
                                                    Pi_hat_seq[:, :np.int((last_k % self.max_iters_to_keep))]), axis=1)
                # clear memory
                Pi_hat_seq = None
                output_Pi_hat_k_seq = np.concatenate((Pi_hat_k_seq[np.int((last_k % self.max_iters_to_keep)):],
                                                      Pi_hat_k_seq[:np.int((last_k % self.max_iters_to_keep))]))
            else:

                output_Pi_hat_seq = Pi_hat_seq[:, :last_k]
                output_Pi_hat_k_seq = Pi_hat_k_seq[:last_k]
                # clear memory
                Pi_hat_seq = None

        if not self.keep_all_iters:
            burn_in_length = None
        self.last_k_ = last_k
        self.Pi_hat_last_k_ = Pi_hat_last_k
        self.full_Pi_hat_seq_ = output_Pi_hat_seq
        self.full_Pi_hat_k_seq_ = output_Pi_hat_k_seq
        self.burn_in_length_ = burn_in_length

        return self

    def get_support(self, indices=False, pi_thr=None):
        """
        Get a mask, or integer index, of the features selected by the meta-algorithm.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        pi_thr : float, default=None
            The selection frequency threshold above which a feature is considered selected.
            A larger threshold indicates a more stringent criterion.
            By default (`None`), a data-driven procedure is run to choose this threshold
            automatically. This is generally recommended, however, this procedure might
            take a long time if [# input features] is large (e.g. hundreds of thousands).
            For many problems, setting this threshold to 0.5 is a reasonable choice.
            Note that this threshold must be within (0.0, 1.0).

        Returns
        -------
        support : array
            If `indices` is `False`, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected by the algorithm. If `indices` is
            `True`, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        check_is_fitted(self)
        error_msg = (
            "pi_thr must be either None"
            "or a float in (0.0, 1.0)."
        )
        if pi_thr is None:
            pi_thr = kde_based_pi_thr(self.Pi_hat_last_k_)
        elif pi_thr < 0:
            raise ValueError(error_msg)
        elif pi_thr > 1.0:
            raise ValueError(error_msg)

        support_ = np.ones(self.Pi_hat_last_k_.size, dtype=bool)
        support_[self.Pi_hat_last_k_ < pi_thr] = False

        return support_ if not indices else np.where(support_)[0]

    def visualize_selection_frequency(self, max_features_to_plot=None):
        """ Visualize the selection frequency of features

        It is generally useful to visualize the selection frequency
        of the input features versus number of iterations for better
        insights into the estimated importance of the features.

        Parameters
        ----------
        max_features_to_plot : int, default=None
            Controls the maximum number of features whose selection frequency
            over iterations are visualized. By default ('None`),
            all input features are shown. However, such visualization
            might consume too much memory if the number of input feature
            is too large (e.g. 5000). In such cases, consider setting
            `max_features_to_plot` to be much smaller than n_features,
            which will only plot a small fraction of features whose
            selection frequency is below 0.3 to save on memory consumption.

        Returns
        -------
        None

        """
        check_is_fitted(self)
        error_msg = (
            "max_features_to_plot must be either None"
            "or a positive integer in (0, M]."
        )
        if max_features_to_plot is None:
            max_features_to_plot = self.Pi_hat_last_k_.size
        elif max_features_to_plot < 0:
            raise ValueError(error_msg)
        elif not isinstance(max_features_to_plot, numbers.Integral):
            raise ValueError(error_msg)

        visualize_selection_frequency_versus_iteration(self.full_Pi_hat_seq_,
                                                       self.full_Pi_hat_k_seq_,
                                                       self.burn_in_length_,
                                                       max_features_to_plot)

    def transform(self, X, pi_thr=None):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        pi_thr : float, default=None
            The selection frequency threshold above which a feature is considered selected.
            See the documentations of `get_support` for details.

        Returns
        -------
        X_r : array of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        check_is_fitted(self)
        # TODO: decide if need to validate X

        mask = self.get_support(pi_thr=pi_thr)
        if not mask.any():
            warn(
                "No features were selected: either the data is"
                " too noisy or the selection threshold is too strict.",
                UserWarning,
            )
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return X[:, safe_mask(X, mask)]



'''
import numpy as np
from minipatch_feature_selection.base_selector import ThresholdedOLS
from common import simple_clone

estimator = ThresholdedOLS(num_features_to_select=2)
a = simple_clone(estimator)

b = simple_clone(a)
b.fit(np.random.normal(0,1,50).reshape(10,5), np.random.normal(0,1,10), np.arange(2))

c = simple_clone(a)
c.fit(np.random.normal(0,1,50).reshape(10,5), np.random.normal(0,1,10), np.arange(2))




'''



