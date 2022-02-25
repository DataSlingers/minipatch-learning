from common import BaseLearner
from sklearn.utils import as_float_array, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.preprocessing import scale
from scipy.sparse import triu, coo_matrix
from itertools import combinations
import joblib
from joblib import Parallel, delayed
import numpy as np
from ray.util.joblib import register_ray
import ray


def _extract_indices(Theta_tilde_k, Fk, data_type):
    """ Extract node pair indices where there are estimated edges.

    Parameters
    ----------
    Theta_tilde_k : ndarray of shape (m ,m)
        The final graph estimate on a minipatch.

    Fk : ndarray of shape (m, )
        The integer index of nodes that are sampled into the current minipatch.

    data_type : ndarray of shape (1, )
        The `data_type.dtype` determines the data type of the
        returned results.

    Returns
    -------

    """

    # find the (i_prime, j_prime) that has an edge
    selected_ind_prime = triu(Theta_tilde_k, k=1)

    # find the (i, j) that is sampled and selected
    selected_ind_row = Fk[selected_ind_prime.row].astype(data_type.dtype)
    selected_ind_col = Fk[selected_ind_prime.col].astype(data_type.dtype)
    selected_ind = np.concatenate((selected_ind_row.reshape(1, -1), selected_ind_col.reshape(1, -1)), axis=0)

    # get the indices that are sampled
    sampled_ind = np.ones((2, Fk.size), dtype=data_type.dtype)
    sampled_ind[1, :] = Fk.astype(data_type.dtype)

    combined_ind = np.concatenate((sampled_ind, selected_ind), axis=1)

    return combined_ind


def _graph_selection_single_fit(base_graph, X_Ik_Fk, Fk, data_type):
    """ Fit the base graph selector on one minipatch.

    Parameters
    ----------
    base_graph : ``estimator`` instance
        An instance of the base graph selector.

    X_Ik_Fk : ndarray of shape (n, m)
        The sampled minipatch with n observations and m nodes.

    Fk : ndarray of shape (m, )
        The integer index of nodes that are sampled into the current minipatch.

    data_type : ndarray of shape (1, )
        The `data_type.dtype` determines the data type of the
        returned results.

    Returns
    -------

    """

    fitted_base_graph = base_graph.fit(X_Ik_Fk)

    S_D_vec_k = _extract_indices(fitted_base_graph.Theta_tilde_, Fk, data_type)

    return S_D_vec_k


def _minipatch_generator(N, M, max_k, n, m, random_state):
    """ Generate random minipatches with n observations and m nodes.

    Parameters
    ----------
    N : int
        The total number of observations in the full data.

    M : int
        The total number of features/nodes in the full data.

    max_k : int
        The total number of minipatches to generate.

    n : int
        The number of observations to sample into a minipatch.

    m : int
        The number of nodes to sample into a minipatch.

    random_state : int
        Controls both the randomness of sampling observations and sampling features into minipatches.

    Returns
    -------

    """
    r = np.random.RandomState(random_state)

    for _ in range(max_k):

        minipatch_idx_Ik = np.sort(r.choice(N, size=n, replace=False))  # uniform sampling of subset of observations

        minipatch_idx_Fk = np.sort(r.choice(M, size=m, replace=False))  # uniform sampling of subset of nodes

        minipatch_idx_Ik_Fk = np.append(minipatch_idx_Ik, minipatch_idx_Fk)

        yield minipatch_idx_Ik_Fk


def _post_processing_S_ind(S_D_vec, m):
    """
    Extract the node pair indices with an estimated edge from the result on one minipatch.
    """

    return S_D_vec[:, m:]


def _post_processing_D_ind(S_D_vec, m):
    """
    Generate indices of all the node pairs sampled into one minipatch.
    """

    return np.array(list(combinations(list(S_D_vec[1, :m]), 2))).transpose()


def _build_graph(mask, M):
    """
    Generate the sparse matrix representing the adjacency matrix of the final estimated graph structure.
    """

    support_ind = np.array(list(combinations(list(np.arange(M)), 2)))[mask, :]

    support_ = coo_matrix((np.ones(support_ind.shape[0], dtype=np.uint8), (support_ind[:, 0], support_ind[:, 1])),
                          shape=(M, M))

    support_ += support_.transpose()

    return support_


class MPGraph(BaseLearner):
    """Gaussian graphical model selection with Minipatch Graph.

    Gaussian graphical model selection refers to the problem of inferring the
    edge set (or sparsity patterns of the precision matrix) from obersved data.
    This is a meta-algorithm that repeatedly fits base graph selectors to many
    random subsets of both observations and features (minipatches) in parallel
    and ensembles the selection events from all these base selectors. At the end
    of the algorithm, the final edge selection frequency is computed for each
    pair of nodes (i, j) as the number of times (i, j) are sampled together and
    the base graph selector puts an edge between them divided by the number of times
    (i, j) are sampled together into minipatches. The algorithm eventually selects
    the set of edges whose selection frequency is above a certain threshold.

    Important note: `MPGraph` assumes that all necessary pre-processing steps
    prior to Gaussian graphical model selection have already been carried out
    on the input data X. For instance, data standardization (centering and/or scaling)
    needs to be performed on the raw data prior to calling `fit()` if such
    data pre-processing is deemed necessary by the users.

    Parameters
    ----------
    base_graph : ``estimator`` instance
        A Gaussian graphical model selector with a ``fit`` method that provides
        an estimate of either the precision matrix or its sparsity pattern.
        See **Notes** for more details.

    minipatch_m_ratio : float, default=0.05
        The fraction of features/nodes to draw from X to train each base graph selector.
        Specifically, `round(minipatch_m_ratio * X.shape[1])` nodes are drawn
        into each minipatch. Thus, `minipatch_m_ratio` should be in the
        interval (0.0, 1.0]. See **Notes** for more details.

    minipatch_n_ratio : float, default=0.5
        The fraction of observations to draw from X to train each base graph selector.
        Specifically, `round(minipatch_n_ratio * X.shape[0])` observations are drawn
        into each minipatch. Thus, `minipatch_n_ratio` should be in the
        interval (0.0, 1.0]. See **Notes** for more details.

    max_k : int, default=None.
        The total number of minipatches to use for training the meta-algorithm.
        If set to `None`, the algorithm will automatically compute `max_k` to
        be `ceil(1 / minipatch_m_ratio) * 50`.

    assume_centered : bool, default=True.
        If False, each column of X is centered to have a mean of zero.

    ray_configure : dict, default=None.
        Dictionary with parameter names (`str`) as keys and specific parameter
        settings as values. This specifies the parameter values for the Ray framework,
        which is used to parallelize computation over minipatches.
        Unless set to `None`, `ray_configure` is required to have at least one of two
        keys: `'memory'` and `'object_store_memory'`. If set to `None`, Ray would
        automatically detect the available memory resources on the system. See the documentation
        of Ray for more details. See **Notes** for further discussion.

        - `'memory'` : int. This specifies the amount of reservable memory resource (in Gigabytes) to create.
        - `'object_store_memory'` : int. The amount of memory (in Gigabytes) to start the object store with.

    n_jobs : int, default=-1.
        The number of jobs to run in parallel. `-1` means using all processors.

    parallel_batch_size : int, default=200.
        The number of tasks to dispatch at once to each worker. It is recommended
        to leave this at the default value. See documentation of `joblib.Parallel`
        for more details. See **Notes** for further discussion.

    parallel_post_processing : bool, default=False.
        Whether to use parallelization for aggregating the selection events from all minipatches.
        Using parallelization for this task can reduce the overall runtime, but might consume
        more memory.

    low_memory_mode : bool, default=False.
        Whether to clear some non-essential intermediate variables to save memory.

    random_state : int, default=0.
        Controls both the randomness of sampling observations and sampling features into minipatches.

    verbose : int, default=0.
        Controls the verbosity: the higher, more messages are displayed.


    Attributes
    ----------
    N_ : int
        The number of observations in the input data.

    M_ : int
        The number of features/nodes in the input data.

    Pi_hat_ : ndarray of shape (M*(M-1)/2, )
        The final edge selection frequency between each pair of nodes (i, j), where :math:`0\leq i < j \leq M-1`.
        Specifically, `Pi_hat_[0]` represents the selection frequency of the edge between the node pair (0, 1),
        `Pi_hat_[1]` represents the selection frequency of the edge between the node pair (0, 2), so on and so forth.
        Each element is in the interval [0.0, 1.0]. A larger value indicates that the corresponding
        edge is more stable.

    S_val_vec_ : ndarray of shape (M*(M-1)/2, )
        The total number of times each node pair (i, j) is sampled together into minipatches and
        there is an estimated edge between them as determined by the base graph selectors.
        The indexing of `S_val_vec_` is the same as `Pi_hat_`.
        If `low_memory_mode=True`, this would be set to `None`.

    D_val_vec_ : ndarray of shape (M*(M-1)/2, )
        The total number of times each node pair (i, j) is sampled together into minipatches.
        The indexing of `D_val_vec_` is the same as `Pi_hat_`.
        If `low_memory_mode=True`, this would be set to `None`.

    Notes
    -----
    - More details about ``base_graph``: The MPGraph meta-algorithm can be employed with
      a wide variety of thresholded Gaussian graphical model selection techniques
      as the base selector on minipatches.
      This package current provides a highly effective base selector classes -
      `minipatch_graphical_model.base_graph.ThresholdedGraphicalLasso`.
      However, user-supplied selector is also allowed as long as the selector class follows the
      same structure as the base graph selector mentioned above (i.e. has a ``fit`` method that
      provides an estimate of either the precision matrix or its sparsity patterns).
    - More details about choice of minipatch size: Suppose the data X has N observations (rows)
      and M nodes (columns). Following the notations of [1], a minipatch is obtained by
      subsampling n observations and m features simultaneously without replacement from X
      using some form of randomization. The parameter `minipatch_m_ratio` represents :math:`m/M`
      and `minipatch_n_ratio` represents :math:`n/N`. As demonstrated in [1], the performance
      of the meta-algorithm is robust for a sensible range of n and m values. The general rule
      of thumb is to take m to be 5% to 10% of M and then pick n relative to
      m such that it well exceeds the sample complexity of the base graph selector used.
    - More details about the parallelization framework: this software uses the Ray framework
      for parallelization. Specifically, ``ray_configure``, ``n_jobs``, and ``parallel_batch_size``
      together controls the behavior of the parallel backend. It is generally okay to leave these
      at default values. However, by the nature of parallelization framework, the computational speed
      can vary across systems with different hardware resources. For the most optimal performance,
      it is sometimes helpful to change these parameter values accordingly based on the available computing
      resources of your specific computer system.
    - We refer the users to the original paper [1] for the detailed algorithm.

    References
    ----------
    .. [1] Yao, T. and Wang, M. and Allen, G. I., "Gaussian Graphical Model Selection for Huge Data
           via Minipatch Learning", arXiv:2110.12067.
    Examples
    --------
    The following example shows how to infer the structure of the
    Gaussian graphical model (sparsity pattern of the precision matrix) from observed data.
    >>> import numpy as np
    >>> from minipatch_graphical_model.base_graph import ThresholdedGraphicalLasso
    >>> from minipatch_graphical_model import MPGraph
    >>> N, M = 1000, 500
    >>> precision = 1.25*np.identity(M) + np.diag(0.5 * np.ones(M - 1), 1) + np.diag(0.5 * np.ones(M - 1), -1)
    >>> X = np.random.RandomState(0).multivariate_normal(np.zeros(M), np.linalg.inv(precision), size=N)
    >>> base_graph = ThresholdedGraphicalLasso()
    >>> mpgraph = MPGraph(base_graph=base_graph,
    ...                   minipatch_m_ratio=(50. / M),
    ...                   minipatch_n_ratio=(60. / N),
    ...                   max_k=1000,
    ...                   parallel_post_processing=True,
    ...                   verbose=0)
    >>> fitted_mpgraph = mpgraph.fit(X)
    >>> estimated_graph = fitted_mpgraph.get_support(support_type="sparse_matrix", pi_thr=0.5)  # doctest: +SKIP
    """

    def __init__(self,
                 base_graph,
                 *,
                 minipatch_m_ratio=0.05,
                 minipatch_n_ratio=0.5,
                 max_k=None,
                 assume_centered=True,
                 ray_configure=None,
                 n_jobs=-1,
                 parallel_batch_size=200,
                 parallel_post_processing=False,
                 low_memory_mode=False,
                 random_state=0,
                 verbose=0):

        self.base_graph = base_graph
        self.minipatch_m_ratio = minipatch_m_ratio
        self.minipatch_n_ratio = minipatch_n_ratio
        self.max_k = max_k
        self.assume_centered = assume_centered
        self.ray_configure = ray_configure
        self.n_jobs = n_jobs
        self.parallel_batch_size = parallel_batch_size
        self.parallel_post_processing = parallel_post_processing
        self.low_memory_mode = low_memory_mode
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the MPGraph model to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data from which to infer the graphical model structure. Note that data frame or sparse matrix format
            are not allowed. Also, the dtype of X has to be numeric (e.g. float, int). NaN/Inf are not
            allowed in the input. 
        y : Ignored.
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        GB = 1000 * 1e6

        if not ray.is_initialized():

            if self.ray_configure is None:

                ray.init(include_dashboard=False)
            else:
                if ("memory" in self.ray_configure) and ("object_store_memory" not in self.ray_configure):

                    ray.init(_memory=self.ray_configure["memory"] * GB, include_dashboard=False)
                elif ("memory" not in self.ray_configure) and ("object_store_memory" in self.ray_configure):

                    ray.init(object_store_memory=self.ray_configure["object_store_memory"] * GB,
                             include_dashboard=False)
                elif ("memory" in self.ray_configure) and ("object_store_memory" in self.ray_configure):

                    ray.init(_memory=self.ray_configure["memory"] * GB,
                             object_store_memory=self.ray_configure["object_store_memory"] * GB,
                             include_dashboard=False)
                else:
                    error_msg = (
                        "ray_configure must either be None"
                        "or a dictionary containing at least one of the key: 'memory' and 'object_store_memory."
                    )
                    raise ValueError(error_msg)

        register_ray()

        X = check_array(X, ensure_min_features=2, force_all_finite=False, estimator=self)
        # make sure the input data is a ndarray
        X = as_float_array(X, copy=False, force_all_finite=False)

        if not self.assume_centered:
            X = scale(X, axis=0, with_mean=True, with_std=False)

        # Dimensions of data matrix
        N, M = X.shape

        n = np.int(np.round(self.minipatch_n_ratio * N))  # number of observations in a minipatch
        m = np.int(np.round(self.minipatch_m_ratio * M))  # number of nodes in a minipatch

        if self.max_k is None:
            self.max_k = int(np.ceil(M / m) * 50)

        if M <= 65000:
            data_type = np.ones(1, dtype=np.uint16)
        else:
            data_type = np.ones(1, dtype=np.uint32)

        # call the generator function to sample minipatches
        random_minipatches = _minipatch_generator(N, M, self.max_k, n, m, self.random_state)

        base_graph = clone(self.base_graph)

        with joblib.parallel_backend("ray", n_jobs=self.n_jobs):
            # fit base graph selectors to all random minipatches in parallel
            S_D_vec_over_minipatches = Parallel(batch_size=self.parallel_batch_size, verbose=self.verbose)(
                delayed(_graph_selection_single_fit)(clone(base_graph),
                                                     X[np.ix_(list(minipatch_idx_Ik_Fk[:n]), list(minipatch_idx_Ik_Fk[n:]))],
                                                     minipatch_idx_Ik_Fk[n:].astype(int),
                                                     data_type) for minipatch_idx_Ik_Fk in random_minipatches)

        if self.verbose > 0:
            print("=== Training over %.i minipatches has finished. Starting post-processing ===" % self.max_k)
        # post-processing: aggregate results of selection events from all minipatches
        if not self.parallel_post_processing:
            # conduct post-processing in serial mode
            for k in range(len(S_D_vec_over_minipatches)):
                if k == 0:
                    # get the (i, j) that is sampled and selected in the kth minipatch (i is guaranteed to be smaller than j)
                    S_ind = S_D_vec_over_minipatches[k][:, m:]
                    # get the (i, j) that is sampled in the kth minipatch (i is guaranteed to be smaller than j)
                    D_ind = np.array(list(combinations(list(S_D_vec_over_minipatches[k][1, :m]), 2))).transpose()
                else:
                    S_ind = np.concatenate((S_ind, S_D_vec_over_minipatches[k][:, m:]), axis=1)
                    D_ind = np.concatenate((D_ind, np.array(list(combinations(list(S_D_vec_over_minipatches[k][1, :m]), 2))).transpose()), axis=1)

        else:
            # conduct post-processing in parallel mode
            with joblib.parallel_backend("ray", n_jobs=self.n_jobs):
                S_ind_list = Parallel(batch_size=self.parallel_batch_size, verbose=self.verbose)(
                    delayed(_post_processing_S_ind)(S_D_vec, m) for S_D_vec in S_D_vec_over_minipatches)
                D_ind_list = Parallel(batch_size=self.parallel_batch_size, verbose=self.verbose)(
                    delayed(_post_processing_D_ind)(S_D_vec, m) for S_D_vec in S_D_vec_over_minipatches)

            S_ind = np.concatenate(S_ind_list, axis=1)

            D_ind = np.concatenate(D_ind_list, axis=1)

        # compute the total number of times each node pair (i, j) is sampled together into minipatches and
        # there is an estimated edge between them as determined by the base graph selectors.
        self.S_val_vec_ = coo_matrix((np.ones(S_ind.shape[1], dtype=np.uint16), (S_ind[0, :], S_ind[1, :])),
                                     shape=(M, M)).todok()[np.triu_indices(M, k=1)].toarray().ravel()
        # compute the total number of times each node pair (i, j) is sampled together into minipatches.
        self.D_val_vec_ = coo_matrix((np.ones(D_ind.shape[1], dtype=np.uint16), (D_ind[0, :], D_ind[1, :])),
                                     shape=(M, M)).todok()[np.triu_indices(M, k=1)].toarray().ravel()

        if self.verbose > 0:
            print("=== Post-processing has finished ===")

        S_D_vec_over_minipatches = None
        S_ind = None
        D_ind = None
        if self.parallel_post_processing:
            S_ind_list = None
            D_ind_list = None
        # compute the final edge selection frequency between each pair of nodes (i, j), for 0 <= i < j <= M-1
        self.Pi_hat_ = (self.S_val_vec_ / np.maximum(1, self.D_val_vec_)).astype(np.float32)
        self.N_ = N
        self.M_ = M
        if self.low_memory_mode:
            self.S_val_vec_ = None
            self.D_val_vec_ = None

        ray.shutdown()

        return self

    def get_support(self, support_type='mask', pi_thr=0.5):
        """
        Get a mask, indicator, matrix, or node pair index, of the edges selected by the meta-algorithm.

        Parameters
        ----------
        support_type : {'mask', 'indicator', 'sparse_matrix', 'node_pair'}, default='mask'
            Specify the format in which the estimated graph structure is returned.

        pi_thr : float, default=0.5
            The selection frequency threshold above which an edge is considered selected.
            A larger threshold indicates a more stringent criterion.
            For many problems, setting this threshold to 0.5 is a reasonable choice.
            Note that this threshold must be within (0.0, 1.0).

        Returns
        -------
        support : ndarray or coo_matrix

            - If `support_type='mask'`, then a boolean array of shape (M*(M-1)/2, ) indicating
              presence of edges between each pair of nodes (i, j), where :math:`0\leq i < j \leq M-1`.
              Specifically, `support[0]=True` iff there is an edge between the node pair (0, 1),
              `support[1]=True` iff there is an edge between the node pair (0, 2), so on and so forth.
            - If `support_type='indicator'`, then a binary array of shape (M*(M-1)/2, ) indicating
              presence of edges between each pair of nodes (i, j), :math:`0\leq i < j\leq M-1`.
              Specifically, `support[0]=1` if there is an edge between the node pair (0, 1) and
              `support[0]=0` otherwise. The indexing is the same as 'mask'.
            - If `support_type='sparse_matrix'`, then a sparse matrix in COO format (coo_matrix)
              of shape (M, M) representing the adjacency matrix of the estimated graph structure.
              The nonzero entries of this sparse matrix indicate presence of edges. Note that one
              may convert the adjacency matrix to a dense array by using `support.toarray()`. However,
              converting to dense array can consume a large amount of memory for large M, so it
              is usually desirable to keep this in sparse matrix format.
            - If `support_type='node_pair'`, then a ndarray of shape (# selected edges, 2) representing
              node pairs (i, j) between which there is an estimated edge, :math:`0\leq i < j\leq M-1`.
              Specifically, each row is a node pair with an estimated edge.
        """

        check_is_fitted(self)
        error_msg = (
            "pi_thr must be a float in (0.0, 1.0)."
        )

        if pi_thr < 0:
            raise ValueError(error_msg)
        elif pi_thr > 1.0:
            raise ValueError(error_msg)

        support_ = (self.Pi_hat_ >= pi_thr)

        if support_type == 'mask':
            return support_
        elif support_type == 'indicator':
            return support_.astype(np.uint8)
        elif support_type == 'sparse_matrix':
            return _build_graph(support_, self.M_)
        elif support_type == 'node_pair':
            support_node_pair = np.array(list(combinations(list(np.arange(self.M_)), 2)))[support_, :]
            return support_node_pair
        else:
            raise ValueError("support_type must be one of {'mask', 'indicator', 'sparse_matrix', 'node_pair'}")




















































































