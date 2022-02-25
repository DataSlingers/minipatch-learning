import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def manual_sort(Pi_hat_k,
                num_top_var_to_check,
                num_top_var_to_check_initial_value):
    """Get the rank ordering of top features with highest selection frequency.

    Parameters
    ----------
    Pi_hat_k : array of shape (n_features, )
        The selection frequency of all input features at the k-th iteration.

    num_top_var_to_check : int
        The number of top features in terms of selection frequency whose rank ordering
        will be computed.

    num_top_var_to_check_initial_value : int
        This specifies the maximum number of top features
        whose rank orderings should be considered when assessing the stopping rule.

    Returns
    -------
    index : array of shape (`minimum(num_top_var_to_check, num_top_var_to_check_initial_value), )
        The indices of the top features at the k-th iteration, ordered in descending selection frequency.

    """

    perfect_stability_mask = Pi_hat_k == 1.
    if np.sum(perfect_stability_mask) == 0:
        Pi_hat_descend_idx_tmp = np.argsort(Pi_hat_k)[::-1]
        top_Pi_hat_descend_idx = Pi_hat_descend_idx_tmp[:num_top_var_to_check]
    elif np.sum(perfect_stability_mask) > 0 and np.sum(perfect_stability_mask) < num_top_var_to_check:
        var_idx_perfect_stability_sorted = np.sort(np.argwhere(perfect_stability_mask).flatten())
        Pi_hat_descend_idx_tmp = np.argsort(Pi_hat_k)[::-1]
        Pi_hat_descend_idx_tmp_of_interest = Pi_hat_descend_idx_tmp[:num_top_var_to_check]
        top_Pi_hat_descend_idx = np.concatenate((var_idx_perfect_stability_sorted, Pi_hat_descend_idx_tmp_of_interest[np.logical_not(np.isin(Pi_hat_descend_idx_tmp_of_interest, var_idx_perfect_stability_sorted))]))
    elif np.sum(perfect_stability_mask) >= num_top_var_to_check:
        var_idx_perfect_stability_sorted = np.sort(np.argwhere(perfect_stability_mask).flatten())

        top_Pi_hat_descend_idx = np.random.choice(var_idx_perfect_stability_sorted,
                                                  size=num_top_var_to_check,
                                                  replace=False)


    if num_top_var_to_check < num_top_var_to_check_initial_value:

        M = Pi_hat_k.size
        top_Pi_hat_descend_idx_final = M * np.ones(num_top_var_to_check_initial_value, dtype=np.int64)
        top_Pi_hat_descend_idx_final[:num_top_var_to_check] = top_Pi_hat_descend_idx
    else:
        top_Pi_hat_descend_idx_final = top_Pi_hat_descend_idx

    return top_Pi_hat_descend_idx_final


def _get_mode_points(data, type_of_inflexion=None, max_points=None):

    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange == 1)[0]

    if idx.size == 0:
        return idx

    if type_of_inflexion == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]

    elif type_of_inflexion == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif type_of_inflexion is not None:
        idx = idx[::2]


    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data) - 1) in idx:
        idx = np.delete(idx, len(data) - 1)
    idx = idx[np.argsort(data[idx])]

    if max_points is not None:
        idx = idx[:max_points]
        if len(idx) < max_points:
            return (np.arange(max_points) + 1) * (len(data) // (max_points + 1))

    return idx


def kde_based_pi_thr(Pi_hat_last_k):
    """Procedure to determine selection frequency threshold.

    This procedure automatically determines the pi_thr
    in a data-driven manner. The detailed algorithm can be
    found in the paper [1].

    Parameters
    ----------
    Pi_hat_last_k : array of shape (n_features, )
        The selection frequency of all input features from the last iteration.

    Returns
    -------
    pi_thr : float
        This is the selection frequency threshold above which a feature is considered selected.

    """

    # fit kernel density estimator
    kde = KernelDensity(bandwidth=np.std(Pi_hat_last_k, ddof=1), kernel='gaussian')
    kde.fit(Pi_hat_last_k[:, None])

    # get evenly spaced data between 0 and 1
    x_d = np.linspace(0, 1, Pi_hat_last_k.size)

    # compute probability of the evenly spaced data based on learned kernel density
    prob = np.exp(kde.score_samples(x_d[:, None]))

    # find the inflexion points corresponding to modes of KDE
    sign_flip_idx_2 = _get_mode_points(prob, type_of_inflexion='max', max_points=None)

    sign_flip_idx_all = _get_mode_points(prob, type_of_inflexion=None, max_points=None)

    sign_flip_idx_min = np.setdiff1d(sign_flip_idx_all, sign_flip_idx_2)

    # pick the anti-mode corresponding to the smallest Pi_hat_last_k
    if sign_flip_idx_min.size == 0:
        pi_thr = 0.5
    else:
        sorted_sign_flip_idx_min_ascend = np.sort(sign_flip_idx_min)
        pi_thr = x_d[sorted_sign_flip_idx_min_ascend[0]]

    return pi_thr


def visualize_selection_frequency_versus_iteration(Pi_hat_seq,
                                                   Pi_hat_k_seq,
                                                   burn_in_length,
                                                   max_features_to_plot):
    """Visualize the selection frequency of input features versus iterations.

    Parameters
    ----------
    Pi_hat_seq : array of shape (n_features, n_iterations)
        This contains the selection frequency of all input features
        over all available iterations.

    Pi_hat_k_seq : array of shape (n_iterations, )
        This contains the iteration numbers corresponding to the columns of `Pi_hat_seq`.

    burn_in_length : int
        The total number of iterations spent in the burn-in stage.

    max_features_to_plot : int
        The maximum number of features whose selection frequency over iterations
        will be plotted.

    Returns
    -------
    None
    """

    # Create the data frame to prepare for plotting
    k_plot_length = Pi_hat_k_seq.size
    M = Pi_hat_seq.shape[0]

    if M <= np.minimum(2000, max_features_to_plot):
        plot_frame = pd.DataFrame({'var_idx': np.tile(np.arange(M), k_plot_length),
                                   'iteration_idx': np.repeat(Pi_hat_k_seq, M),
                                   'pi_hat': Pi_hat_seq.flatten('F')})
    else:
        Pi_thresh = 0.3
        iteration_range = 100

        plot_features_mask = np.any(Pi_hat_seq[:, -iteration_range:] >= Pi_thresh, axis=1)
        prop_var_with_low_Pi_score_to_plot = 0.2
        if M >= 5000:
            prop_var_with_low_Pi_score_to_plot = 1000. / M

        idx_with_low_Pi = np.argwhere(np.logical_not(plot_features_mask)).flatten()
        flip_idx = idx_with_low_Pi[np.argsort(Pi_hat_seq[idx_with_low_Pi, (k_plot_length - 1)])[-np.int(np.round(prop_var_with_low_Pi_score_to_plot * (M - np.sum(plot_features_mask)))):]]
        plot_features_mask[flip_idx] = True

        plot_frame = pd.DataFrame({'var_idx': np.tile(np.argwhere(plot_features_mask).flatten(), k_plot_length),
                                   'iteration_idx': np.repeat(Pi_hat_k_seq, np.sum(plot_features_mask)),
                                   'pi_hat': Pi_hat_seq[plot_features_mask, :].flatten('F')})

    sns.set_style("white")
    f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=150)
    sns.lineplot(x="iteration_idx", y="pi_hat",
                 units="var_idx", estimator=None, lw=1.3,
                 data=plot_frame, legend='full', color='#a3a3c2', sort=True)

    ax.set_xlabel('Number of Iterations k', fontsize=24)
    ax.set_ylabel(r"$\hat{\Pi}^{(k)}_j, \forall j=1,...,M$", fontsize=24)

    if burn_in_length is not None:
        ax.axvline(burn_in_length, ls='--', color=np.array([15, 157, 88]) / 255, lw=2)
    ax.tick_params(axis='both', which='major', labelsize=20)


    plt.show()
    plt.close('all')

