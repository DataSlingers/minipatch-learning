import inspect
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix
import texttable


class BaseLearner:
    """Base class for the estimators in this software package.

    Note: the following code is adapted from the `sklearn.base.BaseEstimator`.
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "estimators should always "
                    "specify parameters in the signature"
                    " of __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


def _eval_selection_accuracy(true_support_indicator, est_support_indicator):

    tn, fp, fn, tp = confusion_matrix(true_support_indicator, est_support_indicator).ravel()

    est_support_total = np.sum(est_support_indicator)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    tnr = tn / (fp + tn)

    prec = tp / (tp + fp)

    f1 = 2*((prec * tpr) / (prec + tpr))

    fdp = fp / np.maximum(est_support_total, 1)

    aggregate_results = [est_support_total, tpr, prec, f1]

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "est_support_total": est_support_total,
            "tpr": tpr, "fpr": fpr, "fnr": fnr, "tnr": tnr,
            "prec": prec, "f1": f1, "fdp": fdp,
            "aggregate_results": aggregate_results}


def _build_table(aggregate_results):

    tab = texttable.Texttable()
    tab.header(['TotalSelected', 'TPR', 'Precision', 'F1 Score'])
    tab.add_row(aggregate_results)

    s = tab.draw()
    print(s)


def display_selection_accuracy(true_support_mask, est_support_mask):
    """Various metrics to evaluate selection performance.

    Parameters
    ----------
    true_support_mask : array of shape (n_features, )
        This is a boolean array of shape [# input features], in which an element
        is True iff its corresponding feature is a truly informative feature.

    est_support_mask : : array of shape (n_features, )
        This is a boolean array of shape [# input features], in which an element
        is True iff its corresponding feature is selected by the algorithm.

    Returns
    -------
    eval_results : dict
        Contains various performance measure such as TPR, F1 Score.

    """

    true_support_indicator = true_support_mask.astype(np.uint8)
    est_support_indicator = est_support_mask.astype(np.uint8)

    eval_results = _eval_selection_accuracy(true_support_indicator, est_support_indicator)

    _build_table(eval_results["aggregate_results"])

    return eval_results





