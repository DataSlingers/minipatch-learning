#############
API Reference
#############

This page contains detailed API references for the minipatch-learning package.


Feature Selection
=================

.. automodule:: mplearn.feature_selection
   :no-members:
   :no-inherited-members:

The following is the API documentation for the feature selection algorithms in the package.

.. currentmodule:: mplearn

Minipatch feature selection
---------------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_selection.AdaSTAMPS

Base feature selector
---------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_selection.base_selector.ThresholdedOLS
   feature_selection.base_selector.DecisionTreeSelector


Gaussian Graphical Model Selection
==================================

.. automodule:: mplearn.graphical_model
   :no-members:
   :no-inherited-members:

The following is the API documentation for the Gaussian graphical model estimators in the package.

.. currentmodule:: mplearn

Minipatch graph selection
-------------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   graphical_model.MPGraph

Base graph selector
-------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   graphical_model.base_graph.ThresholdedGraphicalLasso

