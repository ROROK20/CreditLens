"""
CreditLens — explainable credit scoring with cash-flow signals.

Scores applicants from a FICO score (required) plus transaction and cash-flow
behaviour, explains each decision with Shapley values of the served
probability, and audits fairness by demographic group with its own
uncertainty. Synthetic data only; a reference implementation.
"""

__version__ = "0.1.0"
__author__ = "Rohan Khiani"
__license__ = "MIT"
