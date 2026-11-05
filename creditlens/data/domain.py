"""The validity domain: the input region the model has training data for.

A model has no evidence outside its training range. A tree scores an input
beyond it like the nearest training value, and a logistic model extends a
straight line that nothing tested.

So every input is bounded to the default training data's observed range,
widened by ``MARGIN`` (5%) of that range on each side, rounded outward, and
never past a field's physical limit. The five dollar amounts are drawn
log-normal, so their margin is taken on a log scale. Five joint rules bound
what the marginals cannot:

- ``monthly_inflow`` is income-linked: the generator draws it as
  ``annual_income / 12 x U(0.80, 1.20)``, so the API requires the ratio in
  ``INFLOW_TO_MONTHLY_INCOME`` (0.80-1.20 widened by 5% of its width).
- ``monthly_outflow`` is inflow-linked: the generator draws it as
  ``monthly_inflow x U(0.50, 0.95)``, so the API requires the ratio in
  ``OUTFLOW_TO_INFLOW``.
- ``avg_account_balance`` is income-linked: the generator draws it as
  ``annual_income / 12 x U(0.5, 3.0)``, so the API requires the ratio in
  ``BALANCE_TO_MONTHLY_INCOME``.
- the monthly loan payment (``loan_amount / loan_term_months``) as a share of
  monthly income is at most ``MAX_PAYMENT_BURDEN`` (training maximum 2.67,
  plus 5%).
- ``monthly_inflow - monthly_outflow`` stays in the ``monthly_net_cashflow``
  range. The engine derives that field; it is not an input.

Inside the domain, a value outside the central 99% of the training data
(``WARNING_BAND``: its 0.5th to 99.5th percentile) is scored and flagged in
the response's ``warnings``: few training applicants look like it. The three ratios
above and the payment burden are flagged the same way. So are two or more of those
ratios in their most favourable tenth at once (``CASH_FLOW_EDGE``), the pattern of a
stated cash flow written to pass: these inputs are self-reported, and README
"Model behaviour" measures what claiming them does.

The literals below are what ``derive_validity_domain`` returns for the
default training data; tests/test_validation.py fails if they ever differ.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MARGIN = 0.05
CENTRAL_QUANTILES = (0.005, 0.995)
# Each cash-flow ratio's most favourable share of the training data.
FAVOURABLE_SHARE = 0.10

# Drawn log-normal (or as a multiple of a log-normal income): margin on log scale.
LOG_SCALE = ("annual_income", "loan_amount", "monthly_inflow", "monthly_outflow",
             "avg_account_balance")
INTEGER_FIELDS = ("loan_term_months", "fico_score", "num_delinquencies",
                  "months_since_delinquency", "num_hard_inquiries", "nsf_count",
                  "overdraft_days")
UNIT_FIELDS = ("debt_to_income", "credit_utilization", "paycheck_regularity",
               "bill_payment_consistency", "rent_payment_score",
               "utility_payment_score", "savings_rate")
# Fields measured in whole months or years: bounds round outward to integers.
WHOLE_FIELDS = ("employment_years", "credit_age_months", "banking_tenure_months")

# Physical limits a widened bound never crosses.
PHYSICAL = {
    "fico_score": (300, 850),
    # Credit files keep a delinquency about 7 years; 10 years is the API cap.
    "months_since_delinquency": (0, 120),
    **{name: (0, 1) for name in UNIT_FIELDS},
}

DOMAIN_FIELDS = (
    "annual_income", "loan_amount", "loan_term_months", "debt_to_income",
    "employment_years", "fico_score", "credit_utilization", "num_delinquencies",
    "months_since_delinquency", "num_hard_inquiries", "credit_age_months",
    "monthly_inflow", "monthly_outflow", "monthly_net_cashflow",
    "paycheck_regularity", "bill_payment_consistency", "nsf_count",
    "overdraft_days", "rent_payment_score", "utility_payment_score",
    "banking_tenure_months", "avg_account_balance", "savings_rate",
)


def _sig2(x: float, up: bool) -> float:
    """Round to two significant figures, away from the domain's centre."""
    if x == 0:
        return 0.0
    step = 10 ** (math.floor(math.log10(abs(x))) - 1)
    return float((math.ceil if up else math.floor)(x / step) * step)


def _round_out(name: str, value: float, up: bool) -> float:
    if name in UNIT_FIELDS:
        return (math.ceil if up else math.floor)(round(value * 100, 6)) / 100
    if name in INTEGER_FIELDS or name in WHOLE_FIELDS:
        return float((math.ceil if up else math.floor)(round(value, 6)))
    return _sig2(value, up)


def _round_in(name: str, value: float, up: bool) -> float:
    """Round a warning threshold toward the centre (more warnings, never fewer)."""
    if name in UNIT_FIELDS:
        return (math.ceil if up else math.floor)(round(value * 1000, 6)) / 1000
    if name in INTEGER_FIELDS:
        return float((math.ceil if up else math.floor)(round(value, 6)))
    return _sig2(value, up)


def _observed(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name].astype(float)
    # -1 is the "never delinquent" sentinel, not a month count.
    return s[s >= 0] if name == "months_since_delinquency" else s


def _payment_burden(df: pd.DataFrame) -> pd.Series:
    return (df["loan_amount"] / df["loan_term_months"]) / (df["annual_income"] / 12)


def _ratio_bounds(ratio: pd.Series) -> tuple[float, float]:
    """A joint rule's range: observed, widened by MARGIN, rounded outward to 0.01."""
    pad = MARGIN * (ratio.max() - ratio.min())
    return (math.floor(round((ratio.min() - pad) * 100, 6)) / 100,
            math.ceil(round((ratio.max() + pad) * 100, 6)) / 100)


def _ratio_band(ratio: pd.Series) -> tuple[float, float]:
    """A joint rule's warning band: its central 99%, rounded inward to 0.001."""
    lo, hi = ratio.quantile(CENTRAL_QUANTILES[0]), ratio.quantile(CENTRAL_QUANTILES[1])
    return (math.ceil(round(lo * 1000, 6)) / 1000, math.floor(round(hi * 1000, 6)) / 1000)


def _favourable_edge(ratio: pd.Series, high_is_favourable: bool) -> float:
    """Where a ratio's most favourable tenth starts, rounded inward to 0.01."""
    if high_is_favourable:
        return math.floor(round(float(ratio.quantile(1 - FAVOURABLE_SHARE)) * 100, 6)) / 100
    return math.ceil(round(float(ratio.quantile(FAVOURABLE_SHARE)) * 100, 6)) / 100


def derive_validity_domain(df: pd.DataFrame) -> dict:
    """The domain, joint rules and warning band from a training frame."""
    domain, band = {}, {}
    for name in DOMAIN_FIELDS:
        s = _observed(df, name)
        lo, hi = float(s.min()), float(s.max())
        if name in LOG_SCALE:
            pad = MARGIN * (math.log(hi) - math.log(lo))
            lo, hi = lo / math.exp(pad), hi * math.exp(pad)
        else:
            pad = MARGIN * (hi - lo)
            lo, hi = lo - pad, hi + pad
        lo, hi = _round_out(name, lo, up=False), _round_out(name, hi, up=True)
        if name != "monthly_net_cashflow":  # the only field that may be negative
            lo = max(lo, 0.0)
        if name in PHYSICAL:
            lo, hi = max(lo, PHYSICAL[name][0]), min(hi, PHYSICAL[name][1])
        domain[name] = (lo, hi)
        qlo, qhi = s.quantile(CENTRAL_QUANTILES[0]), s.quantile(CENTRAL_QUANTILES[1])
        band[name] = (_round_in(name, qlo, up=True), _round_in(name, qhi, up=False))

    monthly_income = df["annual_income"] / 12
    inflow_ratio = df["monthly_inflow"] / monthly_income
    outflow_ratio = df["monthly_outflow"] / df["monthly_inflow"]
    balance_ratio = df["avg_account_balance"] / monthly_income
    burden = _payment_burden(df)
    max_burden = math.ceil(round(burden.max() * (1 + MARGIN) * 10, 6)) / 10
    burden_warn = math.floor(round(float(np.quantile(burden, CENTRAL_QUANTILES[1])) * 100, 6)) / 100
    return {"domain": domain, "warning_band": band,
            "inflow_to_monthly_income": _ratio_bounds(inflow_ratio),
            "inflow_to_monthly_income_warning": _ratio_band(inflow_ratio),
            "cash_flow_edge": {"inflow_to_monthly_income": _favourable_edge(inflow_ratio, True),
                               "outflow_to_inflow": _favourable_edge(outflow_ratio, False),
                               "balance_to_monthly_income": _favourable_edge(balance_ratio, True)},
            "outflow_to_inflow": _ratio_bounds(outflow_ratio),
            "outflow_to_inflow_warning": _ratio_band(outflow_ratio),
            "balance_to_monthly_income": _ratio_bounds(balance_ratio),
            "balance_to_monthly_income_warning": _ratio_band(balance_ratio),
            "max_payment_burden": max_burden, "payment_burden_warning": burden_warn}


# --- the default training data's domain (DatasetConfig(): 10,000 rows, seed 42)
VALIDITY_DOMAIN: dict[str, tuple[float, float]] = {
    "annual_income": (12_000, 600_000),
    "loan_amount": (830, 49_000),
    "loan_term_months": (9, 63),
    "debt_to_income": (0.01, 0.74),
    "employment_years": (0, 21),
    "fico_score": (350, 850),
    "credit_utilization": (0, 0.98),
    "num_delinquencies": (0, 5),
    "months_since_delinquency": (0, 120),
    "num_hard_inquiries": (0, 8),
    "credit_age_months": (0, 378),
    "monthly_inflow": (820, 61_000),
    "monthly_outflow": (410, 57_000),
    "monthly_net_cashflow": (-1_200, 25_000),
    "paycheck_regularity": (0, 1),
    "bill_payment_consistency": (0.05, 1),
    "nsf_count": (0, 6),
    "overdraft_days": (0, 10),
    "rent_payment_score": (0.13, 1),
    "utility_payment_score": (0.06, 1),
    "banking_tenure_months": (0, 252),
    "avg_account_balance": (480, 170_000),
    "savings_rate": (0, 0.38),
}
WARNING_BAND: dict[str, tuple[float, float]] = {
    "annual_income": (15_000, 480_000),
    "loan_amount": (1_700, 40_000),
    "loan_term_months": (12, 60),
    "debt_to_income": (0.065, 0.644),
    "employment_years": (0.022, 20),
    "fico_score": (475, 850),
    "credit_utilization": (0.024, 0.813),
    "num_delinquencies": (0, 2),
    "months_since_delinquency": (0, 119),
    "num_hard_inquiries": (0, 4),
    "credit_age_months": (3, 310),
    "monthly_inflow": (1_100, 38_000),
    "monthly_outflow": (640, 29_000),
    "monthly_net_cashflow": (97, 13_000),
    "paycheck_regularity": (0.184, 0.978),
    "bill_payment_consistency": (0.25, 0.98),
    "nsf_count": (0, 3),
    "overdraft_days": (0, 6),
    "rent_payment_score": (0.3, 0.984),
    "utility_payment_score": (0.26, 0.981),
    "banking_tenure_months": (1, 190),
    "avg_account_balance": (870, 81_000),
    "savings_rate": (0.008, 0.303),
}
INFLOW_TO_MONTHLY_INCOME = (0.78, 1.22)
INFLOW_TO_MONTHLY_INCOME_WARNING = (0.802, 1.198)
OUTFLOW_TO_INFLOW = (0.47, 0.98)
OUTFLOW_TO_INFLOW_WARNING = (0.503, 0.947)
BALANCE_TO_MONTHLY_INCOME = (0.37, 3.13)
BALANCE_TO_MONTHLY_INCOME_WARNING = (0.511, 2.988)
MAX_PAYMENT_BURDEN = 2.8
PAYMENT_BURDEN_WARNING = 0.88
CASH_FLOW_EDGE = {"inflow_to_monthly_income": 1.15, "outflow_to_inflow": 0.55,
                  "balance_to_monthly_income": 2.74}


def domain_warnings(values: dict) -> list[str]:
    """Plain-language flags for in-domain values few training applicants share.

    Net cash flow is derived here as everywhere, so no request can skip its warning."""
    out = []
    values = {**values, "monthly_net_cashflow": values["monthly_inflow"] - values["monthly_outflow"]}
    for name, (lo, hi) in WARNING_BAND.items():
        v = values.get(name)
        if v is None or (name == "months_since_delinquency" and v < 0):
            continue
        if v < lo or v > hi:
            side, edge = ("below", lo) if v < lo else ("above", hi)
            out.append(f"{name} {v:g} is {side} {edge:g}, outside the central 99% of "
                       "the training data: few training applicants look like this, "
                       "so treat the score as less certain")
    monthly_income = values["annual_income"] / 12
    inflow = values["monthly_inflow"] / monthly_income
    outflow = values["monthly_outflow"] / values["monthly_inflow"]
    balance = values["avg_account_balance"] / monthly_income
    for name, ratio, (lo, hi) in (
        ("monthly_inflow / (annual_income / 12)", inflow, INFLOW_TO_MONTHLY_INCOME_WARNING),
        ("monthly_outflow / monthly_inflow", outflow, OUTFLOW_TO_INFLOW_WARNING),
        ("avg_account_balance / (annual_income / 12)", balance, BALANCE_TO_MONTHLY_INCOME_WARNING),
    ):
        if ratio < lo or ratio > hi:
            side, edge = ("below", lo) if ratio < lo else ("above", hi)
            out.append(f"{name} is {ratio:.3f}, {side} {edge:g}, outside the central 99% of the "
                       "training data: treat the score as less certain")
    burden = (values["loan_amount"] / values["loan_term_months"]) / monthly_income
    if burden > PAYMENT_BURDEN_WARNING:
        out.append(f"the monthly loan payment is {burden:.2f}x monthly income, above "
                   f"{PAYMENT_BURDEN_WARNING:g} (the central 99% of the training data): "
                   "treat the score as less certain")
    at_edge = [name for name, on_edge in (
        ("inflow", inflow >= CASH_FLOW_EDGE["inflow_to_monthly_income"]),
        ("outflow", outflow <= CASH_FLOW_EDGE["outflow_to_inflow"]),
        ("balance", balance >= CASH_FLOW_EDGE["balance_to_monthly_income"]),
    ) if on_edge]
    if len(at_edge) >= 2:
        out.append(f"{', '.join(at_edge)} are each in the most favourable tenth of the training data at "
                   "once: these figures are self-reported and stated cash flow can be written to pass, "
                   "so verify them independently before relying on this score")
    return out
