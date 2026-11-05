"""
Synthetic credit dataset generator.

Generates credit applicant data, a demographic group and default labels from a
stated ground truth: hand-chosen distributions and a documented label function, not
a fit to any lending dataset. It contains no real PII.

The features are mostly independent draws. Inflow, outflow and balance follow
income, the offered rate follows FICO, and delinquency recency follows the
delinquency count; the rest is drawn on its own, so the correlations a real credit
file has are absent. On the default 10,000 rows (seed 42; 2026-11-04) FICO's correlation
is -0.007 with delinquencies, 0.009 with utilization and -0.007 with income, and banking
tenure exceeds credit age on 37.7% of rows. tests/test_generator.py re-measures this.

What drives the default label: every field in LABEL_EFFECTS, each with a
documented direction and size (default_log_odds), plus the group-C outcome
gap and noise. That includes the alternative data (rent and
utility payments, savings, balance, banking tenure) and the loan's
affordability. Four generated fields have NO effect and are therefore not
model inputs: open accounts and subscriptions, the loan purpose, and the
offered interest rate (the lender's price, derived from FICO).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# The loan purposes the generator draws. Recorded for reference only: loan purpose
# has no effect on the label and is not a model input, so the API rejects the field.
LOAN_PURPOSES = ("debt_consolidation", "home_improvement", "medical", "auto", "education", "other")


@dataclass
class DatasetConfig:
    """Configuration for synthetic credit dataset generation."""

    n_samples: int = 10_000
    default_rate: float = 0.15       # 15% base default rate
    seed: int = 42

    # Feature noise levels
    income_noise: float = 0.20
    transaction_noise: float = 0.15

    # Demographic group distribution
    group_a_fraction: float = 0.60   # majority group
    group_b_fraction: float = 0.25
    group_c_fraction: float = 0.15

    # Outcome gap (ON by default): group C defaults more often than A or B at
    # the SAME features — a log-odds shift of group_c_bias * 10 in the label.
    # No feature carries it and the model never sees the group, so approval
    # rates stay equal across groups (disparate impact ~1); the audit surfaces
    # it as the model UNDER-predicting group C's risk (calibration by group, with
    # its interval, on the held-out set and on a large fresh draw).
    group_c_bias: float = 0.08

    # Opt-in measurement bias (OFF by default): group C's REPORTED bureau score
    # is understated by this many FICO points (e.g. thin files, positive payment
    # history that never reaches the bureau), while their default risk still
    # follows the true score. A model trained on the reported score then denies
    # creditworthy group-C applicants it should approve — the case the fairness
    # audit exists to catch. 0 leaves the dataset byte-identical.
    group_c_fico_shift: float = 0.0


def _solve_intercept(
    log_odds_no_intercept: np.ndarray,
    target_rate: float,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """
    Find the intercept ``b`` such that the mean of ``sigmoid(log_odds + b)``
    equals ``target_rate``.

    The mean predicted probability is monotonically increasing in the intercept,
    so a simple bisection converges reliably. This is how ``config.default_rate``
    is honored: rather than hardcoding an intercept, we solve for the one that
    makes the realized base rate match the configured target.

    Parameters
    ----------
    log_odds_no_intercept : np.ndarray
        Per-sample log-odds excluding the intercept.
    target_rate : float
        Desired mean default probability (e.g., 0.15).
    tol : float
        Convergence tolerance on the realized mean probability.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    float
        The solved intercept.
    """
    target_rate = float(np.clip(target_rate, 1e-6, 1 - 1e-6))
    lo, hi = -20.0, 20.0

    def mean_prob(b: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(log_odds_no_intercept + b)))))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pm = mean_prob(mid)
        if abs(pm - target_rate) < tol:
            return mid
        if pm > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# Ground truth of the synthetic world: the direct effect of each applicant
# field on the log-odds of default (per unit, or per unit of the stated
# transform). Every model input has an entry here, and nothing else does (net
# cash flow is not an input: it is derived from inflow and outflow):
# tests/test_generative_links.py perturbs each field through default_log_odds()
# and requires the model's monotone direction (features/engineering.py
# RISK_DIRECTION) to match.
LABEL_EFFECTS = {
    # bureau
    "fico_score": "-0.020 per point (centred at 700)",
    "debt_to_income": "+3.5",
    "credit_utilization": "+2.5",
    "num_delinquencies": "+0.9 each",
    "months_since_delinquency": "+1.2 x recency risk, (24 - months) / 24 clipped to 0..1; none if never",
    "num_hard_inquiries": "+0.15 each",
    "credit_age_months": "-0.003 per month",
    # cash flow and payments (the alternative data)
    "nsf_count": "+0.35 each",
    "overdraft_days": "+0.02 per day",
    "bill_payment_consistency": "-1.2",
    "paycheck_regularity": "-1.0",
    "rent_payment_score": "-1.2",
    "utility_payment_score": "-0.8",
    "monthly_inflow": "via net cash flow: -0.00015 per $ of (inflow - outflow)",
    "monthly_outflow": "via net cash flow (outflow lowers it)",
    "savings_rate": "-2.5",
    "avg_account_balance": "-0.2 x log(1 + balance / 1000)",
    "banking_tenure_months": "-0.004 per month",
    "employment_years": "-0.03 per year",
    # the requested loan's affordability: payment burden =
    # (loan_amount / loan_term_months) / (annual_income / 12)
    "loan_amount": "via payment burden: +2.0 per unit of burden",
    "loan_term_months": "via payment burden (a longer term lowers it)",
    "annual_income": "via payment burden (a higher income lowers it)",
}


def default_log_odds(applicants: pd.DataFrame) -> np.ndarray:
    """The group-blind, noise-free part of the synthetic default log-odds
    (before the intercept), from applicant fields — see LABEL_EFFECTS.

    Pure and vectorised so the ground truth is inspectable: perturb a field
    and the change here is its true effect. ``months_since_delinquency`` uses
    -1 for "never delinquent", as in the generated data.
    """
    a = applicants
    months = a["months_since_delinquency"].to_numpy(dtype=float)
    recency_risk = np.where(months < 0, 0.0, np.clip((24.0 - months) / 24.0, 0.0, 1.0))
    net = (a["monthly_inflow"] - a["monthly_outflow"]).to_numpy(dtype=float)
    burden = (
        (a["loan_amount"] / a["loan_term_months"].clip(lower=1))
        / (a["annual_income"] / 12).clip(lower=1)
    ).to_numpy(dtype=float)
    col = lambda name: a[name].to_numpy(dtype=float)  # noqa: E731
    return (
        + 0.020 * (700 - col("fico_score"))
        + 3.5 * col("debt_to_income")
        + 2.5 * col("credit_utilization")
        + 0.9 * col("num_delinquencies")
        + 1.2 * recency_risk
        + 0.15 * col("num_hard_inquiries")
        - 0.003 * col("credit_age_months")
        + 0.35 * col("nsf_count")
        + 0.02 * col("overdraft_days")
        - 1.2 * col("bill_payment_consistency")
        - 1.0 * col("paycheck_regularity")
        - 1.2 * col("rent_payment_score")
        - 0.8 * col("utility_payment_score")
        - 0.00015 * net
        - 2.5 * col("savings_rate")
        - 0.2 * np.log1p(col("avg_account_balance") / 1000.0)
        - 0.004 * col("banking_tenure_months")
        - 0.03 * col("employment_years")
        + 2.0 * burden
    )


def generate_credit_dataset(config: DatasetConfig | None = None) -> pd.DataFrame:
    """
    Generate a synthetic credit dataset from the stated ground truth above.

    Features include:
    - Income and employment features
    - Credit utilization and payment history (FICO-like)
    - Transaction-based alternative features
    - Demographic group (for fairness analysis)
    - Loan characteristics
    - Derived behavioral signals

    Parameters
    ----------
    config : DatasetConfig | None
        Generation configuration. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        Dataset with shape (n_samples, 29): 27 raw features, the demographic
        group, and the target.
        Target column: 'default' (0 = no default, 1 = default).
    """
    if config is None:
        config = DatasetConfig()

    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    # --- Demographic groups ---
    groups = rng.choice(
        ["A", "B", "C"],
        size=n,
        p=[config.group_a_fraction, config.group_b_fraction, config.group_c_fraction],
    )

    # --- Core financial features ---
    # Annual income ($20k – $300k), log-normal
    income = np.clip(
        rng.lognormal(np.log(60_000), 0.8, n) * (1 + rng.normal(0, config.income_noise, n)),
        15_000, 500_000,
    )

    # Loan amount ($1k – $40k)
    loan_amount = np.clip(rng.lognormal(np.log(10_000), 0.7, n), 1_000, 40_000)

    # Debt-to-income ratio (0.05 – 0.70)
    dti = np.clip(rng.beta(2, 5, n) * 0.80 + 0.05, 0.05, 0.70)

    # Employment length in years (0-20)
    employment_years = np.clip(rng.exponential(5, n), 0, 20)

    # --- FICO-like credit history features ---
    fico_score = np.clip(rng.normal(680, 80, n), 300, 850).astype(int)
    credit_utilization = np.clip(rng.beta(2, 4, n), 0.01, 0.99)
    num_open_accounts = np.clip(rng.poisson(8, n), 1, 30)
    num_delinquencies = np.clip(rng.poisson(0.3, n), 0, 10)
    # Recency of the last delinquency exists only for applicants who HAVE one:
    # -1 means "never delinquent" and is exactly the num_delinquencies == 0 set, so
    # the default label below charges recency risk only to those who have one.
    months_since_last_delinquency = np.where(
        num_delinquencies > 0, rng.integers(0, 120, size=n), -1
    )
    num_hard_inquiries = np.clip(rng.poisson(1, n), 0, 10)
    credit_age_months = np.clip(rng.exponential(60, n), 3, 360)

    # --- Alternative / transaction-based features ---
    # Monthly cash inflow (correlated with income)
    monthly_inflow = income / 12 * rng.uniform(0.8, 1.2, n)

    # Monthly cash outflow
    monthly_outflow = monthly_inflow * rng.uniform(0.50, 0.95, n)

    # Net monthly cash flow (positive = surplus)
    monthly_net_cashflow = monthly_inflow - monthly_outflow

    # Paycheck regularity (0-1, higher = more regular direct deposits)
    paycheck_regularity = np.clip(rng.beta(4, 2, n), 0, 1)

    # Bill payment consistency (0-1)
    bill_payment_consistency = np.clip(rng.beta(5, 2, n), 0, 1)

    # NSF (non-sufficient funds) incidents in last 12 months
    nsf_count = np.clip(rng.poisson(0.5, n), 0, 20)

    # Number of overdraft days in last 12 months
    overdraft_days = np.clip(rng.poisson(2, n), 0, 60)

    # Rent payment history (0 = renter with irregular payments, 1 = regular)
    rent_payment_score = np.clip(rng.beta(6, 2, n), 0, 1)

    # Utility payment regularity
    utility_payment_score = np.clip(rng.beta(5, 2, n), 0, 1)

    # Banking relationship length (months)
    banking_tenure_months = np.clip(rng.exponential(36, n), 1, 240)

    # Average account balance (correlated with income)
    avg_balance = income / 12 * rng.uniform(0.5, 3.0, n)

    # Savings rate (fraction of income saved)
    savings_rate = np.clip(rng.beta(2, 5, n) * 0.40, 0, 0.60)

    # Monthly subscription services. No effect on the label, so not a model input.
    num_subscriptions = np.clip(rng.poisson(4, n), 0, 15)

    # --- Loan characteristics ---
    loan_purpose = rng.choice(list(LOAN_PURPOSES), size=n)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n,
                                   p=[0.05, 0.10, 0.50, 0.15, 0.20])
    # The score a lender SEES. Equal to the true score unless the opt-in
    # group_c_fico_shift understates group C's (no random draw: shift 0 keeps
    # the RNG stream, and so the whole dataset, byte-identical).
    reported_fico = fico_score
    if config.group_c_fico_shift:
        reported_fico = np.where(
            groups == "C",
            np.clip(fico_score - config.group_c_fico_shift, 300, 850),
            fico_score,
        ).astype(int)
    # The lender's offered rate, priced on the reported score. Recorded for
    # reference only: it is NOT a model input (features/engineering.py). A score that consumes the lender's own risk-based price
    # is circular — the rate is FICO again (correlation about -0.92) and an
    # outcome of the credit decision, not evidence for it. The draw stays so
    # the random stream, and so every dataset, is unchanged.
    interest_rate_applied = np.clip(
        0.05 + (850 - reported_fico) / 850 * 0.25 + rng.normal(0, 0.01, n),
        0.04, 0.36,
    )

    # --- Generate default labels ---
    # Default probability is a logistic function of the applicant fields, with
    # the documented effects in LABEL_EFFECTS (default_log_odds above), plus
    # the group-C outcome gap and noise. Nothing leaks the answer: the model has
    # to learn the relationship from the features. The label uses the TRUE
    # FICO; the dataset records the reported one (see group_c_fico_shift).
    truth = pd.DataFrame({
        "fico_score": fico_score, "debt_to_income": dti, "credit_utilization": credit_utilization,
        "num_delinquencies": num_delinquencies,
        "months_since_delinquency": months_since_last_delinquency,
        "num_hard_inquiries": num_hard_inquiries, "credit_age_months": credit_age_months,
        "nsf_count": nsf_count, "overdraft_days": overdraft_days,
        "bill_payment_consistency": bill_payment_consistency,
        "paycheck_regularity": paycheck_regularity, "rent_payment_score": rent_payment_score,
        "utility_payment_score": utility_payment_score, "monthly_inflow": monthly_inflow,
        "monthly_outflow": monthly_outflow, "savings_rate": savings_rate,
        "avg_account_balance": avg_balance, "banking_tenure_months": banking_tenure_months,
        "employment_years": employment_years, "loan_amount": loan_amount,
        "loan_term_months": loan_term_months, "annual_income": income,
    }, copy=False)

    # Group-C bias, expressed on the log-odds scale: group_c_bias x 10 is the
    # log-odds shift (0.8 by default). That realizes a gap in default rate of
    # about 7.6 points (ten draws of 400,000, 2026-11-04,
    # tests/test_fairness_population.py); the default 10,000 rows happen to read a
    # 5.6 point gap in default rate (group C 19.6%, group A 14.0%).
    group_c_logit = (groups == "C").astype(float) * config.group_c_bias * 10.0

    # transaction_noise perturbs the log-odds, scaled ×2 so it widens the score
    # spread without swamping the real signal.
    label_noise = rng.normal(0, config.transaction_noise * 2.0, n)

    log_odds_no_intercept = default_log_odds(truth) + group_c_logit + label_noise

    # Solve (bisection) for the intercept that makes the realized mean
    # P(default) ≈ config.default_rate.
    intercept = _solve_intercept(log_odds_no_intercept, config.default_rate)

    log_odds = log_odds_no_intercept + intercept
    default_prob = 1 / (1 + np.exp(-log_odds))
    default = (rng.uniform(0, 1, n) < default_prob).astype(int)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        # Core
        "annual_income": income,
        "loan_amount": loan_amount,
        "debt_to_income": dti,
        "employment_years": employment_years,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate_applied,
        "loan_purpose": loan_purpose,
        # FICO-like (the reported score; the label above used the true one)
        "fico_score": reported_fico,
        "credit_utilization": credit_utilization,
        "num_open_accounts": num_open_accounts,
        "num_delinquencies": num_delinquencies,
        "months_since_delinquency": months_since_last_delinquency,
        "num_hard_inquiries": num_hard_inquiries,
        "credit_age_months": credit_age_months,
        # Transaction/alternative
        "monthly_inflow": monthly_inflow,
        "monthly_outflow": monthly_outflow,
        "monthly_net_cashflow": monthly_net_cashflow,
        "paycheck_regularity": paycheck_regularity,
        "bill_payment_consistency": bill_payment_consistency,
        "nsf_count": nsf_count,
        "overdraft_days": overdraft_days,
        "rent_payment_score": rent_payment_score,
        "utility_payment_score": utility_payment_score,
        "banking_tenure_months": banking_tenure_months,
        "avg_account_balance": avg_balance,
        "savings_rate": savings_rate,
        "num_subscriptions": num_subscriptions,
        # Demographic
        "demographic_group": groups,
        # Target
        "default": default,
    }, copy=False)  # these arrays are this frame's alone: a copy doubled a large draw's peak

    return df


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
