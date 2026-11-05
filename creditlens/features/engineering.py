"""
Feature engineering for credit scoring.

Transforms raw applicant data into model-ready features:
- the new loan's payment burden
- net cash flow (inflow - outflow), always derived
- delinquency recency capped where its effect on default ends
- account balance on the log scale the label uses
- a refusal, at scoring time, of any required input that is absent or NaN
- feature scaling

The model features are exactly the terms of the synthetic default label
(data/generator.py default_log_odds), each once, with the monotone direction
of its effect (RISK_DIRECTION); tests/test_generative_links.py checks both,
and measures how far the served model's response to each input tracks its true
effect (README "Model behaviour" publishes the table).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# The raw applicant fields the model reads: the API's numeric inputs. Every
# model feature below is computed from these. Net cash flow is not among them: it
# is derived, always (see _add_derived_features).
MODEL_INPUTS = [
    "annual_income", "loan_amount", "loan_term_months", "debt_to_income",
    "employment_years", "fico_score", "credit_utilization", "num_delinquencies",
    "months_since_delinquency", "num_hard_inquiries", "credit_age_months",
    "monthly_inflow", "monthly_outflow",
    "paycheck_regularity", "bill_payment_consistency", "nsf_count",
    "overdraft_days", "rent_payment_score", "utility_payment_score",
    "banking_tenure_months", "avg_account_balance", "savings_rate",
]

# The model's features: the label's own terms, each once, so each input reaches the
# model by one path. Fields with no effect on the label stay out, and raw income, loan
# amount, term, inflow and outflow reach it only through the burden and net cash flow
# they act through: a second path would let the model learn income as a proxy.
NUMERIC_FEATURES = [
    "fico_score", "debt_to_income", "credit_utilization",
    "num_delinquencies", "months_since_delinquency", "num_hard_inquiries",
    "credit_age_months", "nsf_count", "overdraft_days",
    "bill_payment_consistency", "paycheck_regularity", "rent_payment_score",
    "utility_payment_score", "monthly_net_cashflow", "savings_rate",
    "avg_account_balance", "banking_tenure_months", "employment_years",
    # Derived (added by _add_derived_features): the loan's payment burden,
    # (loan_amount / loan_term_months) / (annual_income / 12).
    "monthly_payment_burden",
]

CATEGORICAL_FEATURES: list[str] = []

# The recency of the last delinquency stops mattering at 24 months in the label
# (data/generator.py: risk ramps from 1 at 0 months to 0 at 24). The feature is
# capped there, and "never delinquent" (-1) is mapped to the cap, so recency earns
# nothing past the plateau.
RECENCY_PLATEAU_MONTHS = 24

# Direction of default risk for every model feature: +1 = a higher value may
# only RAISE predicted P(default), -1 = may only LOWER it. It is the direction
# of the feature's effect in the synthetic label (data/generator.py
# LABEL_EFFECTS). LogisticScorer enforces it as bounds on its weights and
# XGBoostScorer as `monotone_constraints`. Standard scaling is increasing, so the
# directions hold in raw units.
RISK_DIRECTION: dict[str, int] = {
    "fico_score": -1, "debt_to_income": +1, "credit_utilization": +1,
    "num_delinquencies": +1, "months_since_delinquency": -1,  # capped at the plateau
    "num_hard_inquiries": +1, "credit_age_months": -1,
    "employment_years": -1, "monthly_net_cashflow": -1,
    "paycheck_regularity": -1, "bill_payment_consistency": -1,
    "nsf_count": +1, "overdraft_days": +1,
    "rent_payment_score": -1, "utility_payment_score": -1,
    "banking_tenure_months": -1, "avg_account_balance": -1, "savings_rate": -1,
    "monthly_payment_burden": +1,
}

# The same direction for every raw input. Five act only through a derived
# feature: loan amount, term and income through the payment burden, inflow and
# outflow through net cash flow. Each moves that feature in the direction that
# changes risk its own way, so the served score is monotone in every input.
INPUT_RISK_DIRECTION: dict[str, int] = {
    **{name: RISK_DIRECTION[name] for name in MODEL_INPUTS if name in RISK_DIRECTION},
    "annual_income": -1, "loan_amount": +1, "loan_term_months": -1,
    "monthly_inflow": -1, "monthly_outflow": +1,
}

# Attribution is over the applicant's INPUTS, not the model's features, so a
# reason code names something the applicant supplied. Fields
# the API couples move together as one player: income with the cash flows and
# balance it bounds (inflow 0.78-1.22 x income / 12, outflow a share of inflow,
# balance a multiple of income / 12), and the delinquency count with its recency
# (-1 exactly when the count is 0). Net cash flow is derived inside the value
# function, never varied on its own.
ATTRIBUTION_GROUPS: dict[str, tuple[str, ...]] = {
    "income_cash_flow_and_balance": ("annual_income", "monthly_inflow", "monthly_outflow",
                                     "avg_account_balance"),
    "delinquency_history": ("num_delinquencies", "months_since_delinquency"),
}


def attribution_players() -> dict[str, tuple[str, ...]]:
    """Player name -> the input fields it moves, in MODEL_INPUTS order."""
    players: dict[str, tuple[str, ...]] = {}
    for name in MODEL_INPUTS:
        group = next((g for g, fields in ATTRIBUTION_GROUPS.items() if name in fields), None)
        if group is None:
            players[name] = (name,)
        elif group not in players:
            players[group] = ATTRIBUTION_GROUPS[group]
    return players

TARGET_COL = "default"
GROUP_COL = "demographic_group"

# The raw inputs a derived model feature is computed from.
_DERIVED_FROM: dict[str, tuple[str, ...]] = {
    "monthly_payment_burden": ("annual_income", "loan_amount", "loan_term_months"),
    "monthly_net_cashflow": ("monthly_inflow", "monthly_outflow"),
}


class FeatureEngineer:
    """
    Transforms raw credit application data into model features.

    Parameters
    ----------
    scale : bool
        Apply StandardScaler to numeric features. Default True.
    impute_missing : bool
        What ``transform`` does with a required input that is absent or NaN.
        False (default) refuses with a ``ValueError`` naming it. True fills it
        with the training median and warns what was filled: for analysis on
        incomplete data, never for a decision about an applicant.

    Examples
    --------
    >>> from creditlens.data.generator import generate_credit_dataset
    >>> df = generate_credit_dataset()
    >>> engineer = FeatureEngineer()
    >>> X_train, y_train = engineer.fit_transform(df_train)
    >>> X_test, y_test = engineer.transform(df_test)
    """

    def __init__(self, scale: bool = True, impute_missing: bool = False) -> None:
        self.scale = scale
        self.impute_missing = impute_missing
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy="median")
        self._cat_encoders: dict[str, LabelEncoder] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit transformer on training data and transform.

        Parameters
        ----------
        df : pd.DataFrame
            Raw training data.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (X_train features, y_train labels)
        """
        df = self._add_derived_features(df.copy())
        y = df[TARGET_COL].astype(int)

        # Encode categoricals
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                self._cat_encoders[col] = enc

        # Select feature columns
        feat_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
        X = df[feat_cols].copy()

        # Impute
        X_arr = self._imputer.fit_transform(X)
        X = pd.DataFrame(X_arr, columns=feat_cols, index=df.index)

        # Scale
        if self.scale:
            X_arr = self._scaler.fit_transform(X)
            X = pd.DataFrame(X_arr, columns=feat_cols, index=df.index)

        self._feature_names = feat_cols
        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Transform new data using fitted parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Raw applicant data.

        Missing inputs
        --------------
        A required input that is absent or NaN is refused, because every way of
        scoring without it asserts something about the applicant. Filling with the
        training median moves a low-FICO row's score from 607 to 972 with `fico_score`
        dropped (measured 2026-11-04). `months_since_delinquency` says "never" with -1,
        so NaN never has to mean it.

        `impute_missing=True` applies the median fill for analysis on incomplete
        data, with a warning naming what was filled. A zero fill is not an option: it
        asserts a value no applicant stated.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            (X features, y labels if TARGET_COL present else None)
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fit before transform. Call fit_transform first.")

        self._check_required_inputs(df)
        df = self._add_derived_features(df.copy())
        y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None

        for col, enc in self._cat_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                known = set(enc.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else enc.classes_[0])
                df[col] = enc.transform(df[col])

        feat_cols = self._feature_names
        X = df[[c for c in feat_cols if c in df.columns]].copy()

        # Only with impute_missing=True can a trained column still be absent here
        # (see _check_required_inputs): NaN, which the imputer below fills with the
        # training median. A 0.0 fill would be a confident false claim.
        for col in feat_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feat_cols]

        X_arr = self._imputer.transform(X)
        X = pd.DataFrame(X_arr, columns=feat_cols, index=df.index)

        if self.scale:
            X_arr = self._scaler.transform(X)
            X = pd.DataFrame(X_arr, columns=feat_cols, index=df.index)

        return X, y

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names.copy()


    def _check_required_inputs(self, df: pd.DataFrame) -> None:
        """Refuse (or, with impute_missing, warn about) absent or NaN required inputs."""
        required = dict.fromkeys(
            field for feature in self._feature_names for field in _DERIVED_FROM.get(feature, (feature,))
        )
        problems = [f"{name} (column absent)" for name in required if name not in df.columns]
        present = [name for name in required if name in df.columns]
        for name, count in df[present].isna().sum().items():
            if count:
                problems.append(f"{name} (NaN in {count} of {len(df)} rows)")
        if not problems:
            return
        listed = "; ".join(problems)
        if not self.impute_missing:
            raise ValueError(
                f"Cannot score with required inputs missing: {listed}. Supply them "
                "(months_since_delinquency is -1 for never delinquent), or pass "
                "FeatureEngineer(impute_missing=True) to fill them with training medians."
            )
        warnings.warn(f"Filled with training medians, so these scores are not the applicant's: {listed}.",
                      UserWarning, stacklevel=3)

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features from raw columns."""
        # Monthly payment burden: estimated payment / monthly income (the same
        # definition the synthetic label uses)
        monthly_income = df["annual_income"] / 12
        monthly_payment = df["loan_amount"] / df["loan_term_months"].clip(lower=1)
        df["monthly_payment_burden"] = monthly_payment / monthly_income.clip(lower=1)

        # monthly_net_cashflow is a TRAINED feature that the generator defines
        # exactly as inflow - outflow (data/generator.py), so it is always that:
        # one path for every caller. A frame without inflow and outflow is refused
        # at scoring time (_check_required_inputs). A value already present never wins,
        # so a sent or stale figure cannot score differently from the same applicant
        # without it.
        if {"monthly_inflow", "monthly_outflow"} <= set(df.columns):
            df["monthly_net_cashflow"] = df["monthly_inflow"] - df["monthly_outflow"]

        # Delinquency recency, capped at the plateau where the label stops
        # caring; "never delinquent" (-1) sits at the cap. NaN stays NaN: it is not
        # a claim that the applicant was never delinquent.
        if "months_since_delinquency" in df.columns:
            months = df["months_since_delinquency"]
            df["months_since_delinquency"] = months.where(months.isna() | (months >= 0),
                                                          RECENCY_PLATEAU_MONTHS).clip(
                upper=RECENCY_PLATEAU_MONTHS)

        # The label's balance term is log(1 + balance / 1000), so the feature is too:
        # a linear model reads the same transform the generator uses. It increases
        # with balance, so RISK_DIRECTION holds in raw units.
        if "avg_account_balance" in df.columns:
            df["avg_account_balance"] = np.log1p(df["avg_account_balance"] / 1000.0)

        return df
