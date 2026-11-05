"""The library path refuses to score without a required input.

README: "A FICO score is required." Median-filling an absent column moves a low-FICO
applicant's score from 607 to 972 with `fico_score` dropped, and a NaN recency is not a
claim of "never delinquent". So every input is required at scoring time, in the library
as in the API; training on the generator's own values is unaffected.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.decision import probability_to_score
from creditlens.features.engineering import MODEL_INPUTS, RECENCY_PLATEAU_MONTHS, FeatureEngineer
from creditlens.models.scorer import LogisticScorer


@pytest.fixture(scope="module")
def data():
    train, test = train_test_split(generate_credit_dataset(DatasetConfig()))
    return train, test


@pytest.fixture(scope="module")
def fitted(data):
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(data[0])
    return engineer, LogisticScorer().fit(X, y)


@pytest.mark.parametrize("name", MODEL_INPUTS)
def test_an_absent_input_is_refused_by_name(fitted, data, name):
    engineer, _ = fitted
    with pytest.raises(ValueError, match=rf"{name} \(column absent\)"):
        engineer.transform(data[1].iloc[[0]].drop(columns=[name]))


@pytest.mark.parametrize("name", MODEL_INPUTS)
def test_a_nan_input_is_refused_by_name_with_its_count(fitted, data, name):
    engineer, _ = fitted
    rows = data[1].iloc[:3].copy()
    rows[name] = [rows[name].iloc[0], np.nan, np.nan]
    with pytest.raises(ValueError, match=rf"{name} \(NaN in 2 of 3 rows\)"):
        engineer.transform(rows)


def test_a_dropped_fico_cannot_lift_a_weak_applicant(fitted, data):
    """Median-filled, this 607 would read 972."""
    engineer, model = fitted
    row = data[1][data[1].fico_score < 520].iloc[[0]]
    assert probability_to_score(model.predict_proba(engineer.transform(row)[0])[:, 1])[0] < 800
    with pytest.raises(ValueError, match="fico_score"):
        engineer.transform(row.drop(columns=["fico_score"]))


def test_nan_recency_is_not_never_delinquent(fitted, data):
    engineer, _ = fitted
    row = data[1][data[1].num_delinquencies >= 1].iloc[[0]].copy()
    row["months_since_delinquency"] = np.nan
    with pytest.raises(ValueError, match="months_since_delinquency"):
        engineer.transform(row)


def test_every_generator_row_scores_with_no_warning(fitted, data):
    """-1 ("never delinquent") is the generator's own value, not a missing one."""
    engineer, _ = fitted
    assert (data[1].months_since_delinquency == -1).any()
    assert not data[1][MODEL_INPUTS].isna().any().any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X, _ = engineer.transform(data[1])
    assert not X.isna().to_numpy().any()


def test_a_redundant_net_cashflow_does_not_excuse_absent_cash_flows(fitted, data):
    engineer, _ = fitted
    row = data[1].iloc[[0]]
    assert "monthly_net_cashflow" in row.columns
    with pytest.raises(ValueError, match="monthly_inflow"):
        engineer.transform(row.drop(columns=["monthly_inflow"]))


def test_opting_in_fills_and_warns_what_was_filled(data):
    engineer = FeatureEngineer(impute_missing=True)
    engineer.fit_transform(data[0])
    row = data[1].iloc[[0]].drop(columns=["fico_score"]).assign(savings_rate=np.nan)
    with pytest.warns(UserWarning, match=r"fico_score \(column absent\).*savings_rate \(NaN in 1 of 1 rows\)"):
        X, _ = engineer.transform(row)
    assert not X.isna().to_numpy().any()


def test_training_handling_of_the_generators_values_is_unchanged(data):
    """Never-delinquent maps to the plateau; a real recency stays, capped."""
    engineer = FeatureEngineer(scale=False)
    X, _ = engineer.fit_transform(data[0])
    months = data[0].months_since_delinquency
    expected = months.where(months >= 0, RECENCY_PLATEAU_MONTHS).clip(upper=RECENCY_PLATEAU_MONTHS)
    pd.testing.assert_series_equal(X["months_since_delinquency"], expected.astype(float), check_names=False)


def test_a_training_frame_with_nan_recency_does_not_learn_never_delinquent(data):
    """Fit-time NaN goes to the imputer as missing, not to the "never delinquent" plateau
    by a comparison that happens to be False. Delinquent rows only, so the two differ."""
    frame = data[0][data[0].months_since_delinquency.between(0, 23)].copy()
    blanked = frame.index[:5]
    frame.loc[blanked, "months_since_delinquency"] = np.nan
    X, _ = FeatureEngineer(scale=False).fit_transform(frame)
    median = frame.months_since_delinquency.median()
    assert median < RECENCY_PLATEAU_MONTHS
    assert (X.loc[blanked, "months_since_delinquency"] == median).all()
