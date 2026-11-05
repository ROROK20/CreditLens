"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from creditlens.data.generator import generate_credit_dataset, DatasetConfig
from creditlens.features.engineering import FeatureEngineer


@pytest.fixture(scope="module")
def dataset():
    config = DatasetConfig(n_samples=500, seed=42)
    return generate_credit_dataset(config)


class TestFeatureEngineer:
    def test_fit_transform_returns_dataframe_series(self, dataset):
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(dataset.copy())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_features(self, dataset):
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(dataset.copy())
        assert "default" not in X.columns

    def test_group_not_in_features(self, dataset):
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(dataset.copy())
        assert "demographic_group" not in X.columns

    def test_no_nan_after_transform(self, dataset):
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(dataset.copy())
        assert not X.isna().any().any()

    def test_derived_features_created(self, dataset):
        engineer = FeatureEngineer()
        X, _ = engineer.fit_transform(dataset.copy())
        assert "monthly_payment_burden" in X.columns
        # Composites re-weighting fields already present would give inputs a second
        # path into the model.
        assert "alternative_credit_score" not in X.columns
        assert "payment_reliability_index" not in X.columns

    def test_transform_before_fit_raises(self, dataset):
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError):
            engineer.transform(dataset.copy())

    def test_transform_consistent_with_fit_transform(self, dataset):
        """Transform on train data should produce same result as fit_transform."""
        train, test = dataset[:400], dataset[400:]
        engineer = FeatureEngineer(scale=False)  # no scale for exact comparison
        X_train, y_train = engineer.fit_transform(train.copy())
        X_test, y_test = engineer.transform(test.copy())
        assert X_test.columns.tolist() == X_train.columns.tolist()

    def test_fields_without_an_effect_on_default_are_not_features(self, dataset):
        """loan_purpose, open accounts and subscriptions have no effect on the
        synthetic default label, so they are not model features at all."""
        engineer = FeatureEngineer()
        X, _ = engineer.fit_transform(dataset.copy())
        for col in ("loan_purpose", "num_open_accounts", "num_subscriptions", "interest_rate",
                    "loan_to_income_ratio", "cashflow_stability"):
            assert col not in X.columns, col
        assert all(X[c].dtype.kind == "f" for c in X.columns)

    def test_feature_names_populated(self, dataset):
        engineer = FeatureEngineer()
        engineer.fit_transform(dataset.copy())
        assert len(engineer.feature_names) > 0

    def test_scale_reduces_range(self, dataset):
        """After StandardScaler, most feature values should be near [-3, 3]."""
        engineer = FeatureEngineer(scale=True)
        X, _ = engineer.fit_transform(dataset.copy())
        # Means should be near 0
        assert abs(X.mean().mean()) < 1.0

    def test_monthly_payment_burden_positive(self, dataset):
        # The payment burden is the affordability measure the label uses.
        engineer = FeatureEngineer(scale=False)
        X, _ = engineer.fit_transform(dataset.copy())
        assert (X["monthly_payment_burden"] >= 0).all()

    def test_delinquency_recency_is_capped_where_its_effect_ends(self, dataset):
        """The label's recency risk is zero from 24 months on; the feature is
        capped there, with "never delinquent" (-1) at the cap."""
        engineer = FeatureEngineer(scale=False)
        engineer.fit_transform(dataset.copy())
        rows = dataset.iloc[[0, 0, 0, 0, 0]].copy()
        rows["num_delinquencies"] = [1, 1, 1, 1, 0]
        rows["months_since_delinquency"] = [6, 24, 60, 119, -1]
        X, _ = engineer.transform(rows)
        assert X["months_since_delinquency"].tolist() == [6, 24, 24, 24, 24]


class TestMissingTrainedColumns:
    """transform(impute_missing=True) on a payload that omits columns the model was
    TRAINED on.

    The omitted column is debt_to_income: a trained, label-driving column that feeds
    no derived feature and whose 0.0 fill (no debt at all) would be a plausible-sounding
    false claim. tests/test_required_library_inputs.py pins the default, a refusal.
    """

    def _fit(self, dataset):
        engineer = FeatureEngineer(scale=False, impute_missing=True)
        engineer.fit_transform(dataset.copy())
        return engineer

    def test_missing_column_is_imputed_with_training_median_not_zero(self, dataset):
        """A dropped trained column must arrive as the TRAINING median, not 0.0."""
        engineer = self._fit(dataset)
        train_median = float(dataset["debt_to_income"].median())
        assert train_median > 0.0, "fixture sanity: the median must differ from a 0.0 fill"

        row = dataset.iloc[[0]].drop(columns=["debt_to_income"])
        X, _ = engineer.transform(row)

        assert "debt_to_income" in X.columns
        got = float(X["debt_to_income"].iloc[0])
        assert got == pytest.approx(train_median, rel=1e-9), (
            f"missing debt_to_income imputed as {got}, expected training median {train_median}"
        )
        assert got != 0.0, "a trained-but-absent column must not be zero-filled"

    def test_missing_column_matches_explicit_median_payload(self, dataset):
        """Omitting a column == sending the training median for it, feature-for-feature."""
        engineer = self._fit(dataset)
        train_median = float(dataset["debt_to_income"].median())

        row = dataset.iloc[[0]]
        omitted, _ = engineer.transform(row.drop(columns=["debt_to_income"]))
        explicit, _ = engineer.transform(row.assign(debt_to_income=train_median))

        pd.testing.assert_frame_equal(omitted, explicit)

    def test_net_cashflow_absent_is_derived_from_inflow_minus_outflow(self, dataset):
        """monthly_net_cashflow is reconstructed exactly, never imputed or zeroed."""
        engineer = self._fit(dataset)
        row = dataset.iloc[[0]]
        expected = float(row["monthly_inflow"].iloc[0] - row["monthly_outflow"].iloc[0])
        # Guard against a vacuous pass: the identity must be distinguishable
        # both from 0.0 and from the training median.
        assert expected != 0.0
        assert expected != pytest.approx(float(dataset["monthly_net_cashflow"].median()))

        X, _ = engineer.transform(row.drop(columns=["monthly_net_cashflow"]))
        assert float(X["monthly_net_cashflow"].iloc[0]) == pytest.approx(expected)

    def test_a_present_net_cashflow_never_overrides_the_derivation(self, dataset):
        """A value already in the frame never wins: there is one path, so a redundant
        field cannot flip a decision."""
        engineer = self._fit(dataset)
        row = dataset.iloc[[0]]
        derived = float(row["monthly_inflow"].iloc[0] - row["monthly_outflow"].iloc[0])
        X, _ = engineer.transform(row.assign(monthly_net_cashflow=1234.5))
        assert float(X["monthly_net_cashflow"].iloc[0]) == pytest.approx(derived)
    def test_no_nan_survives_a_missing_column(self, dataset):
        """Whatever the caller omits, the model never sees NaN."""
        engineer = self._fit(dataset)
        row = dataset.iloc[[0]].drop(
            columns=["debt_to_income", "monthly_net_cashflow", "savings_rate"]
        )
        X, _ = engineer.transform(row)
        assert not X.isna().to_numpy().any()
        assert list(X.columns) == engineer.feature_names

    def test_api_payload_shape_scores_like_the_full_row(self, dataset):
        """End-to-end guard in score space.

        Drops a trained column and asserts the score equals the score for an
        explicit training-median payload, and is NOT the score a 0.0 fill produces.
        The API refuses a missing input, so this guards the library path, transform().
        """
        from creditlens.models.scorer import XGBoostScorer

        engineer = FeatureEngineer(scale=True, impute_missing=True)
        X_train, y_train = engineer.fit_transform(dataset.copy())
        model = XGBoostScorer(n_estimators=50)
        model.fit(X_train, y_train)

        def scores(frame):
            X, _ = engineer.transform(frame)
            return np.round((1 - model.predict_proba(X)[:, 1]) * 1000)

        rows = dataset.iloc[:50]
        median = float(dataset["debt_to_income"].median())
        omitted = scores(rows.drop(columns=["debt_to_income"]))
        np.testing.assert_array_equal(omitted, scores(rows.assign(debt_to_income=median)))
        # Non-vacuity: a 0.0 fill scores this batch differently.
        assert (omitted != scores(rows.assign(debt_to_income=0.0))).any()
