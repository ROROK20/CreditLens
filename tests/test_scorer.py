"""Tests for credit scoring models."""

import numpy as np
import pandas as pd
import pytest

from creditlens.data.generator import generate_credit_dataset, DatasetConfig, train_test_split
from creditlens.features.engineering import FeatureEngineer
from creditlens.models.scorer import LogisticScorer, XGBoostScorer, NeuralNetScorer, evaluate_model


@pytest.fixture(scope="module")
def dataset():
    config = DatasetConfig(n_samples=2000, seed=42)
    df = generate_credit_dataset(config)
    train_df, test_df = train_test_split(df, test_size=0.25, seed=42)
    return train_df, test_df


@pytest.fixture(scope="module")
def engineered_data(dataset):
    train_df, test_df = dataset
    engineer = FeatureEngineer(scale=True)
    X_train, y_train = engineer.fit_transform(train_df)
    X_test, y_test = engineer.transform(test_df)
    return X_train, y_train, X_test, y_test, engineer


@pytest.fixture(scope="module")
def sample_data(engineered_data):
    X_train, y_train, X_test, y_test, _ = engineered_data
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_xgb(engineered_data):
    X_train, y_train, *_ = engineered_data
    model = XGBoostScorer(n_estimators=50)  # fewer trees for faster tests
    model.fit(X_train, y_train)
    return model


class TestXGBoostScorer:
    def test_fit_returns_self(self, engineered_data):
        X_train, y_train, *_ = engineered_data
        model = XGBoostScorer(n_estimators=30)
        result = model.fit(X_train, y_train)
        assert result is model

    def test_predict_proba_shape(self, trained_xgb, engineered_data):
        _, _, X_test, _, _ = engineered_data
        proba = trained_xgb.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_predict_proba_sums_to_one(self, trained_xgb, engineered_data):
        _, _, X_test, _, _ = engineered_data
        proba = trained_xgb.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_values_in_zero_one(self, trained_xgb, engineered_data):
        _, _, X_test, _, _ = engineered_data
        proba = trained_xgb.predict_proba(X_test)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_score_range(self, trained_xgb, engineered_data):
        _, _, X_test, _, _ = engineered_data
        scores = trained_xgb.predict_score(X_test)
        assert (scores >= 0).all()
        assert (scores <= 1000).all()

    def test_predict_binary_output(self, trained_xgb, engineered_data):
        _, _, X_test, _, _ = engineered_data
        preds = trained_xgb.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_auc_above_random(self, trained_xgb, engineered_data):
        from sklearn.metrics import roc_auc_score
        _, _, X_test, y_test, _ = engineered_data
        proba = trained_xgb.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc > 0.60, f"AUC {auc:.3f} not much better than random"

    def test_feature_importances_non_empty(self, trained_xgb):
        importances = trained_xgb.feature_importances()
        assert len(importances) > 0
        assert importances.sum() > 0

    def test_feature_importances_untrained_returns_empty_series(self):
        # An untrained model has no importances: return an empty Series (the
        # documented contract), not a crash. The failure path logs rather than
        # swallowing silently — see scorer.feature_importances().
        importances = XGBoostScorer().feature_importances()
        assert isinstance(importances, pd.Series)
        assert importances.empty

    def test_untrained_raises(self, engineered_data):
        _, _, X_test, _, _ = engineered_data
        model = XGBoostScorer()
        with pytest.raises(RuntimeError):
            model.predict_proba(X_test)


class TestNeuralNetScorer:
    def test_fit_and_predict(self, engineered_data):
        X_train, y_train, X_test, _, _ = engineered_data
        model = NeuralNetScorer(hidden_layers=(64, 32), max_iter=50)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[1] == 2
        assert (proba >= 0).all()

    def test_score_range(self, engineered_data):
        X_train, y_train, X_test, _, _ = engineered_data
        model = NeuralNetScorer(hidden_layers=(32,), max_iter=30)
        model.fit(X_train, y_train)
        scores = model.predict_score(X_test)
        assert (scores >= 0).all() and (scores <= 1000).all()


class TestModelEvaluation:
    def test_evaluation_returns_object(self, trained_xgb, engineered_data):
        _, _, X_test, y_test, _ = engineered_data
        eval_ = evaluate_model(trained_xgb, X_test, y_test, "test")
        assert eval_.auc_roc > 0
        assert eval_.auc_pr > 0
        assert 0 <= eval_.gini_coefficient <= 1

    def test_gini_equals_2auc_minus_1(self, trained_xgb, engineered_data):
        _, _, X_test, y_test, _ = engineered_data
        eval_ = evaluate_model(trained_xgb, X_test, y_test, "test")
        assert abs(eval_.gini_coefficient - (2 * eval_.auc_roc - 1)) < 1e-6

    def test_confusion_matrix_shape(self, trained_xgb, engineered_data):
        _, _, X_test, y_test, _ = engineered_data
        eval_ = evaluate_model(trained_xgb, X_test, y_test, "test")
        cm = eval_.confusion_matrix
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestSampleWeights:
    """The fairness report recommends reweighing (FairnessAuditor.mitigation_reweighting),
    so the advice must be followable: fit() accepts weights and uses them."""

    def test_no_weights_trains_exactly_as_before(self, engineered_data, trained_xgb):
        X_train, y_train, X_test, *_ = engineered_data
        again = XGBoostScorer(n_estimators=50).fit(X_train, y_train, sample_weight=None)
        np.testing.assert_array_equal(again.predict_proba(X_test), trained_xgb.predict_proba(X_test))

    def test_uniform_weights_equal_no_weights(self, engineered_data, trained_xgb):
        X_train, y_train, X_test, *_ = engineered_data
        uniform = XGBoostScorer(n_estimators=50).fit(X_train, y_train, sample_weight=np.ones(len(X_train)))
        np.testing.assert_allclose(uniform.predict_proba(X_test), trained_xgb.predict_proba(X_test), atol=1e-12)

    def test_reweighing_weights_change_the_model(self, dataset, engineered_data, trained_xgb):
        from creditlens.fairness.metrics import FairnessAuditor

        train_df, _ = dataset
        X_train, y_train, X_test, *_ = engineered_data
        weights = FairnessAuditor().mitigation_reweighting(X_train, y_train, train_df["demographic_group"])
        assert not np.allclose(weights, 1.0)  # the default data has a group-C outcome gap
        reweighed = XGBoostScorer(n_estimators=50).fit(X_train, y_train, sample_weight=weights)
        moved = np.abs(reweighed.predict_proba(X_test)[:, 1] - trained_xgb.predict_proba(X_test)[:, 1])
        assert moved.max() > 1e-3

    @pytest.mark.parametrize("bad", [np.ones(3), np.full(1500, np.nan), -np.ones(1500)])
    def test_invalid_weights_are_refused(self, engineered_data, bad):
        X_train, y_train, *_ = engineered_data
        assert len(X_train) == 1500
        with pytest.raises(ValueError, match="sample_weight"):
            XGBoostScorer(n_estimators=10).fit(X_train, y_train, sample_weight=bad)


class TestLogisticScorer:
    """The model the API serves."""

    def test_has_the_scorer_interface(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = LogisticScorer().fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert ((model.predict_score(X_test) >= 0) & (model.predict_score(X_test) <= 1000)).all()
        assert evaluate_model(model, X_test, y_test).auc_roc > 0.7
        assert not model.feature_importances().empty

    def test_is_deterministic(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        a = LogisticScorer().fit(X_train, y_train).predict_proba(X_test)
        b = LogisticScorer().fit(X_train, y_train).predict_proba(X_test)
        np.testing.assert_array_equal(a, b)

    def test_an_untrained_model_refuses_to_predict(self, sample_data):
        with pytest.raises(RuntimeError, match="not trained"):
            LogisticScorer().predict_proba(sample_data[1])

    def test_recovers_the_weights_of_a_logistic_label(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"fico_score": rng.normal(680, 80, 60_000), "nsf_count": rng.poisson(1.0, 60_000)})
        eta = -0.02 * (X["fico_score"] - 680) + 0.5 * X["nsf_count"] - 2.0
        y = pd.Series((rng.uniform(size=len(X)) < 1 / (1 + np.exp(-eta))).astype(int))
        model = LogisticScorer().fit(X, y)
        assert model.coefficients()["fico_score"] == pytest.approx(-0.02, abs=0.001)
        assert model.coefficients()["nsf_count"] == pytest.approx(0.5, abs=0.03)
        assert model.intercept() + 680 * model.coefficients()["fico_score"] == pytest.approx(-2.0, abs=0.06)

    def test_a_wrong_sign_estimate_is_exactly_zero_when_constrained(self):
        """overdraft_days may only raise risk; here more of it goes with less risk."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"overdraft_days": rng.poisson(2.0, 20_000).astype(float),
                          "fico_score": rng.normal(680, 80, 20_000)})
        eta = -0.5 * X["overdraft_days"] - 0.01 * (X["fico_score"] - 680)
        y = pd.Series((rng.uniform(size=len(X)) < 1 / (1 + np.exp(-eta))).astype(int))
        assert LogisticScorer().fit(X, y).coefficients()["overdraft_days"] == 0.0
        assert LogisticScorer(monotone=False).fit(X, y).coefficients()["overdraft_days"] < -0.3

    def test_sample_weight(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        plain = LogisticScorer().fit(X_train, y_train).predict_proba(X_test)
        uniform = LogisticScorer().fit(X_train, y_train, sample_weight=np.ones(len(X_train))).predict_proba(X_test)
        np.testing.assert_allclose(plain, uniform, atol=1e-7)
        heavy = np.where(y_train.to_numpy() == 1, 5.0, 1.0)
        reweighed = LogisticScorer().fit(X_train, y_train, sample_weight=heavy).predict_proba(X_test)
        assert reweighed[:, 1].mean() > plain[:, 1].mean()
        for bad in (np.ones(3), -np.ones(len(X_train))):
            with pytest.raises(ValueError, match="sample_weight"):
                LogisticScorer().fit(X_train, y_train, sample_weight=bad)

    def test_save_and_load_round_trip(self, sample_data, tmp_path):
        X_train, X_test, y_train, _ = sample_data
        model = LogisticScorer().fit(X_train, y_train)
        path = tmp_path / "model.joblib"
        model.save(path)
        np.testing.assert_array_equal(LogisticScorer().load(path).predict_proba(X_test), model.predict_proba(X_test))
