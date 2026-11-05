"""Tests for the synthetic credit dataset generator."""

import numpy as np

from creditlens.data.generator import DatasetConfig, generate_credit_dataset


class TestDelinquencyRecency:
    def test_recency_exists_exactly_for_applicants_with_a_delinquency(self):
        """months_since_delinquency == -1 ("never delinquent") must be exactly the
        applicants with zero delinquencies, so nobody without one is charged recency
        risk in the default label."""
        df = generate_credit_dataset(DatasetConfig(n_samples=5000, seed=3))
        never = df["months_since_delinquency"] == -1
        clean = df["num_delinquencies"] == 0
        assert clean.any() and (~clean).any(), "fixture must contain both kinds"
        assert (never == clean).all()

    def test_recency_of_delinquent_applicants_is_a_valid_month_count(self):
        df = generate_credit_dataset(DatasetConfig(n_samples=5000, seed=3))
        months = df.loc[df["num_delinquencies"] > 0, "months_since_delinquency"]
        assert months.between(0, 119).all()
        # Spread over the whole window, not collapsed onto a constant.
        assert months.nunique() > 50

    def test_generation_is_deterministic_for_a_seed(self):
        a = generate_credit_dataset(DatasetConfig(n_samples=300, seed=11))
        b = generate_credit_dataset(DatasetConfig(n_samples=300, seed=11))
        assert a.equals(b)
        assert np.array_equal(a["default"].values, b["default"].values)


class TestGroupSettings:
    """What the default data does and does not contain,
    and the opt-in measurement bias that gives the audit something to catch."""

    def test_the_documented_outcome_gap_is_the_measured_one(self):
        """The generator's comment states the gap group_c_bias realizes: the default
        10,000 rows have 5.6 points (the population has about 7.6)."""
        from pathlib import Path

        import creditlens.data.generator as generator

        df = generate_credit_dataset(DatasetConfig())
        rate = df.groupby("demographic_group")["default"].mean()
        text = " ".join(Path(generator.__file__).read_text(encoding="utf-8").split()).replace(" # ", " ")
        assert (f"{(rate['C'] - rate['A']) * 100:.1f} point gap in default rate "
                f"(group C {rate['C']:.1%}, group A {rate['A']:.1%}") in text

    def test_default_outcome_gap_is_carried_by_no_feature(self):
        df = generate_credit_dataset(DatasetConfig(n_samples=20000, seed=4))
        a, c = df[df["demographic_group"] == "A"], df[df["demographic_group"] == "C"]
        # group_c_bias: group C defaults more often ...
        assert c["default"].mean() - a["default"].mean() > 0.05
        # ... at the SAME features: every numeric feature has the same mean in
        # both groups to within 0.1 SD (about 5 standard errors at this size).
        for col in df.columns.drop(["default", "demographic_group", "loan_purpose"]):
            gap = abs(c[col].mean() - a[col].mean()) / df[col].std()
            assert gap < 0.1, (col, gap)

    def test_fico_shift_off_by_default_leaves_the_dataset_byte_identical(self):
        default = generate_credit_dataset(DatasetConfig(n_samples=3000, seed=42))
        explicit = generate_credit_dataset(DatasetConfig(n_samples=3000, seed=42, group_c_fico_shift=0.0))
        assert DatasetConfig().group_c_fico_shift == 0.0
        assert default.equals(explicit)

    def test_fico_shift_understates_group_c_reported_score_but_not_its_risk(self):
        true = generate_credit_dataset(DatasetConfig(n_samples=3000, seed=42))
        shifted = generate_credit_dataset(DatasetConfig(n_samples=3000, seed=42, group_c_fico_shift=80))
        is_c = true["demographic_group"] == "C"
        assert (shifted["demographic_group"] == true["demographic_group"]).all()
        # Only group C's REPORTED score moves, by 80 points (floored at 300).
        np.testing.assert_array_equal(shifted.loc[~is_c, "fico_score"], true.loc[~is_c, "fico_score"])
        np.testing.assert_array_equal(
            shifted.loc[is_c, "fico_score"], np.clip(true.loc[is_c, "fico_score"] - 80, 300, 850))
        # Default risk follows the TRUE score: the labels do not change at all.
        np.testing.assert_array_equal(shifted["default"], true["default"])
        # The lender prices on the score it sees.
        assert (shifted.loc[is_c, "interest_rate"] >= true.loc[is_c, "interest_rate"]).all()
        assert shifted.loc[is_c, "interest_rate"].mean() > true.loc[is_c, "interest_rate"].mean() + 0.01
        others = shifted.columns.drop(["fico_score", "interest_rate"])
        assert shifted[others].equals(true[others])


def test_the_docs_say_what_the_generator_is():
    """The docs called this data "realistic". It is a stated ground truth with mostly
    independent features, which one line of pandas shows, so the docs now say that and
    give these numbers."""
    import re
    from pathlib import Path

    from creditlens.data import generator

    df = generate_credit_dataset(DatasetConfig())
    fact = (f"FICO's correlation is {df.fico_score.corr(df.num_delinquencies):.3f} with delinquencies, "
            f"{df.fico_score.corr(df.credit_utilization):.3f} with utilization and "
            f"{df.fico_score.corr(df.annual_income):.3f} with income, and banking tenure exceeds credit age on "
            f"{(df.banking_tenure_months > df.credit_age_months).mean() * 100:.1f}% of rows")
    root = Path(__file__).resolve().parents[1]
    texts = [(root / "README.md").read_text(encoding="utf-8"), (root / "docs" / "architecture.md").read_text(encoding="utf-8"),
             generator.__doc__]
    for text in texts:
        assert fact in re.sub(r"\s+", " ", text), text[:40]
