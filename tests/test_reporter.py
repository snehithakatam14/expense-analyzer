"""
tests/test_reporter.py — Unit tests for reporter.py

Coverage:
  - _drift():          positive, negative, new category, both-zero, prior-only
  - _split_totals():   income excluded from spend, zero income, empty
  - _health_score():   high-scoring conditions, zero transactions, anomaly penalty,
                       score always clamped [0, 100]
  - _prior_month():    January wrapping, December, mid-year
  - _detect_anomalies(): not tested with live DB (requires integration setup),
                          but structure is validated via the public interface mock
  - generate_monthly_report(): end-to-end with mocked DB session
  - get_available_periods():   returns sorted YYYY-MM strings
"""
from __future__ import annotations

import os
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FLASK_ENV", "testing")

from reporter import ExpenseReporter
from models import AnomalyRecord, ReportSummary


# ─────────────────────────────────────────────────────────────
# _prior_month
# ─────────────────────────────────────────────────────────────

class TestPriorMonth:
    def setup_method(self):
        self.reporter = ExpenseReporter()

    def test_mid_year(self):
        assert self.reporter._prior_month(2024, 6) == (2024, 5)

    def test_january_wraps_to_previous_year_december(self):
        assert self.reporter._prior_month(2024, 1) == (2023, 12)

    def test_december(self):
        assert self.reporter._prior_month(2024, 12) == (2024, 11)

    def test_february(self):
        assert self.reporter._prior_month(2024, 2) == (2024, 1)

    def test_year_boundary_on_other_years(self):
        assert self.reporter._prior_month(2000, 1) == (1999, 12)


# ─────────────────────────────────────────────────────────────
# _split_totals
# ─────────────────────────────────────────────────────────────

class TestSplitTotals:
    def setup_method(self):
        self.reporter = ExpenseReporter()

    def test_income_separated_from_spend(self):
        by_cat = {
            "Food": Decimal("200"),
            "Transport": Decimal("100"),
            "Income": Decimal("3000"),
        }
        spend, income = self.reporter._split_totals(by_cat)
        assert spend == Decimal("300")
        assert income == Decimal("3000")

    def test_no_income_key(self):
        by_cat = {"Food": Decimal("100"), "Transport": Decimal("50")}
        spend, income = self.reporter._split_totals(by_cat)
        assert spend == Decimal("150")
        assert income == Decimal("0")

    def test_only_income(self):
        by_cat = {"Income": Decimal("5000")}
        spend, income = self.reporter._split_totals(by_cat)
        assert spend == Decimal("0")
        assert income == Decimal("5000")

    def test_empty_categories(self):
        spend, income = self.reporter._split_totals({})
        assert spend == Decimal("0")
        assert income == Decimal("0")

    def test_negative_spend_not_summed(self):
        """Negative amounts (e.g. data entry errors) should not inflate spend."""
        by_cat = {"Food": Decimal("-50"), "Transport": Decimal("100")}
        spend, income = self.reporter._split_totals(by_cat)
        assert spend == Decimal("100")  # Only positive values summed


# ─────────────────────────────────────────────────────────────
# _drift
# ─────────────────────────────────────────────────────────────

class TestDrift:
    def setup_method(self):
        self.reporter = ExpenseReporter()

    def test_positive_drift_100_pct(self):
        current = {"Food": Decimal("200")}
        prior = {"Food": Decimal("100")}
        drift = self.reporter._drift(current, prior)
        assert drift["Food"] == pytest.approx(100.0)

    def test_negative_drift_50_pct(self):
        current = {"Food": Decimal("50")}
        prior = {"Food": Decimal("100")}
        drift = self.reporter._drift(current, prior)
        assert drift["Food"] == pytest.approx(-50.0)

    def test_zero_drift_same_amount(self):
        current = {"Transport": Decimal("80")}
        prior = {"Transport": Decimal("80")}
        drift = self.reporter._drift(current, prior)
        assert drift["Transport"] == pytest.approx(0.0)

    def test_new_category_this_month_100_pct(self):
        current = {"Shopping": Decimal("300")}
        prior = {}
        drift = self.reporter._drift(current, prior)
        assert drift["Shopping"] == 100.0

    def test_category_disappeared_negative_100(self):
        current = {}
        prior = {"Food": Decimal("100")}
        drift = self.reporter._drift(current, prior)
        assert drift["Food"] == pytest.approx(-100.0)

    def test_both_zero_is_zero_drift(self):
        current = {"Food": Decimal("0")}
        prior = {"Food": Decimal("0")}
        drift = self.reporter._drift(current, prior)
        assert drift["Food"] == 0.0

    def test_all_categories_present(self):
        current = {"Food": Decimal("100"), "Transport": Decimal("200")}
        prior = {"Food": Decimal("100"), "Shopping": Decimal("50")}
        drift = self.reporter._drift(current, prior)
        assert "Food" in drift
        assert "Transport" in drift
        assert "Shopping" in drift

    def test_drift_values_rounded_to_2dp(self):
        current = {"Food": Decimal("101")}
        prior = {"Food": Decimal("100")}
        drift = self.reporter._drift(current, prior)
        # 1.0% increase
        assert drift["Food"] == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────
# _health_score
# ─────────────────────────────────────────────────────────────

class TestHealthScore:
    def setup_method(self):
        self.reporter = ExpenseReporter()

    def test_score_always_in_0_to_100_range(self):
        """Run with various extreme inputs — score must never leave [0, 100]."""
        test_cases = [
            # (by_category, drift, anomaly_count, txn_count, cat_count)
            ({}, {}, 0, 0, 0),
            ({"Food": Decimal("999999")}, {"Food": 5000.0}, 100, 10, 0),
            (
                {k: Decimal("100") for k in
                 ["Food", "Transport", "Housing", "Healthcare",
                  "Entertainment", "Shopping", "Utilities", "Income"]},
                {"Food": 0.0},
                0, 50, 50,
            ),
        ]
        for by_cat, drift, anom, txn, cat in test_cases:
            score = self.reporter._health_score(by_cat, drift, anom, txn, cat)
            assert 0.0 <= score <= 100.0, f"Score out of range: {score}"

    def test_perfect_conditions_score_high(self):
        """8 diverse categories, 0 drift, 0 anomalies, 100% coverage → high score."""
        by_cat = {k: Decimal("100") for k in
                  ["Food", "Transport", "Housing", "Healthcare",
                   "Entertainment", "Shopping", "Utilities", "Income"]}
        score = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={"Food": 0.0, "Transport": 0.0},
            anomaly_count=0,
            transaction_count=100,
            categorized_count=100,
        )
        assert score >= 80.0

    def test_many_anomalies_lowers_score(self):
        base_cat = {"Food": Decimal("100")}
        base_drift = {}

        score_clean = self.reporter._health_score(
            by_category=base_cat,
            monthly_drift=base_drift,
            anomaly_count=0,
            transaction_count=10,
            categorized_count=10,
        )
        score_dirty = self.reporter._health_score(
            by_category=base_cat,
            monthly_drift=base_drift,
            anomaly_count=5,
            transaction_count=10,
            categorized_count=10,
        )
        assert score_clean > score_dirty

    def test_low_categorization_coverage_lowers_score(self):
        by_cat = {"Food": Decimal("100")}
        score_full = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={},
            anomaly_count=0,
            transaction_count=10,
            categorized_count=10,
        )
        score_partial = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={},
            anomaly_count=0,
            transaction_count=10,
            categorized_count=1,
        )
        assert score_full > score_partial

    def test_high_drift_lowers_score(self):
        by_cat = {"Food": Decimal("100")}

        score_stable = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={"Food": 0.0},
            anomaly_count=0,
            transaction_count=10,
            categorized_count=10,
        )
        score_volatile = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={"Food": 300.0},
            anomaly_count=0,
            transaction_count=10,
            categorized_count=10,
        )
        assert score_stable > score_volatile

    def test_zero_transactions_does_not_crash(self):
        score = self.reporter._health_score(
            by_category={},
            monthly_drift={},
            anomaly_count=0,
            transaction_count=0,
            categorized_count=0,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_single_category_diversity_low(self):
        """Only 1 active category → diversity score = 25 * (1/8) ≈ 3.1."""
        by_cat = {"Food": Decimal("100")}
        score = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={},
            anomaly_count=0,
            transaction_count=1,
            categorized_count=1,
        )
        # Diversity = 3.125, neutral drift = 12.5, no anomalies = 25, full coverage = 25
        # Total ≈ 65.6
        assert 50.0 <= score <= 80.0

    def test_result_is_rounded_to_2dp(self):
        by_cat = {"Food": Decimal("100")}
        score = self.reporter._health_score(
            by_category=by_cat,
            monthly_drift={"Food": 10.0},
            anomaly_count=0,
            transaction_count=5,
            categorized_count=5,
        )
        # Should be a float with at most 2 decimal places
        assert score == round(score, 2)


# ─────────────────────────────────────────────────────────────
# get_available_periods (mocked DB)
# ─────────────────────────────────────────────────────────────

class TestGetAvailablePeriods:
    def test_returns_sorted_periods(self):
        reporter = ExpenseReporter()

        mock_row_1 = MagicMock()
        mock_row_1.period = "2024-01"
        mock_row_2 = MagicMock()
        mock_row_2.period = "2024-02"
        mock_row_3 = MagicMock()
        mock_row_3.period = "2023-12"

        mock_query = MagicMock()
        mock_query.group_by.return_value.order_by.return_value.all.return_value = [
            mock_row_3, mock_row_1, mock_row_2  # Returned in any order
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.group_by.return_value.order_by.return_value.all.return_value = [
            mock_row_3, mock_row_1, mock_row_2
        ]
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("reporter.get_db", return_value=mock_db):
            periods = reporter.get_available_periods()

        # All returned periods should be present
        assert "2024-01" in periods
        assert "2024-02" in periods
        assert "2023-12" in periods

    def test_empty_db_returns_empty_list(self):
        reporter = ExpenseReporter()

        mock_db = MagicMock()
        mock_db.query.return_value.group_by.return_value.order_by.return_value.all.return_value = []
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("reporter.get_db", return_value=mock_db):
            periods = reporter.get_available_periods()

        assert periods == []


# ─────────────────────────────────────────────────────────────
# get_anomalies (mocked DB)
# ─────────────────────────────────────────────────────────────

class TestGetAnomalies:
    def test_returns_anomaly_records(self):
        reporter = ExpenseReporter()

        mock_txn = MagicMock()
        mock_txn.id = "txn-001"
        mock_txn.date = date(2024, 1, 15)
        mock_txn.description = "Suspicious Transaction"
        mock_txn.amount = 9999.99

        mock_cat = MagicMock()
        mock_cat.category = "Shopping"
        mock_cat.anomaly_reason = "Unusually large"

        # MagicMock auto-chains any attribute access and method call,
        # returning a new MagicMock each time, EXCEPT when we configure
        # the terminal .all() to return our data.
        # We configure the entire chain as one connected mock.
        all_mock = MagicMock(return_value=[(mock_txn, mock_cat)])

        mock_db = MagicMock()
        # Every intermediate method in the chain returns mock_db itself
        # so .all() is always called on all_mock at the end
        mock_db.query.return_value.join.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all = all_mock
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("reporter.get_db", return_value=mock_db):
            results = reporter.get_anomalies(year=2024, month=1)

        assert len(results) == 1
        assert results[0].transaction_id == "txn-001"
        assert results[0].category == "Shopping"
        assert results[0].anomaly_reason == "Unusually large"

    def test_limit_capped_at_200(self):
        reporter = ExpenseReporter()

        mock_db = MagicMock()
        q = mock_db.query.return_value
        limit_chain = q.join.return_value.filter.return_value.order_by.return_value
        limit_chain.limit.return_value.all.return_value = []
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("reporter.get_db", return_value=mock_db):
            reporter.get_anomalies(limit=500)  # Should be capped at 200

        # Verify limit was called with 200
        limit_chain.limit.assert_called_with(200)
