# Monthly expense reports: category totals, MoM drift, Z-score anomaly detection, health score.
from __future__ import annotations

import logging
import math
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from database import get_db
from models import AnomalyRecord, CategoryORM, ReportORM, ReportSummary, TransactionORM

logger = logging.getLogger(__name__)


class ExpenseReporter:
    """
    Generates monthly expense reports with budget metrics, drift analysis,
    and anomaly detection from SQLite data.

    Example::

        reporter = ExpenseReporter()
        summary = reporter.generate_monthly_report(2024, 1)
        print(summary.budget_health_score)

    Args:
        zscore_threshold: Standard deviations above mean to flag as an anomaly.
                          Default is 2.0 (flags the top ~2.3% of amounts).
    """

    def __init__(self, zscore_threshold: float = 2.0) -> None:
        self._zscore_threshold = zscore_threshold

    # ─────────────────────────────────────────────────────────────
    # Public Interface
    # ─────────────────────────────────────────────────────────────

    def generate_monthly_report(self, year: int, month: int) -> ReportSummary:
        """
        Generate (or regenerate) the full expense report for a given month.

        The report is persisted/overwritten in the reports table after generation
        so it can be quickly re-fetched without recomputation.

        Args:
            year:  4-digit year (2000–2100).
            month: Month number 1–12.

        Returns:
            ReportSummary with all metrics populated.
        """
        period = f"{year:04d}-{month:02d}"
        logger.info("Generating monthly report for %s", period)

        with get_db() as db:
            by_category = self._spending_by_category(db, year, month)
            total_spend, total_income = self._split_totals(by_category)

            prior_year, prior_month = self._prior_month(year, month)
            prior_by_category = self._spending_by_category(db, prior_year, prior_month)
            monthly_drift = self._drift(by_category, prior_by_category)

            anomalies = self._detect_anomalies(db, year, month)

            txn_count = self._count_transactions(db, year, month)
            cat_count = self._count_categorized(db, year, month)

            health_score = self._health_score(
                by_category=by_category,
                monthly_drift=monthly_drift,
                anomaly_count=len(anomalies),
                transaction_count=txn_count,
                categorized_count=cat_count,
            )

            summary = ReportSummary(
                period=period,
                total_spend=total_spend,
                total_income=total_income,
                net=total_income - total_spend,
                by_category=by_category,
                monthly_drift=monthly_drift,
                anomalies=anomalies,
                budget_health_score=health_score,
                transaction_count=txn_count,
                categorized_count=cat_count,
            )

        self._cache_report(summary)
        return summary

    def get_anomalies(
        self,
        year: int | None = None,
        month: int | None = None,
        limit: int = 50,
    ) -> list[AnomalyRecord]:
        """
        Return AI-flagged anomalous transactions, optionally filtered by period.

        Args:
            year:  Optional year filter.
            month: Optional month filter (only applies when year is also set).
            limit: Maximum records to return (max 200).

        Returns:
            List of AnomalyRecord sorted by date descending.
        """
        with get_db() as db:
            query = (
                db.query(TransactionORM, CategoryORM)
                .join(CategoryORM, TransactionORM.id == CategoryORM.transaction_id)
                .filter(CategoryORM.is_anomaly == True)  # noqa: E712
            )
            if year is not None:
                query = query.filter(extract("year", TransactionORM.date) == year)
            if month is not None:
                query = query.filter(extract("month", TransactionORM.date) == month)

            rows = (
                query.order_by(TransactionORM.date.desc())
                .limit(min(limit, 200))
                .all()
            )

        return [
            AnomalyRecord(
                transaction_id=txn.id,
                date=txn.date,
                description=txn.description,
                amount=Decimal(str(txn.amount)),
                category=cat.category,
                anomaly_reason=cat.anomaly_reason or "Flagged by AI categorizer",
                z_score=None,
            )
            for txn, cat in rows
        ]

    def get_available_periods(self) -> list[str]:
        """
        Return a sorted list of all months (YYYY-MM) that contain transactions.

        Useful for populating a period selector in the front-end.
        """
        with get_db() as db:
            rows = (
                db.query(func.strftime("%Y-%m", TransactionORM.date).label("period"))
                .group_by("period")
                .order_by("period")
                .all()
            )
        return [r.period for r in rows if r.period]

    # ─────────────────────────────────────────────────────────────
    # Aggregation Queries
    # ─────────────────────────────────────────────────────────────

    def _spending_by_category(
        self, db: Session, year: int, month: int
    ) -> dict[str, Decimal]:
        """
        Return the sum of transaction amounts per category for the given period.
        Only includes transactions that have been categorized.
        """
        rows = (
            db.query(
                CategoryORM.category,
                func.sum(TransactionORM.amount).label("total"),
            )
            .join(TransactionORM, CategoryORM.transaction_id == TransactionORM.id)
            .filter(
                extract("year", TransactionORM.date) == year,
                extract("month", TransactionORM.date) == month,
            )
            .group_by(CategoryORM.category)
            .all()
        )
        return {r.category: Decimal(str(r.total or "0")) for r in rows}

    def _count_transactions(self, db: Session, year: int, month: int) -> int:
        """Total transaction count for the period (categorized or not)."""
        return (
            db.query(func.count(TransactionORM.id))
            .filter(
                extract("year", TransactionORM.date) == year,
                extract("month", TransactionORM.date) == month,
            )
            .scalar()
            or 0
        )

    def _count_categorized(self, db: Session, year: int, month: int) -> int:
        """Count of transactions with an associated CategoryORM record."""
        return (
            db.query(func.count(TransactionORM.id))
            .join(CategoryORM, TransactionORM.id == CategoryORM.transaction_id)
            .filter(
                extract("year", TransactionORM.date) == year,
                extract("month", TransactionORM.date) == month,
            )
            .scalar()
            or 0
        )

    # ─────────────────────────────────────────────────────────────
    # Metric Calculations
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_totals(
        by_category: dict[str, Decimal],
    ) -> tuple[Decimal, Decimal]:
        """
        Split aggregated category totals into total_spend and total_income.

        Income category amounts are treated as positive inflows.
        Expense category amounts may be positive (database convention: raw amount).
        """
        income = by_category.get("Income", Decimal("0"))
        spend = sum(
            v for k, v in by_category.items() if k != "Income" and v > Decimal("0")
        )
        return spend, income

    @staticmethod
    def _drift(
        current: dict[str, Decimal],
        prior: dict[str, Decimal],
    ) -> dict[str, float]:
        """
        Compute month-over-month percentage change per category.

        Formula:  drift = (current - prior) / |prior| × 100

        Edge cases:
          - prior == 0 and current > 0  → 100.0 (new category this month)
          - prior == 0 and current == 0 → 0.0
          - category only in prior      → -100.0 (dropped to zero)
        """
        all_cats = set(current) | set(prior)
        drift: dict[str, float] = {}

        for cat in all_cats:
            cur = float(current.get(cat, Decimal("0")))
            prv = float(prior.get(cat, Decimal("0")))

            if prv == 0.0:
                drift[cat] = 100.0 if cur > 0.0 else 0.0
            else:
                drift[cat] = round((cur - prv) / abs(prv) * 100.0, 2)

        return drift

    def _detect_anomalies(
        self, db: Session, year: int, month: int
    ) -> list[AnomalyRecord]:
        """
        Identify anomalous transactions using two complementary methods:

        1. **AI-flagged**: Any transaction where CategoryORM.is_anomaly == True.
        2. **Statistical Z-score**: Transactions whose |amount| exceeds
           mean + (zscore_threshold × std) for their category across ALL history.
           Requires ≥ 3 historical data points to be meaningful.

        Results from both methods are merged and de-duplicated by transaction_id.
        """
        anomalies: list[AnomalyRecord] = []
        seen_ids: set[str] = set()

        # ── Method 1: AI-flagged ─────────────────────────────────────
        ai_rows = (
            db.query(TransactionORM, CategoryORM)
            .join(CategoryORM, TransactionORM.id == CategoryORM.transaction_id)
            .filter(
                CategoryORM.is_anomaly == True,  # noqa: E712
                extract("year", TransactionORM.date) == year,
                extract("month", TransactionORM.date) == month,
            )
            .all()
        )

        for txn, cat in ai_rows:
            anomalies.append(
                AnomalyRecord(
                    transaction_id=txn.id,
                    date=txn.date,
                    description=txn.description,
                    amount=Decimal(str(txn.amount)),
                    category=cat.category,
                    anomaly_reason=cat.anomaly_reason or "Flagged by AI categorizer",
                    z_score=None,
                )
            )
            seen_ids.add(txn.id)

        # ── Method 2: Statistical Z-score ─────────────────────────────
        # Get distinct categories present in the target period
        period_categories: list[str] = [
            row[0]
            for row in (
                db.query(CategoryORM.category)
                .join(TransactionORM, CategoryORM.transaction_id == TransactionORM.id)
                .filter(
                    extract("year", TransactionORM.date) == year,
                    extract("month", TransactionORM.date) == month,
                )
                .distinct()
                .all()
            )
        ]

        for category in period_categories:
            # Fetch all historical absolute amounts for this category
            hist_amounts: list[float] = [
                float(row[0])
                for row in (
                    db.query(func.abs(TransactionORM.amount))
                    .join(CategoryORM, TransactionORM.id == CategoryORM.transaction_id)
                    .filter(CategoryORM.category == category)
                    .all()
                )
                if row[0] is not None
            ]

            if len(hist_amounts) < 3:
                logger.debug(
                    "Skipping Z-score for category '%s': only %d data points.",
                    category, len(hist_amounts),
                )
                continue

            mean = sum(hist_amounts) / len(hist_amounts)
            variance = sum((a - mean) ** 2 for a in hist_amounts) / len(hist_amounts)
            std = math.sqrt(variance)

            if std == 0.0:
                continue  # All amounts are identical — no meaningful variance

            # Check current period transactions for this category
            period_rows = (
                db.query(TransactionORM, CategoryORM)
                .join(CategoryORM, TransactionORM.id == CategoryORM.transaction_id)
                .filter(
                    CategoryORM.category == category,
                    extract("year", TransactionORM.date) == year,
                    extract("month", TransactionORM.date) == month,
                )
                .all()
            )

            for txn, cat in period_rows:
                if txn.id in seen_ids:
                    continue  # Already flagged by AI method

                abs_amount = abs(float(txn.amount))
                z_score = (abs_amount - mean) / std

                if abs(z_score) > self._zscore_threshold:
                    anomalies.append(
                        AnomalyRecord(
                            transaction_id=txn.id,
                            date=txn.date,
                            description=txn.description,
                            amount=Decimal(str(txn.amount)),
                            category=cat.category,
                            anomaly_reason=(
                                f"Amount {abs_amount:,.2f} is {abs(z_score):.1f}σ from "
                                f"the {category} category mean of {mean:,.2f}"
                            ),
                            z_score=round(z_score, 3),
                        )
                    )
                    seen_ids.add(txn.id)

        logger.info(
            "Anomaly detection complete for %04d-%02d: %d anomalies found.",
            year, month, len(anomalies),
        )
        return anomalies

    def _health_score(
        self,
        by_category: dict[str, Decimal],
        monthly_drift: dict[str, float],
        anomaly_count: int,
        transaction_count: int,
        categorized_count: int,
    ) -> float:
        """
        Compute a Budget Health Score in [0, 100].

        Component breakdown (25 pts each):

        1. **Diversity** — number of active categories / 8, capped at 25 pts.
           Encourages balanced spending across categories.

        2. **Drift** — starts at 25 pts, deducted based on average |drift|.
           avg_drift / 4.0 deducted, clamped to [0, 25].
           A 100% average drift → 0 pts; 0% drift → 25 pts.

        3. **Anomaly rate** — starts at 25 pts.
           anomaly_rate = anomaly_count / transaction_count.
           Deducted as: 25 × (1 − anomaly_rate × 10), clamped to [0, 25].
           10% anomaly rate → 0 pts.

        4. **Categorization coverage** — (categorized / total) × 25.
           100% coverage → 25 pts; 0% → 0 pts.
        """
        score = 0.0

        # 1. Category diversity
        active = len([v for v in by_category.values() if v > Decimal("0")])
        score += min(active / 8.0, 1.0) * 25.0

        # 2. Drift penalty
        if monthly_drift:
            avg_abs_drift = sum(abs(v) for v in monthly_drift.values()) / len(monthly_drift)
            score += max(0.0, 25.0 - avg_abs_drift / 4.0)
        else:
            score += 12.5  # No prior-month data → neutral

        # 3. Anomaly penalty
        if transaction_count > 0:
            rate = anomaly_count / transaction_count
            score += max(0.0, 25.0 * (1.0 - rate * 10.0))
        else:
            score += 25.0  # No transactions → no anomalies possible

        # 4. Categorization coverage
        if transaction_count > 0:
            score += (categorized_count / transaction_count) * 25.0

        return round(min(max(score, 0.0), 100.0), 2)

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _prior_month(year: int, month: int) -> tuple[int, int]:
        """Return (year, month) of the month immediately preceding the input."""
        if month == 1:
            return year - 1, 12
        return year, month - 1

    def _cache_report(self, summary: ReportSummary) -> None:
        """
        Persist (or overwrite) the ReportSummary as a JSON blob in the reports table.
        Serializes Decimal via model_dump_json (Pydantic handles it automatically).
        """
        report_json = summary.model_dump_json()

        with get_db() as db:
            existing = db.query(ReportORM).filter_by(period=summary.period).first()
            now = datetime.utcnow()

            if existing:
                existing.total_spend = float(summary.total_spend)
                existing.budget_health_score = float(summary.budget_health_score)
                existing.report_data = report_json
                existing.generated_at = now
            else:
                db.add(
                    ReportORM(
                        id=str(uuid.uuid4()),
                        period=summary.period,
                        total_spend=float(summary.total_spend),
                        budget_health_score=float(summary.budget_health_score),
                        report_data=report_json,
                        generated_at=now,
                    )
                )

        logger.info("Report for %s cached to database.", summary.period)
