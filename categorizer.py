# Sends transactions to OpenAI in batches, enforces a Pydantic schema, retries on transient errors.
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Sequence

import openai
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings
from database import get_db
from models import (
    BatchCategorizationResponse,
    CategoryORM,
    CategoryResponse,
    NormalizedTransaction,
    TransactionCategorization,
    TransactionCategory,
    TransactionORM,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert financial transaction categorizer trained on millions of bank records.

Your task: analyze each transaction's description, merchant, amount, and date to assign
the most accurate financial category.

Available categories:
  Food         — Restaurants, cafes, grocery stores, food delivery, supermarkets
  Transport    — Uber, Lyft, taxis, gas stations, parking, public transit, tolls
  Housing      — Rent, mortgage, HOA fees, home maintenance, property tax
  Healthcare   — Hospitals, pharmacies, doctors, dentists, health insurance, medical devices
  Entertainment— Movies, concerts, streaming (Netflix/Spotify/Disney+), gaming, sports
  Shopping     — Clothing, electronics, Amazon, department stores, online retail
  Utilities    — Standalone bills: internet, phone, electricity, water, gas
  Income       — Salary deposits, freelance payments, refunds, bank interest, transfers IN
  Travel       — Hotels, flights, Airbnb, travel agencies, foreign currency transactions
  Education    — Tuition, textbooks, online courses, certifications, school supplies
  Other        — Anything that does not clearly fit the above categories

Categorization rules:
  1. Use the FULL description + merchant name + amount for context.
  2. High confidence (>0.85) = you are certain; Low confidence (<0.60) = genuinely ambiguous.
  3. Mark is_anomaly=true ONLY if the amount is unusually large for the category OR
     the description contains suspicious patterns (duplicate charges, round-dollar fraud indicators).
  4. sub_category must be specific: e.g., "Coffee & Cafes", not just "Cafes".
  5. Return EXACTLY one result per transaction in the results array, in the same order.
  6. Never hallucinate categories — use Other when uncertain.
"""


# ─────────────────────────────────────────────────────────────
# Retry Decorator (module-level so it's re-usable)
# ─────────────────────────────────────────────────────────────

_RETRYABLE = (APITimeoutError, RateLimitError, APIConnectionError)

_retry_policy = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,  # After all retries, re-raise the original exception
)


# ─────────────────────────────────────────────────────────────
# TransactionCategorizer
# ─────────────────────────────────────────────────────────────


class TransactionCategorizer:
    """
    Categorizes financial transactions using the OpenAI Structured Outputs API.

    The OpenAI API enforces the ``BatchCategorizationResponse`` Pydantic schema,
    so the response is always a fully-typed Python object — no parsing fragility.

    Example::

        categorizer = TransactionCategorizer()
        results = categorizer.categorize(transactions)
        for r in results:
            print(r.category, r.confidence)

    Args:
        api_key:    Override the OpenAI API key from settings.
        model:      Override the model from settings.
        batch_size: Number of transactions per API call (default from settings).
        persist:    Write CategoryORM records to SQLite (default True).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        persist: bool = True,
    ) -> None:
        self._client = openai.OpenAI(
            api_key=api_key or settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
        )
        self._model = model or settings.openai_model
        self._batch_size = batch_size or settings.categorizer_batch_size
        self._persist = persist

    # ── Public Interface ─────────────────────────────────────────────────

    def categorize(
        self,
        transactions: Sequence[NormalizedTransaction],
    ) -> list[CategoryResponse]:
        """
        Categorize a sequence of transactions, processing them in batches.

        Guarantees one CategoryResponse per input transaction, even on partial
        API failures (failed batches fall back to TransactionCategory.OTHER).

        Args:
            transactions: Transactions to categorize.

        Returns:
            List of CategoryResponse, one per input transaction, in input order.
        """
        if not transactions:
            logger.info("categorize() called with empty list — no-op.")
            return []

        txn_list = list(transactions)
        batches = self._create_batches(txn_list)

        logger.info(
            "Categorizing %d transaction(s) across %d batch(es) [model=%s, batch_size=%d]",
            len(txn_list), len(batches), self._model, self._batch_size,
        )

        all_results: list[CategoryResponse] = []

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(
                "Processing batch %d/%d (%d transactions)", batch_num, len(batches), len(batch)
            )
            try:
                batch_results = self._categorize_batch(batch)
                all_results.extend(batch_results)
            except RetryError as exc:
                logger.error(
                    "Batch %d/%d: all %d retry attempts exhausted — using fallback. Error: %s",
                    batch_num, len(batches), settings.max_retries, exc,
                )
                all_results.extend(self._fallback_results(batch))
            except APIStatusError as exc:
                logger.error(
                    "Batch %d/%d: unrecoverable API error (HTTP %d) — using fallback. Error: %s",
                    batch_num, len(batches), exc.status_code, exc.message,
                )
                all_results.extend(self._fallback_results(batch))

            # Polite inter-batch delay to avoid burst rate limiting
            if batch_num < len(batches):
                time.sleep(0.5)

        if self._persist:
            self._persist_results(all_results)

        return all_results

    def categorize_uncategorized(self) -> list[CategoryResponse]:
        """
        Fetch all DB transactions without a CategoryORM record and categorize them.

        Ideal for running as a background job or scheduled task after bulk uploads.

        Returns:
            List of CategoryResponse for newly categorized transactions.
        """
        # Build NormalizedTransaction list INSIDE the session so ORM
        # attributes are accessed while the connection is still open.
        with get_db() as db:
            rows = (
                db.query(TransactionORM)
                .outerjoin(CategoryORM, TransactionORM.id == CategoryORM.transaction_id)
                .filter(CategoryORM.id.is_(None))
                .all()
            )

            if not rows:
                logger.info("All transactions are already categorized.")
                return []

            logger.info("Found %d uncategorized transaction(s) to process.", len(rows))

            transactions = [
                NormalizedTransaction(
                    id=orm.id,
                    date=orm.date,
                    description=orm.description,
                    amount=orm.amount,
                    currency=orm.currency,
                    merchant_name=orm.merchant_name or "",
                    source_file=orm.source_file,
                    ingested_at=orm.ingested_at,
                    content_hash=orm.content_hash,
                )
                for orm in rows
            ]

        return self.categorize(transactions)

    # ── Core Categorization Logic ─────────────────────────────────────────

    def _categorize_batch(
        self, batch: list[NormalizedTransaction]
    ) -> list[CategoryResponse]:
        """
        Send one batch to the OpenAI API and map results back to CategoryResponse.
        Raises on unrecoverable errors; retryable errors handled by _call_openai().
        """
        prompt = self._build_batch_prompt(batch)
        parsed = self._call_openai(prompt, expected_count=len(batch))

        # Align results with input transactions by index
        results: list[CategoryResponse] = []
        for txn, raw in zip(batch, parsed.results):
            results.append(
                CategoryResponse(
                    transaction_id=txn.id,
                    category=raw.category,
                    sub_category=raw.sub_category,
                    confidence=raw.confidence,
                    reasoning=raw.reasoning,
                    is_anomaly=raw.is_anomaly,
                    anomaly_reason=raw.anomaly_reason,
                )
            )

        return results

    @_retry_policy
    def _call_openai(
        self,
        prompt: str,
        expected_count: int,
    ) -> BatchCategorizationResponse:
        """
        Make a single Structured Outputs API call.

        The ``@_retry_policy`` decorator wraps this method with exponential
        backoff retries on transient OpenAI errors.

        Args:
            prompt:         User-side prompt text.
            expected_count: Number of transactions in the batch (for validation).

        Returns:
            BatchCategorizationResponse validated by Pydantic.

        Raises:
            APITimeoutError / RateLimitError / APIConnectionError: retried by decorator.
            APIStatusError: unrecoverable, propagated to caller.
            ValueError: If the API returns a null parsed object (e.g., content filter).
        """
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=BatchCategorizationResponse,
            temperature=0.1,   # Low temperature → deterministic, consistent categorization
            max_tokens=4096,   # Generous ceiling for large batches
        )

        parsed: BatchCategorizationResponse | None = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError(
                "OpenAI returned a null parsed response — likely a content policy refusal. "
                "Check the raw response for details."
            )

        actual = len(parsed.results)
        if actual != expected_count:
            logger.warning(
                "OpenAI returned %d results for %d transactions. "
                "Padding/truncating to match.",
                actual, expected_count,
            )
            # Pad missing with fallback items
            while len(parsed.results) < expected_count:
                parsed.results.append(
                    TransactionCategorization(
                        category=TransactionCategory.OTHER,
                        sub_category="Uncategorized",
                        confidence=0.0,
                        reasoning="Missing from API response.",
                        is_anomaly=False,
                        anomaly_reason=None,
                    )
                )
            # Truncate extras
            parsed.results = parsed.results[:expected_count]

        return parsed

    # ── Prompt Building ──────────────────────────────────────────────────

    def _build_batch_prompt(self, transactions: list[NormalizedTransaction]) -> str:
        """
        Construct a structured user-side prompt for a batch of transactions.

        Each transaction is formatted as a numbered line with date, amount,
        currency, description, and optionally merchant name.
        """
        n = len(transactions)
        lines: list[str] = [
            f"Categorize the following {n} financial transaction(s).",
            f"Return EXACTLY {n} result(s) in the 'results' array, in the same order as listed.",
            "",
            "Transactions:",
        ]

        for i, txn in enumerate(transactions, start=1):
            amount_str = f"{txn.amount:,.2f} {txn.currency}"
            merchant = f" | Merchant: {txn.merchant_name}" if txn.merchant_name else ""
            lines.append(
                f"  {i}. Date: {txn.date}  Amount: {amount_str}  "
                f"Description: {txn.description}{merchant}"
            )

        return "\n".join(lines)

    # ── Utility ──────────────────────────────────────────────────────────

    def _create_batches(
        self, transactions: list[NormalizedTransaction]
    ) -> list[list[NormalizedTransaction]]:
        """Split ``transactions`` into sub-lists of at most ``_batch_size`` items."""
        return [
            transactions[i : i + self._batch_size]
            for i in range(0, len(transactions), self._batch_size)
        ]

    @staticmethod
    def _fallback_results(
        transactions: list[NormalizedTransaction],
    ) -> list[CategoryResponse]:
        """
        Return a list of 'Other / Uncategorized' CategoryResponse objects.
        Used when all retry attempts fail so the pipeline never stalls.
        """
        return [
            CategoryResponse(
                transaction_id=txn.id,
                category=TransactionCategory.OTHER,
                sub_category="Uncategorized",
                confidence=0.0,
                reasoning="Categorization failed due to API error — fallback applied.",
                is_anomaly=False,
                anomaly_reason=None,
            )
            for txn in transactions
        ]

    # ── Persistence ──────────────────────────────────────────────────────

    def _persist_results(self, results: list[CategoryResponse]) -> None:
        """
        Upsert CategoryORM records for every CategoryResponse.

        Uses update-in-place for existing records so re-categorization is safe
        and idempotent — calling categorize() twice produces the same DB state.
        """
        with get_db() as db:
            transaction_ids = [r.transaction_id for r in results]

            # Bulk-load existing records in one query
            existing_map: dict[str, CategoryORM] = {
                row.transaction_id: row
                for row in db.query(CategoryORM)
                .filter(CategoryORM.transaction_id.in_(transaction_ids))
                .all()
            }

            for result in results:
                now = datetime.utcnow()
                if result.transaction_id in existing_map:
                    # Update existing record
                    orm = existing_map[result.transaction_id]
                    orm.category = result.category.value
                    orm.sub_category = result.sub_category
                    orm.confidence = float(result.confidence)
                    orm.reasoning = result.reasoning
                    orm.is_anomaly = result.is_anomaly
                    orm.anomaly_reason = result.anomaly_reason
                    orm.categorized_at = now
                else:
                    # Insert new record
                    db.add(
                        CategoryORM(
                            id=str(uuid.uuid4()),
                            transaction_id=result.transaction_id,
                            category=result.category.value,
                            sub_category=result.sub_category,
                            confidence=float(result.confidence),
                            reasoning=result.reasoning,
                            is_anomaly=result.is_anomaly,
                            anomaly_reason=result.anomaly_reason,
                            categorized_at=now,
                        )
                    )

        logger.info(
            "Persisted %d category record(s) to database (%d updated, %d inserted).",
            len(results),
            len([r for r in results if r.transaction_id in existing_map]),
            len([r for r in results if r.transaction_id not in existing_map]),
        )
