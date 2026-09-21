"""
tests/test_categorizer.py — Unit tests for categorizer.py

Coverage:
  - Empty input → immediate empty return
  - Single transaction happy path (mocked OpenAI)
  - Batch splitting logic (7 transactions, batch_size=5 → 2 API calls)
  - Anomaly detection propagated from API response
  - Retry on APITimeoutError (succeeds on second attempt)
  - Graceful fallback (OTHER / confidence=0.0) when all retries exhausted
  - API prompt structure verification
  - Transaction ID mapping across batch
  - _create_batches() size correctness
  - _fallback_results() structure
  - Persistence upsert logic (mocked)
"""
from __future__ import annotations

import os
import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FLASK_ENV", "testing")

from categorizer import TransactionCategorizer
from models import (
    BatchCategorizationResponse,
    CategoryResponse,
    NormalizedTransaction,
    TransactionCategorization,
    TransactionCategory,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_txn(
    txn_id: str = "txn-001",
    description: str = "Starbucks Coffee",
    amount: float = -5.75,
    txn_date: date = date(2024, 1, 15),
    merchant_name: str = "Starbucks",
) -> NormalizedTransaction:
    return NormalizedTransaction(
        id=txn_id,
        date=txn_date,
        description=description,
        amount=Decimal(str(amount)),
        currency="USD",
        merchant_name=merchant_name,
        source_file="test.csv",
        content_hash="a" * 64,
    )


def _make_api_response(
    categorizations: list[tuple[TransactionCategory, str, float, bool]] | None = None,
) -> MagicMock:
    """
    Build a mock OpenAI structured-output response.

    Each tuple in categorizations: (category, sub_category, confidence, is_anomaly)
    Defaults to a single FOOD / Coffee & Cafes / 0.95 / False result.
    """
    if categorizations is None:
        categorizations = [(TransactionCategory.FOOD, "Coffee & Cafes", 0.95, False)]

    results = [
        TransactionCategorization(
            category=cat,
            sub_category=sub,
            confidence=conf,
            reasoning="Test reasoning",
            is_anomaly=anomaly,
            anomaly_reason="Suspicious" if anomaly else None,
        )
        for cat, sub, conf, anomaly in categorizations
    ]

    batch = BatchCategorizationResponse(results=results)

    mock_msg = MagicMock()
    mock_msg.parsed = batch

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    return mock_resp


@pytest.fixture()
def cat() -> TransactionCategorizer:
    """Categorizer with test API key and persistence disabled."""
    return TransactionCategorizer(
        api_key="test-key",
        model="gpt-4o-mini",
        batch_size=5,
        persist=False,
    )


# ─────────────────────────────────────────────────────────────
# Basic behaviour
# ─────────────────────────────────────────────────────────────

class TestCategorizerBasic:
    def test_empty_list_returns_empty(self, cat):
        assert cat.categorize([]) == []

    def test_single_transaction_returns_one_result(self, cat):
        txn = _make_txn()
        mock_resp = _make_api_response()
        with patch.object(cat._client.beta.chat.completions, "parse", return_value=mock_resp):
            results = cat.categorize([txn])
        assert len(results) == 1

    def test_result_has_correct_transaction_id(self, cat):
        txn = _make_txn(txn_id="my-unique-id")
        mock_resp = _make_api_response()
        with patch.object(cat._client.beta.chat.completions, "parse", return_value=mock_resp):
            results = cat.categorize([txn])
        assert results[0].transaction_id == "my-unique-id"

    def test_category_matches_api_response(self, cat):
        txn = _make_txn()
        mock_resp = _make_api_response([(TransactionCategory.FOOD, "Coffee & Cafes", 0.95, False)])
        with patch.object(cat._client.beta.chat.completions, "parse", return_value=mock_resp):
            results = cat.categorize([txn])
        assert results[0].category == TransactionCategory.FOOD
        assert results[0].sub_category == "Coffee & Cafes"
        assert results[0].confidence == pytest.approx(0.95)

    def test_is_anomaly_false_propagated(self, cat):
        txn = _make_txn()
        mock_resp = _make_api_response([(TransactionCategory.FOOD, "Cafe", 0.9, False)])
        with patch.object(cat._client.beta.chat.completions, "parse", return_value=mock_resp):
            results = cat.categorize([txn])
        assert results[0].is_anomaly is False
        assert results[0].anomaly_reason is None

    def test_is_anomaly_true_with_reason(self, cat):
        txn = _make_txn(amount=-9999.99)
        mock_resp = _make_api_response([(TransactionCategory.SHOPPING, "Electronics", 0.8, True)])
        # Patch is_anomaly to True in the parsed model
        mock_resp.choices[0].message.parsed.results[0].anomaly_reason = "Unusually large"
        with patch.object(cat._client.beta.chat.completions, "parse", return_value=mock_resp):
            results = cat.categorize([txn])
        assert results[0].is_anomaly is True
        assert results[0].anomaly_reason == "Unusually large"


# ─────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────

class TestBatchProcessing:
    def test_7_transactions_make_2_api_calls(self, cat):
        """batch_size=5 → ceil(7/5)=2 calls."""
        transactions = [_make_txn(txn_id=f"id-{i}") for i in range(7)]

        def side_effect(prompt: str, expected_count: int) -> BatchCategorizationResponse:
            return BatchCategorizationResponse(results=[
                TransactionCategorization(
                    category=TransactionCategory.OTHER,
                    sub_category="Test",
                    confidence=0.5,
                    reasoning="Test",
                    is_anomaly=False,
                )
            ] * expected_count)

        with patch.object(cat, "_call_openai", side_effect=side_effect) as mock_call:
            results = cat.categorize(transactions)

        assert mock_call.call_count == 2
        assert len(results) == 7

    def test_results_order_matches_input_order(self, cat):
        """Transaction IDs in results must match input order."""
        transactions = [_make_txn(txn_id=f"id-{i}") for i in range(4)]
        categories = [
            TransactionCategory.FOOD,
            TransactionCategory.TRANSPORT,
            TransactionCategory.SHOPPING,
            TransactionCategory.UTILITIES,
        ]

        def side_effect(prompt: str, expected_count: int) -> BatchCategorizationResponse:
            return BatchCategorizationResponse(results=[
                TransactionCategorization(
                    category=categories[i],
                    sub_category="Sub",
                    confidence=0.9,
                    reasoning="Test",
                    is_anomaly=False,
                )
                for i in range(expected_count)
            ])

        with patch.object(cat, "_call_openai", side_effect=side_effect):
            results = cat.categorize(transactions)

        for i, result in enumerate(results):
            assert result.transaction_id == f"id-{i}"
            assert result.category == categories[i]

    def test_exact_batch_size_makes_1_call(self, cat):
        """Exactly batch_size=5 transactions → exactly 1 API call."""
        transactions = [_make_txn(txn_id=f"id-{i}") for i in range(5)]

        def side_effect(prompt, expected_count):
            return BatchCategorizationResponse(results=[
                TransactionCategorization(
                    category=TransactionCategory.FOOD,
                    sub_category="Sub",
                    confidence=0.9,
                    reasoning="Test",
                    is_anomaly=False,
                )
            ] * expected_count)

        with patch.object(cat, "_call_openai", side_effect=side_effect) as mock_call:
            cat.categorize(transactions)
        assert mock_call.call_count == 1


# ─────────────────────────────────────────────────────────────
# _create_batches
# ─────────────────────────────────────────────────────────────

class TestCreateBatches:
    def test_exact_multiple(self, cat):
        txns = [_make_txn(txn_id=str(i)) for i in range(10)]
        batches = cat._create_batches(txns)
        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_remainder(self, cat):
        txns = [_make_txn(txn_id=str(i)) for i in range(7)]
        batches = cat._create_batches(txns)
        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 2

    def test_less_than_batch_size(self, cat):
        txns = [_make_txn(txn_id=str(i)) for i in range(3)]
        batches = cat._create_batches(txns)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_empty_input(self, cat):
        assert cat._create_batches([]) == []


# ─────────────────────────────────────────────────────────────
# Retry logic
# ─────────────────────────────────────────────────────────────

class TestRetryLogic:
    def test_retries_on_timeout_then_succeeds(self, cat):
        """First call times out, second succeeds."""
        from openai import APITimeoutError

        txn = _make_txn()
        mock_resp = _make_api_response()
        call_count = 0

        def mock_parse(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APITimeoutError(request=MagicMock())
            return mock_resp

        with patch.object(cat._client.beta.chat.completions, "parse", side_effect=mock_parse):
            results = cat.categorize([txn])

        assert call_count == 2
        assert len(results) == 1

    def test_fallback_on_all_retries_exhausted(self, cat):
        """All retries fail → fallback to OTHER with confidence=0."""
        from tenacity import RetryError

        txn = _make_txn(txn_id="fallback-txn")

        # Patch _call_openai (which has @_retry_policy) so RetryError
        # fires immediately without tenacity sleep delays
        with patch.object(
            cat,
            "_call_openai",
            side_effect=RetryError(last_attempt=MagicMock()),
        ):
            results = cat.categorize([txn])

        assert len(results) == 1
        assert results[0].category == TransactionCategory.OTHER
        assert results[0].confidence == 0.0
        assert results[0].transaction_id == "fallback-txn"
        # reasoning contains "api error" after lowercasing
        reasoning_lower = results[0].reasoning.lower()
        assert "api error" in reasoning_lower or "failed" in reasoning_lower

    def test_fallback_on_api_status_error(self, cat):
        """Non-retryable HTTP error → fallback immediately."""
        from openai import APIStatusError

        txn = _make_txn(txn_id="status-error-txn")
        mock_response = MagicMock()
        mock_response.status_code = 400

        with patch.object(
            cat,
            "_call_openai",
            side_effect=APIStatusError(
                message="Bad request",
                response=mock_response,
                body={},
            ),
        ):
            results = cat.categorize([txn])

        assert results[0].category == TransactionCategory.OTHER


# ─────────────────────────────────────────────────────────────
# _fallback_results
# ─────────────────────────────────────────────────────────────

class TestFallbackResults:
    def test_fallback_count_matches_input(self, cat):
        txns = [_make_txn(txn_id=f"id-{i}") for i in range(4)]
        results = TransactionCategorizer._fallback_results(txns)
        assert len(results) == 4

    def test_fallback_all_other_category(self, cat):
        txns = [_make_txn()]
        results = TransactionCategorizer._fallback_results(txns)
        assert results[0].category == TransactionCategory.OTHER

    def test_fallback_zero_confidence(self, cat):
        txns = [_make_txn()]
        results = TransactionCategorizer._fallback_results(txns)
        assert results[0].confidence == 0.0

    def test_fallback_not_anomaly(self, cat):
        txns = [_make_txn()]
        results = TransactionCategorizer._fallback_results(txns)
        assert results[0].is_anomaly is False


# ─────────────────────────────────────────────────────────────
# Prompt structure
# ─────────────────────────────────────────────────────────────

class TestPromptBuilding:
    def test_prompt_contains_all_descriptions(self, cat):
        txns = [
            _make_txn(txn_id="1", description="Netflix Subscription"),
            _make_txn(txn_id="2", description="Uber Ride"),
        ]
        prompt = cat._build_batch_prompt(txns)
        assert "Netflix Subscription" in prompt
        assert "Uber Ride" in prompt

    def test_prompt_contains_transaction_count(self, cat):
        txns = [_make_txn(txn_id=str(i)) for i in range(3)]
        prompt = cat._build_batch_prompt(txns)
        assert "3" in prompt

    def test_prompt_contains_date(self, cat):
        txn = _make_txn(txn_date=date(2024, 6, 15))
        prompt = cat._build_batch_prompt([txn])
        assert "2024-06-15" in prompt

    def test_prompt_contains_amount(self, cat):
        txn = _make_txn(amount=-99.99)
        prompt = cat._build_batch_prompt([txn])
        assert "99.99" in prompt

    def test_prompt_contains_merchant_when_present(self, cat):
        txn = _make_txn(merchant_name="Starbucks Corp")
        prompt = cat._build_batch_prompt([txn])
        assert "Starbucks Corp" in prompt

    def test_prompt_numbered_correctly(self, cat):
        txns = [_make_txn(txn_id=str(i), description=f"Txn {i}") for i in range(3)]
        prompt = cat._build_batch_prompt(txns)
        assert "  1." in prompt
        assert "  2." in prompt
        assert "  3." in prompt


# ─────────────────────────────────────────────────────────────
# Persistence (mocked)
# ─────────────────────────────────────────────────────────────

class TestCategorizerPersistence:
    def test_persist_called_when_enabled(self):
        cat = TransactionCategorizer(api_key="key", persist=True)
        txn = _make_txn()

        with patch.object(cat, "_call_openai") as mock_call, \
             patch.object(cat, "_persist_results") as mock_persist:
            mock_call.return_value = BatchCategorizationResponse(results=[
                TransactionCategorization(
                    category=TransactionCategory.FOOD,
                    sub_category="Cafe",
                    confidence=0.9,
                    reasoning="Test",
                    is_anomaly=False,
                )
            ])
            cat.categorize([txn])

        mock_persist.assert_called_once()

    def test_persist_not_called_when_disabled(self):
        cat = TransactionCategorizer(api_key="key", persist=False)
        txn = _make_txn()

        with patch.object(cat, "_call_openai") as mock_call, \
             patch.object(cat, "_persist_results") as mock_persist:
            mock_call.return_value = BatchCategorizationResponse(results=[
                TransactionCategorization(
                    category=TransactionCategory.FOOD,
                    sub_category="Cafe",
                    confidence=0.9,
                    reasoning="Test",
                    is_anomaly=False,
                )
            ])
            cat.categorize([txn])

        mock_persist.assert_not_called()
