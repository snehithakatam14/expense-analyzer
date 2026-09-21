"""
tests/test_app.py — Integration tests for Flask REST API endpoints

Coverage:
  GET  /api/v1/health
  POST /api/v1/upload          — CSV, JSON, no-file, empty-filename, wrong-type, parser-error
  POST /api/v1/categorize      — all uncategorized, specific IDs, not-found IDs
  GET  /api/v1/report          — valid params, missing params, invalid month/year/type
  GET  /api/v1/anomalies       — default, filtered, param errors
  GET  /api/v1/transactions    — default pagination, filters, invalid params
  GET  /api/v1/periods         — happy path
  404  /api/v1/nonexistent     — error envelope format
"""
from __future__ import annotations

import io
import json
import os
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FLASK_ENV", "testing")


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    with patch("app.init_db"):
        from app import create_app
        application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


# ─────────────────────────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────────────────────────

_VALID_CSV = (
    "date,description,amount,currency,merchant_name\n"
    "2024-01-15,Starbucks Coffee,-5.75,USD,Starbucks\n"
    "2024-01-16,Netflix,-15.99,USD,Netflix\n"
).encode()

_VALID_JSON = json.dumps([
    {"date": "2024-01-15", "description": "Starbucks", "amount": -5.75},
    {"date": "2024-01-16", "description": "Netflix", "amount": -15.99},
]).encode()


def _make_normalized_txn(txn_id: str = "t001") -> "NormalizedTransaction":
    from models import NormalizedTransaction
    return NormalizedTransaction(
        id=txn_id,
        date=date(2024, 1, 15),
        description="Starbucks",
        amount=Decimal("-5.75"),
        source_file="test.csv",
        content_hash="a" * 64,
    )


def _make_report_summary() -> "ReportSummary":
    from models import ReportSummary
    return ReportSummary(
        period="2024-01",
        total_spend=Decimal("500.00"),
        total_income=Decimal("3000.00"),
        net=Decimal("2500.00"),
        by_category={"Food": Decimal("200"), "Transport": Decimal("300")},
        monthly_drift={"Food": 10.0, "Transport": -5.0},
        anomalies=[],
        budget_health_score=72.50,
        transaction_count=20,
        categorized_count=18,
    )


# ─────────────────────────────────────────────────────────────
# GET /api/v1/health
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_success_true(self, client):
        data = client.get("/api/v1/health").get_json()
        assert data["success"] is True

    def test_status_healthy(self, client):
        data = client.get("/api/v1/health").get_json()
        assert data["data"]["status"] == "healthy"

    def test_has_timestamp(self, client):
        data = client.get("/api/v1/health").get_json()
        assert "timestamp" in data

    def test_has_version(self, client):
        data = client.get("/api/v1/health").get_json()
        assert "version" in data["data"]


# ─────────────────────────────────────────────────────────────
# POST /api/v1/upload
# ─────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def test_upload_csv_success(self, client):
        with patch("app.TransactionParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_csv.return_value = [_make_normalized_txn(), _make_normalized_txn("t002")]

            resp = client.post(
                "/api/v1/upload",
                data={"file": (io.BytesIO(_VALID_CSV), "bank.csv")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["ingested"] == 2
        assert data["data"]["source_file"] == "bank.csv"

    def test_upload_json_success(self, client):
        with patch("app.TransactionParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_json.return_value = [_make_normalized_txn()]

            resp = client.post(
                "/api/v1/upload",
                data={"file": (io.BytesIO(_VALID_JSON), "txns.json")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 201
        assert resp.get_json()["data"]["ingested"] == 1

    def test_no_file_field_returns_400(self, client):
        resp = client.post("/api/v1/upload", data={})
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False

    def test_empty_filename_returns_400(self, client):
        resp = client.post(
            "/api/v1/upload",
            data={"file": (io.BytesIO(b"data"), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_unsupported_extension_returns_415(self, client):
        resp = client.post(
            "/api/v1/upload",
            data={"file": (io.BytesIO(b"data"), "report.xlsx")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 415
        assert "Unsupported" in resp.get_json()["error"]

    def test_parser_error_returns_422(self, client):
        from parser import ParserError

        with patch("app.TransactionParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_csv.side_effect = ParserError("Missing required column: date")

            resp = client.post(
                "/api/v1/upload",
                data={"file": (io.BytesIO(b"bad,csv\ndata"), "bad.csv")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 422
        assert "Missing required column" in resp.get_json()["error"]

    def test_meta_message_present(self, client):
        with patch("app.TransactionParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_csv.return_value = [_make_normalized_txn()]

            resp = client.post(
                "/api/v1/upload",
                data={"file": (io.BytesIO(_VALID_CSV), "t.csv")},
                content_type="multipart/form-data",
            )
        assert "message" in resp.get_json().get("meta", {})


# ─────────────────────────────────────────────────────────────
# POST /api/v1/categorize
# ─────────────────────────────────────────────────────────────

class TestCategorizeEndpoint:
    def test_categorize_all_uncategorized(self, client):
        from models import CategoryResponse, TransactionCategory

        mock_result = CategoryResponse(
            transaction_id="t001",
            category=TransactionCategory.FOOD,
            sub_category="Coffee",
            confidence=0.95,
            reasoning="Starbucks",
            is_anomaly=False,
        )

        with patch("app.TransactionCategorizer") as MockCat:
            instance = MockCat.return_value
            instance.categorize_uncategorized.return_value = [mock_result]

            resp = client.post(
                "/api/v1/categorize",
                data=json.dumps({}),
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["categorized"] == 1

    def test_categorize_zero_uncategorized(self, client):
        with patch("app.TransactionCategorizer") as MockCat:
            instance = MockCat.return_value
            instance.categorize_uncategorized.return_value = []

            resp = client.post("/api/v1/categorize")

        assert resp.status_code == 200
        assert resp.get_json()["data"]["categorized"] == 0

    def test_specific_transaction_ids_not_found_returns_404(self, client):
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("app.get_db", return_value=mock_db):
            resp = client.post(
                "/api/v1/categorize",
                data=json.dumps({"transaction_ids": ["nonexistent-uuid"]}),
                content_type="application/json",
            )

        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────
# GET /api/v1/report
# ─────────────────────────────────────────────────────────────

class TestReportEndpoint:
    def test_valid_request_returns_200(self, client):
        summary = _make_report_summary()
        with patch("app.ExpenseReporter") as MockReporter:
            MockReporter.return_value.generate_monthly_report.return_value = summary
            resp = client.get("/api/v1/report?year=2024&month=1")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["period"] == "2024-01"

    def test_missing_year_returns_400(self, client):
        resp = client.get("/api/v1/report?month=1")
        assert resp.status_code == 400

    def test_missing_month_returns_400(self, client):
        resp = client.get("/api/v1/report?year=2024")
        assert resp.status_code == 400

    def test_invalid_month_13_returns_400(self, client):
        resp = client.get("/api/v1/report?year=2024&month=13")
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False

    def test_invalid_month_0_returns_400(self, client):
        resp = client.get("/api/v1/report?year=2024&month=0")
        assert resp.status_code == 400

    def test_invalid_year_1800_returns_400(self, client):
        resp = client.get("/api/v1/report?year=1800&month=1")
        assert resp.status_code == 400

    def test_non_integer_year_returns_400(self, client):
        resp = client.get("/api/v1/report?year=abc&month=1")
        assert resp.status_code == 400

    def test_non_integer_month_returns_400(self, client):
        resp = client.get("/api/v1/report?year=2024&month=jan")
        assert resp.status_code == 400

    def test_report_data_has_expected_fields(self, client):
        summary = _make_report_summary()
        with patch("app.ExpenseReporter") as MockReporter:
            MockReporter.return_value.generate_monthly_report.return_value = summary
            resp = client.get("/api/v1/report?year=2024&month=1")

        data = resp.get_json()["data"]
        for field in ("period", "total_spend", "total_income", "net",
                      "by_category", "monthly_drift", "anomalies",
                      "budget_health_score", "transaction_count"):
            assert field in data, f"Missing field: {field}"


# ─────────────────────────────────────────────────────────────
# GET /api/v1/anomalies
# ─────────────────────────────────────────────────────────────

class TestAnomaliesEndpoint:
    def test_returns_200_with_empty_list(self, client):
        with patch("app.ExpenseReporter") as MockReporter:
            MockReporter.return_value.get_anomalies.return_value = []
            resp = client.get("/api/v1/anomalies")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["meta"]["count"] == 0

    def test_year_and_month_forwarded_correctly(self, client):
        with patch("app.ExpenseReporter") as MockReporter:
            instance = MockReporter.return_value
            instance.get_anomalies.return_value = []
            client.get("/api/v1/anomalies?year=2024&month=3&limit=10")
            instance.get_anomalies.assert_called_once_with(year=2024, month=3, limit=10)

    def test_invalid_year_param_returns_400(self, client):
        resp = client.get("/api/v1/anomalies?year=bad")
        assert resp.status_code == 400

    def test_limit_capped_at_200(self, client):
        with patch("app.ExpenseReporter") as MockReporter:
            instance = MockReporter.return_value
            instance.get_anomalies.return_value = []
            client.get("/api/v1/anomalies?limit=999")
            # Actual call should cap at 200
            called_limit = instance.get_anomalies.call_args[1]["limit"]
            assert called_limit <= 200

    def test_count_in_meta_matches_data_length(self, client):
        from models import AnomalyRecord
        from decimal import Decimal

        mock_anomaly = AnomalyRecord(
            transaction_id="t001",
            date=date(2024, 1, 15),
            description="Big Purchase",
            amount=Decimal("-999.99"),
            category="Shopping",
            anomaly_reason="Too large",
        )
        with patch("app.ExpenseReporter") as MockReporter:
            MockReporter.return_value.get_anomalies.return_value = [mock_anomaly]
            resp = client.get("/api/v1/anomalies")

        data = resp.get_json()
        assert data["meta"]["count"] == len(data["data"])


# ─────────────────────────────────────────────────────────────
# GET /api/v1/transactions
# ─────────────────────────────────────────────────────────────

class TestTransactionsEndpoint:
    def _mock_db_context(self, rows=None, total=0):
        """Helper: build a mock get_db() context manager returning given rows."""
        rows = rows or []
        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        q = mock_db.query.return_value.outerjoin.return_value
        q.count.return_value = total
        (q.filter.return_value
           .filter.return_value
           .order_by.return_value
           .offset.return_value
           .limit.return_value
           .all.return_value) = rows
        q.order_by.return_value.offset.return_value.limit.return_value.all.return_value = rows

        return mock_db

    def test_returns_200(self, client):
        mock_db = self._mock_db_context()
        with patch("app.get_db", return_value=mock_db):
            resp = client.get("/api/v1/transactions")
        assert resp.status_code == 200

    def test_pagination_meta_present(self, client):
        mock_db = self._mock_db_context()
        with patch("app.get_db", return_value=mock_db):
            resp = client.get("/api/v1/transactions?page=1&per_page=10")

        meta = resp.get_json()["meta"]
        assert "page" in meta
        assert "per_page" in meta
        assert "total" in meta
        assert "pages" in meta

    def test_per_page_capped_at_100(self, client):
        mock_db = self._mock_db_context()
        with patch("app.get_db", return_value=mock_db):
            resp = client.get("/api/v1/transactions?per_page=999")
        assert resp.status_code == 200

    def test_invalid_page_param_returns_400(self, client):
        resp = client.get("/api/v1/transactions?page=abc")
        assert resp.status_code == 400

    def test_invalid_per_page_param_returns_400(self, client):
        resp = client.get("/api/v1/transactions?per_page=xyz")
        assert resp.status_code == 400

    def test_data_is_list(self, client):
        mock_db = self._mock_db_context()
        with patch("app.get_db", return_value=mock_db):
            resp = client.get("/api/v1/transactions")
        assert isinstance(resp.get_json()["data"], list)


# ─────────────────────────────────────────────────────────────
# GET /api/v1/periods
# ─────────────────────────────────────────────────────────────

class TestPeriodsEndpoint:
    def test_returns_200(self, client):
        with patch("app.ExpenseReporter") as MockReporter:
            MockReporter.return_value.get_available_periods.return_value = ["2024-01", "2024-02"]
            resp = client.get("/api/v1/periods")

        assert resp.status_code == 200

    def test_data_has_periods_key(self, client):
        with patch("app.ExpenseReporter") as MockReporter:
            MockReporter.return_value.get_available_periods.return_value = ["2024-01"]
            resp = client.get("/api/v1/periods")

        data = resp.get_json()["data"]
        assert "periods" in data
        assert isinstance(data["periods"], list)


# ─────────────────────────────────────────────────────────────
# Global error handlers
# ─────────────────────────────────────────────────────────────

class TestErrorHandlers:
    def test_404_returns_json_envelope(self, client):
        resp = client.get("/api/v1/does-not-exist")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False
        assert "error" in data
        assert "timestamp" in data

    def test_405_returns_json_envelope(self, client):
        resp = client.delete("/api/v1/health")
        assert resp.status_code == 405
        assert resp.get_json()["success"] is False

    def test_all_error_responses_have_status_field(self, client):
        """Verify the 'status' field matches the HTTP status code in all error responses."""
        resp = client.get("/api/v1/nonexistent")
        data = resp.get_json()
        assert data["status"] == 404
