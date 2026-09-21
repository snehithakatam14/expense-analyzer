# Flask REST API. All routes are under /api/v1/ via Blueprint.
from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from typing import Any

from flask import Blueprint, Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from categorizer import TransactionCategorizer
from config import get_settings
from database import get_db, init_db
from models import NormalizedTransaction, TransactionORM, CategoryORM
from parser import ParserError, TransactionParser
from reporter import ExpenseReporter

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────
# Blueprint
# ─────────────────────────────────────────────────────────────

api_bp = Blueprint("api", __name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"csv", "json"})


# ─────────────────────────────────────────────────────────────
# Dashboard (root route)
# ─────────────────────────────────────────────────────────────

def _register_root(app: Flask) -> None:
    """Register the / route on the app (not the blueprint) to serve the dashboard."""

    @app.get("/")
    def dashboard():
        """Serve the live analytics dashboard."""
        return render_template("dashboard.html")


# ─────────────────────────────────────────────────────────────
# Response Helpers
# ─────────────────────────────────────────────────────────────


def _error(message: str, status: HTTPStatus) -> tuple[Any, int]:
    """Standardised error envelope."""
    return (
        jsonify(
            {
                "success": False,
                "error": message,
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        ),
        status.value,
    )


def _ok(
    data: Any,
    status: HTTPStatus = HTTPStatus.OK,
    meta: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    """Standardised success envelope."""
    payload: dict[str, Any] = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if meta:
        payload["meta"] = meta
    return jsonify(payload), status.value


def _serialize(obj: Any) -> Any:
    """
    Recursively convert non-JSON-native types to JSON-safe equivalents.
    Decimal → str (preserves precision), date → ISO string.
    """
    from datetime import date as date_type

    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, date_type):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in _ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────


@api_bp.get("/health")
def health():
    """
    GET /api/v1/health

    Liveness probe — confirms the API process is up and the DB is reachable.

    Returns 200 with environment info.  No auth required.
    """
    return _ok(
        {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.flask_env,
            "model": settings.openai_model,
        }
    )


@api_bp.post("/upload")
def upload():
    """
    POST /api/v1/upload

    Upload a CSV or JSON file of bank/credit card transactions.

    Request:
        Content-Type: multipart/form-data
        Field: file  (required) — CSV or JSON file

    Response 201:
        { "data": { "ingested": int, "source_file": str }, "meta": { "message": str } }

    Errors:
        400 — No file, empty filename
        415 — Unsupported file type (only .csv and .json accepted)
        422 — File is parseable format but content is invalid
    """
    if "file" not in request.files:
        return _error(
            "No file part in request. Use field name 'file' with multipart/form-data.",
            HTTPStatus.BAD_REQUEST,
        )

    file = request.files["file"]

    if not file.filename:
        return _error("No file selected (empty filename).", HTTPStatus.BAD_REQUEST)

    if not _allowed_file(file.filename):
        return _error(
            f"Unsupported file type. Accepted extensions: {sorted(_ALLOWED_EXTENSIONS)}",
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        )

    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[1].lower()
    raw_bytes: bytes = file.read()

    parser = TransactionParser(persist=True)

    try:
        if extension == "csv":
            transactions = parser.parse_csv(raw_bytes, source_file=filename)
        else:
            transactions = parser.parse_json(raw_bytes, source_file=filename)
    except ParserError as exc:
        return _error(str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)

    return _ok(
        data={"ingested": len(transactions), "source_file": filename},
        status=HTTPStatus.CREATED,
        meta={"message": f"Successfully parsed {len(transactions)} transaction(s)."},
    )


@api_bp.post("/categorize")
def categorize():
    """
    POST /api/v1/categorize

    Run the AI categorizer on uncategorized transactions.

    Optional JSON body:
        { "transaction_ids": ["uuid1", "uuid2"] }
        — If omitted, ALL uncategorized transactions are processed.

    Response 200:
        { "data": { "categorized": int } }

    Errors:
        404 — Provided transaction_ids not found
        500 — OpenAI API failure after all retries
    """
    body: dict[str, Any] = request.get_json(silent=True) or {}
    transaction_ids: list[str] | None = body.get("transaction_ids")

    cat = TransactionCategorizer(persist=True)

    try:
        if transaction_ids:
            # Categorize specific transactions by ID
            with get_db() as db:
                orm_rows = (
                    db.query(TransactionORM)
                    .filter(TransactionORM.id.in_(transaction_ids))
                    .all()
                )

            if not orm_rows:
                return _error(
                    "No transactions found for the provided IDs.",
                    HTTPStatus.NOT_FOUND,
                )

            transactions = [
                NormalizedTransaction(
                    id=t.id,
                    date=t.date,
                    description=t.description,
                    amount=Decimal(str(t.amount)),
                    currency=t.currency,
                    merchant_name=t.merchant_name or "",
                    source_file=t.source_file,
                    ingested_at=t.ingested_at,
                    content_hash=t.content_hash,
                )
                for t in orm_rows
            ]
            results = cat.categorize(transactions)
        else:
            results = cat.categorize_uncategorized()

    except Exception as exc:
        logger.exception("Categorization endpoint error")
        return _error(
            f"Categorization failed: {exc}",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return _ok(
        data={"categorized": len(results)},
        meta={"message": f"Categorized {len(results)} transaction(s)."},
    )


@api_bp.get("/report")
def report():
    """
    GET /api/v1/report?year=2024&month=1

    Generate (or regenerate) the monthly expense report.

    Query Parameters:
        year  (int, required) — 4-digit year, 2000–2100
        month (int, required) — month number 1–12

    Response 200:
        Full ReportSummary JSON with totals, drift, anomalies, and health score.

    Errors:
        400 — Missing/invalid year or month
        500 — Database or computation error
    """
    raw_year = request.args.get("year")
    raw_month = request.args.get("month")

    if not raw_year or not raw_month:
        return _error(
            "Query parameters 'year' and 'month' are required. "
            "Example: /api/v1/report?year=2024&month=1",
            HTTPStatus.BAD_REQUEST,
        )

    try:
        year = int(raw_year)
        month = int(raw_month)
    except ValueError:
        return _error("'year' and 'month' must be integers.", HTTPStatus.BAD_REQUEST)

    if not (1 <= month <= 12):
        return _error("'month' must be between 1 and 12.", HTTPStatus.BAD_REQUEST)
    if not (2000 <= year <= 2100):
        return _error("'year' must be between 2000 and 2100.", HTTPStatus.BAD_REQUEST)

    try:
        reporter = ExpenseReporter(zscore_threshold=settings.anomaly_zscore_threshold)
        summary = reporter.generate_monthly_report(year, month)
        return _ok(data=_serialize(summary.model_dump()))
    except Exception as exc:
        logger.exception("Report generation failed for %04d-%02d", year, month)
        return _error(f"Report generation failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)


@api_bp.get("/anomalies")
def anomalies():
    """
    GET /api/v1/anomalies?year=2024&month=1&limit=50

    Return AI-flagged anomalous transactions.

    Query Parameters (all optional):
        year  (int) — Filter by year
        month (int) — Filter by month (only used when year is also set)
        limit (int) — Max results, default 50, capped at 200

    Response 200:
        { "data": [AnomalyRecord, ...], "meta": { "count": int } }
    """
    year: int | None = None
    month: int | None = None
    limit = 50

    try:
        if "year" in request.args:
            year = int(request.args["year"])
        if "month" in request.args:
            month = int(request.args["month"])
        if "limit" in request.args:
            limit = min(max(1, int(request.args["limit"])), 200)
    except ValueError:
        return _error("Query parameter values must be integers.", HTTPStatus.BAD_REQUEST)

    try:
        reporter = ExpenseReporter()
        records = reporter.get_anomalies(year=year, month=month, limit=limit)
        serialized = _serialize([r.model_dump() for r in records])
        return _ok(data=serialized, meta={"count": len(records)})
    except Exception as exc:
        logger.exception("Failed to fetch anomalies")
        return _error(f"Failed to fetch anomalies: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)


@api_bp.get("/transactions")
def transactions():
    """
    GET /api/v1/transactions?page=1&per_page=20&year=2024&month=1&category=Food

    Paginated list of transactions with optional filters.

    Query Parameters:
        page      (int) — Page number, default 1
        per_page  (int) — Results per page, default 20, max 100
        year      (int) — Optional year filter
        month     (int) — Optional month filter (requires year)
        category  (str) — Optional category name filter

    Response 200:
        { "data": [TransactionItem, ...], "meta": { page, per_page, total, pages } }

    Each TransactionItem includes an embedded "category" object when available.
    """
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(max(1, int(request.args.get("per_page", 20))), 100)
    except ValueError:
        return _error("'page' and 'per_page' must be integers.", HTTPStatus.BAD_REQUEST)

    year: int | None = None
    month: int | None = None
    category_filter: str | None = request.args.get("category")

    try:
        if "year" in request.args:
            year = int(request.args["year"])
        if "month" in request.args:
            month = int(request.args["month"])
    except ValueError:
        return _error("'year' and 'month' must be integers.", HTTPStatus.BAD_REQUEST)

    with get_db() as db:
        query = db.query(TransactionORM).outerjoin(
            CategoryORM, TransactionORM.id == CategoryORM.transaction_id
        )

        if year is not None:
            query = query.filter(extract("year", TransactionORM.date) == year)
        if month is not None:
            query = query.filter(extract("month", TransactionORM.date) == month)
        if category_filter:
            query = query.filter(CategoryORM.category == category_filter)

        total = query.count()
        rows = (
            query.order_by(TransactionORM.date.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        data = [_format_transaction(t) for t in rows]

    return _ok(
        data=data,
        meta={
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": max(1, math.ceil(total / per_page)) if total else 1,
        },
    )


@api_bp.get("/periods")
def periods():
    """
    GET /api/v1/periods

    Return a sorted list of all YYYY-MM periods that contain transaction data.
    Useful for populating a period selector in a front-end.
    """
    reporter = ExpenseReporter()
    return _ok(data={"periods": reporter.get_available_periods()})


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────


def _format_transaction(txn: TransactionORM) -> dict[str, Any]:
    """Serialize a TransactionORM (with optional joined Category) to a dict."""
    item: dict[str, Any] = {
        "id": txn.id,
        "date": txn.date.isoformat(),
        "description": txn.description,
        "amount": str(txn.amount),
        "currency": txn.currency,
        "merchant_name": txn.merchant_name,
        "source_file": txn.source_file,
        "ingested_at": txn.ingested_at.isoformat(),
        "category": None,
    }
    if txn.category:
        item["category"] = {
            "name": txn.category.category,
            "sub_category": txn.category.sub_category,
            "confidence": float(txn.category.confidence),
            "is_anomaly": txn.category.is_anomaly,
            "anomaly_reason": txn.category.anomaly_reason,
        }
    return item


# ─────────────────────────────────────────────────────────────
# Import guard (needed for /transactions endpoint)
# ─────────────────────────────────────────────────────────────

import math
from sqlalchemy import extract


# ─────────────────────────────────────────────────────────────
# Application Factory
# ─────────────────────────────────────────────────────────────


def create_app() -> Flask:
    """
    Flask application factory.

    Creates the Flask app, registers the API blueprint, initialises the database,
    and attaches global error handlers.

    Usage::

        app = create_app()
        app.run()

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.flask_secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_size_mb * 1024 * 1024
    app.config["TESTING"] = settings.flask_env == "testing"
    app.config["JSON_SORT_KEYS"] = False

    init_db()

    _register_root(app)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # ── Global error handlers ────────────────────────────────

    @app.errorhandler(400)
    def bad_request(e: Any) -> tuple[Any, int]:
        return _error("Bad request.", HTTPStatus.BAD_REQUEST)

    @app.errorhandler(404)
    def not_found(e: Any) -> tuple[Any, int]:
        return _error("Resource not found.", HTTPStatus.NOT_FOUND)

    @app.errorhandler(405)
    def method_not_allowed(e: Any) -> tuple[Any, int]:
        return _error("Method not allowed.", HTTPStatus.METHOD_NOT_ALLOWED)

    @app.errorhandler(413)
    def payload_too_large(e: Any) -> tuple[Any, int]:
        return _error(
            f"File too large. Maximum allowed size: {settings.max_upload_size_mb} MB.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    @app.errorhandler(500)
    def internal_error(e: Any) -> tuple[Any, int]:
        logger.exception("Unhandled server error")
        return _error("Internal server error.", HTTPStatus.INTERNAL_SERVER_ERROR)

    return app


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    application = create_app()
    application.run(
        host="0.0.0.0",
        port=5000,
        debug=settings.flask_env == "development",
    )
