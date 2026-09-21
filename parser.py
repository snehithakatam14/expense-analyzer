# Ingests CSV/JSON transaction data, normalizes fields, deduplicates by SHA-256, persists to DB.
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import uuid
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import chardet
import pandas as pd

from database import get_db
from models import NormalizedTransaction, TransactionORM

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────


class ParserError(Exception):
    """Base exception for all parser failures."""


class MissingColumnError(ParserError):
    """Raised when a required canonical column cannot be resolved from the input headers."""

    def __init__(self, column: str, available: list[str]) -> None:
        self.column = column
        self.available = available
        super().__init__(
            f"Required column '{column}' not found in input. "
            f"Available columns: {available}. "
            f"Supported aliases: {COLUMN_ALIASES.get(column, [column])}"
        )


class DateParseError(ParserError):
    """Raised when a date string cannot be interpreted in any supported format."""

    def __init__(self, raw_value: str, row_index: int) -> None:
        self.raw_value = raw_value
        self.row_index = row_index
        super().__init__(
            f"Cannot parse date '{raw_value}' at row {row_index}. "
            f"Supported formats: {', '.join(DATE_FORMATS)}"
        )


class AmountParseError(ParserError):
    """Raised when an amount value cannot be coerced to Decimal."""

    def __init__(self, raw_value: str, row_index: int) -> None:
        self.raw_value = raw_value
        self.row_index = row_index
        super().__init__(
            f"Cannot parse amount '{raw_value}' at row {row_index}. "
            "Expected a numeric string, optionally with currency symbols or commas."
        )


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Maps a canonical column name to the list of aliases banks typically use.
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": [
        "date", "transaction_date", "txn_date", "trans_date",
        "posted_date", "value_date", "booking_date", "settlement_date",
    ],
    "description": [
        "description", "desc", "memo", "narrative", "details",
        "particulars", "transaction_description", "reference", "note",
    ],
    "amount": [
        "amount", "amt", "value", "transaction_amount",
        "debit_amount", "credit_amount", "net_amount", "sum", "total",
    ],
    "currency": [
        "currency", "currency_code", "ccy", "iso_currency", "currency_iso",
    ],
    "merchant_name": [
        "merchant_name", "merchant", "payee", "vendor", "counterparty",
        "beneficiary", "description_short", "shop_name",
    ],
}

REQUIRED_COLUMNS: list[str] = ["date", "description", "amount"]

DATE_FORMATS: list[str] = [
    "%Y-%m-%d",    # ISO 8601
    "%m/%d/%Y",    # US
    "%d-%m-%Y",    # EU dash
    "%d/%m/%Y",    # EU slash
    "%Y/%m/%d",    # ISO slash
    "%m-%d-%Y",    # US dash
    "%d.%m.%Y",    # EU dot
    "%Y.%m.%d",    # ISO dot
    "%b %d, %Y",   # Jan 15, 2024
    "%B %d, %Y",   # January 15, 2024
    "%d %b %Y",    # 15 Jan 2024
    "%d %B %Y",    # 15 January 2024
    "%Y%m%d",      # Compact YYYYMMDD
]


# ─────────────────────────────────────────────────────────────
# Pure Helper Functions (unit-testable in isolation)
# ─────────────────────────────────────────────────────────────


def _detect_encoding(raw_bytes: bytes) -> str:
    """
    Detect the character encoding of a byte string using chardet.
    Falls back to UTF-8 if detection confidence is below 60%.
    """
    result = chardet.detect(raw_bytes)
    encoding: str = result.get("encoding") or "utf-8"
    confidence: float = result.get("confidence") or 0.0
    if confidence < 0.6:
        logger.warning(
            "Low encoding detection confidence (%.0f%%). Defaulting to utf-8.", confidence * 100
        )
        return "utf-8"
    logger.debug("Detected encoding '%s' with %.0f%% confidence.", encoding, confidence * 100)
    return encoding


def _parse_date(raw: str, row_index: int) -> date:
    """
    Attempt to parse ``raw`` using each format in DATE_FORMATS.

    Args:
        raw:        Raw date string from the input file.
        row_index:  1-based row number for error messages.

    Returns:
        Parsed ``datetime.date`` object.

    Raises:
        DateParseError: If no format matches.
    """
    cleaned = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise DateParseError(raw_value=raw, row_index=row_index)


def _parse_amount(raw: str | float | int, row_index: int) -> Decimal:
    """
    Coerce ``raw`` to a Decimal, stripping currency symbols and commas.

    Examples that succeed:
        "-5.75", "$99.99", "1,234.56", -12.50, "€ 200", "(50.00)"

    Args:
        raw:        Raw amount value from the input file.
        row_index:  1-based row number for error messages.

    Returns:
        Decimal representation of the amount.

    Raises:
        AmountParseError: If the value cannot be interpreted as a number.
    """
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))

    cleaned = str(raw).strip()

    # Handle accounting notation: (50.00) → -50.00
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    # Remove commas (thousand separators) and whitespace
    cleaned = cleaned.replace(",", "").replace(" ", "")

    # Strip currency symbols while keeping minus/plus signs
    cleaned = "".join(
        ch for ch in cleaned if ch.isdigit() or ch in (".", "-", "+")
    )

    if not cleaned or cleaned in ("-", "+", "."):
        raise AmountParseError(raw_value=str(raw), row_index=row_index)

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        raise AmountParseError(raw_value=str(raw), row_index=row_index)


def _resolve_column(canonical: str, available: list[str]) -> str | None:
    """
    Find the actual column name in ``available`` that matches any alias for ``canonical``.
    Comparison is case-insensitive and strips surrounding whitespace.

    Returns None if no alias matches.
    """
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    lower_map: dict[str, str] = {c.strip().lower(): c for c in available}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _resolve_columns(columns: list[str]) -> dict[str, str | None]:
    """
    Build a mapping from every canonical column name to its actual name in the dataset.
    Values are None for columns that were not found.
    """
    return {canonical: _resolve_column(canonical, columns) for canonical in COLUMN_ALIASES}


def _compute_content_hash(txn_date: date, description: str, amount: Decimal) -> str:
    """
    Compute a SHA-256 fingerprint of the three fields that uniquely identify
    a transaction for de-duplication purposes.

    Both description and amount are normalised before hashing so that minor
    whitespace or precision differences do not produce false misses.
    """
    normalised = f"{txn_date.isoformat()}|{description.strip().lower()}|{amount}"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _is_null_row(row: dict[str, Any]) -> bool:
    """Return True if every value in the row is None or an empty/whitespace string."""
    return all(v is None or str(v).strip() == "" for v in row.values())


def _row_to_normalized(
    row: dict[str, Any],
    col_map: dict[str, str | None],
    row_index: int,
    source_file: str,
) -> NormalizedTransaction:
    """
    Convert a raw row dict to a fully validated NormalizedTransaction.

    All string fields are stripped.  Optional fields default to empty string / "USD".
    The content_hash is computed inside NormalizedTransaction.model_post_init().

    Raises:
        DateParseError:   If the date field cannot be parsed.
        AmountParseError: If the amount field cannot be coerced.
    """
    # ── Date ────────────────────────────────────────────────
    date_col = col_map.get("date")
    raw_date = row.get(date_col or "__missing__", "") if date_col else ""
    if raw_date is None or str(raw_date).strip() == "":
        raise DateParseError(raw_value="<empty>", row_index=row_index)
    txn_date = _parse_date(str(raw_date), row_index)

    # ── Description ─────────────────────────────────────────
    desc_col = col_map.get("description")
    raw_desc = row.get(desc_col or "__missing__", "") if desc_col else ""
    description = str(raw_desc).strip() if raw_desc is not None else ""

    # ── Amount ───────────────────────────────────────────────
    amount_col = col_map.get("amount")
    raw_amount = row.get(amount_col or "__missing__", None) if amount_col else None
    if raw_amount is None or str(raw_amount).strip() == "":
        raise AmountParseError(raw_value="<empty>", row_index=row_index)
    amount = _parse_amount(raw_amount, row_index)

    # ── Currency (optional) ──────────────────────────────────
    currency = "USD"
    currency_col = col_map.get("currency")
    if currency_col and currency_col in row and row[currency_col]:
        raw_currency = str(row[currency_col]).strip().upper()
        currency = raw_currency[:3] if raw_currency else "USD"

    # ── Merchant name (optional) ─────────────────────────────
    merchant_name = ""
    merchant_col = col_map.get("merchant_name")
    if merchant_col and merchant_col in row and row[merchant_col]:
        merchant_name = str(row[merchant_col]).strip()

    # If merchant_name absent, derive it from the description prefix
    if not merchant_name and description:
        merchant_name = (
            description.split(" - ")[0].strip()
            if " - " in description
            else description.split("|")[0].strip()
        )

    return NormalizedTransaction(
        id=str(uuid.uuid4()),
        date=txn_date,
        description=description,
        amount=amount,
        currency=currency,
        merchant_name=merchant_name,
        source_file=source_file,
        ingested_at=datetime.utcnow(),
        # content_hash computed in model_post_init
    )


def _persist_transactions(transactions: list[NormalizedTransaction]) -> tuple[int, int]:
    """
    Write a list of NormalizedTransactions to the database, skipping duplicates.

    Uses content_hash for idempotent de-duplication (bulk-check first, then insert).

    Returns:
        (inserted_count, duplicate_count)
    """
    if not transactions:
        return 0, 0

    hashes = [t.content_hash for t in transactions]

    with get_db() as db:
        # Bulk existence check: single query instead of N queries
        existing_hashes: set[str] = {
            row[0]
            for row in db.query(TransactionORM.content_hash)
            .filter(TransactionORM.content_hash.in_(hashes))
            .all()
        }

        inserted = 0
        duplicates = 0

        for txn in transactions:
            if txn.content_hash in existing_hashes:
                logger.debug("Duplicate skipped: hash=%s", txn.content_hash[:16])
                duplicates += 1
                continue

            db.add(
                TransactionORM(
                    id=txn.id,
                    date=txn.date,
                    description=txn.description,
                    amount=float(txn.amount),
                    currency=txn.currency,
                    merchant_name=txn.merchant_name,
                    source_file=txn.source_file,
                    ingested_at=txn.ingested_at,
                    content_hash=txn.content_hash,
                )
            )
            # Track locally so same-batch duplicates are also caught
            existing_hashes.add(txn.content_hash)
            inserted += 1

    return inserted, duplicates


# ─────────────────────────────────────────────────────────────
# TransactionParser — Public API
# ─────────────────────────────────────────────────────────────


class TransactionParser:
    """
    Ingests raw CSV or JSON transaction data, normalizes all fields,
    handles edge cases, de-duplicates, and optionally persists to the database.

    Typical usage::

        parser = TransactionParser()
        transactions = parser.parse_csv(uploaded_file_bytes, source_file="bank_jan.csv")

    Args:
        persist: If True (default), de-duplicated transactions are written to SQLite.
                 Set to False in tests or dry-run scenarios.
    """

    def __init__(self, persist: bool = True) -> None:
        self._persist = persist

    # ── CSV ──────────────────────────────────────────────────────────────

    def parse_csv(
        self,
        source: str | bytes | io.IOBase,
        source_file: str = "<upload>",
    ) -> list[NormalizedTransaction]:
        """
        Parse a CSV file and return a list of NormalizedTransaction objects.

        ``source`` may be:
          - A file path string or Path object
          - Raw bytes (e.g. from ``flask.request.files["file"].read()``)
          - A file-like object (anything with a ``.read()`` method)

        The parser auto-detects file encoding via chardet and retries with UTF-8
        if confidence is low.

        Per-row errors (malformed dates, bad amounts) are logged as warnings and
        the row is skipped.  The method only raises if *no* valid rows were found.

        Args:
            source:       Input data — file path, bytes, or file-like object.
            source_file:  Label attached to each transaction for audit purposes.

        Returns:
            List of NormalizedTransaction (may be shorter than total row count
            if some rows were skipped due to errors or duplicate detection).

        Raises:
            ParserError:         Empty file, no headers, or all rows failed.
            MissingColumnError:  A required column is absent from the headers.
        """
        raw_bytes = self._read_to_bytes(source)

        if not raw_bytes.strip():
            raise ParserError(f"File '{source_file}' is empty or contains only whitespace.")

        encoding = _detect_encoding(raw_bytes)
        text = raw_bytes.decode(encoding, errors="replace")

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise ParserError(f"CSV '{source_file}' has no headers or is malformed.")

        columns: list[str] = [c for c in reader.fieldnames if c is not None]
        col_map = _resolve_columns(columns)
        self._validate_required_columns(col_map, columns)

        transactions: list[NormalizedTransaction] = []
        parse_errors: list[str] = []

        for row_index, row in enumerate(reader, start=2):  # Row 1 is the header
            if _is_null_row(row):
                logger.debug("CSV row %d is empty — skipped.", row_index)
                continue
            try:
                txn = _row_to_normalized(row, col_map, row_index, source_file)
                transactions.append(txn)
            except ParserError as exc:
                logger.warning("CSV row %d skipped (%s): %s", row_index, source_file, exc)
                parse_errors.append(f"Row {row_index}: {exc}")

        if not transactions:
            detail = "\n".join(parse_errors[:10])
            raise ParserError(
                f"No valid transactions found in '{source_file}'.\n{detail}"
            )

        logger.info(
            "CSV parsed — %d valid, %d skipped | source='%s'",
            len(transactions), len(parse_errors), source_file,
        )

        if self._persist:
            inserted, dupes = _persist_transactions(transactions)
            logger.info("DB write — %d inserted, %d duplicates", inserted, dupes)

        return transactions

    # ── JSON ─────────────────────────────────────────────────────────────

    def parse_json(
        self,
        source: str | bytes | io.IOBase | list[dict[str, Any]],
        source_file: str = "<upload>",
    ) -> list[NormalizedTransaction]:
        """
        Parse JSON transaction data and return a list of NormalizedTransaction objects.

        ``source`` may be:
          - An already-parsed Python list of dicts
          - Raw bytes or a file-like object containing JSON
          - A JSON file path string

        Accepted JSON shapes:
          - Array of transaction objects: ``[{...}, {...}]``
          - Wrapped object: ``{"transactions": [{...}]}`` (also tries "data", "records", "items")
          - Single transaction object: ``{...}``

        Args:
            source:       Input data in one of the accepted forms.
            source_file:  Label for audit trail.

        Returns:
            List of NormalizedTransaction objects.

        Raises:
            ParserError:         Empty input, invalid JSON, or all rows failed.
            MissingColumnError:  A required column is absent from the JSON keys.
        """
        raw_data = self._load_json(source, source_file)
        raw_data = self._unwrap_json(raw_data, source_file)

        if not raw_data:
            raise ParserError(f"JSON input '{source_file}' contains no transactions.")

        # Column names are the union of all keys across all rows
        all_keys: list[str] = list({k for row in raw_data for k in row})
        col_map = _resolve_columns(all_keys)
        self._validate_required_columns(col_map, all_keys)

        transactions: list[NormalizedTransaction] = []
        parse_errors: list[str] = []

        for row_index, row in enumerate(raw_data, start=1):
            if _is_null_row(row):
                logger.debug("JSON row %d is empty — skipped.", row_index)
                continue
            try:
                txn = _row_to_normalized(row, col_map, row_index, source_file)
                transactions.append(txn)
            except ParserError as exc:
                logger.warning("JSON row %d skipped (%s): %s", row_index, source_file, exc)
                parse_errors.append(f"Row {row_index}: {exc}")

        if not transactions:
            detail = "\n".join(parse_errors[:10])
            raise ParserError(
                f"No valid transactions found in '{source_file}'.\n{detail}"
            )

        logger.info(
            "JSON parsed — %d valid, %d skipped | source='%s'",
            len(transactions), len(parse_errors), source_file,
        )

        if self._persist:
            inserted, dupes = _persist_transactions(transactions)
            logger.info("DB write — %d inserted, %d duplicates", inserted, dupes)

        return transactions

    # ── DataFrame ────────────────────────────────────────────────────────

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        source_file: str = "<dataframe>",
    ) -> list[NormalizedTransaction]:
        """
        Parse a pandas DataFrame directly.

        Useful when data arrives pre-processed (e.g., from a database export
        or programmatic test fixtures).

        Args:
            df:           Input DataFrame. Column names are matched against aliases.
            source_file:  Label for audit trail.

        Returns:
            List of NormalizedTransaction objects.

        Raises:
            ParserError:         Empty DataFrame or all rows failed.
            MissingColumnError:  A required column is absent.
        """
        if df.empty:
            raise ParserError(f"DataFrame '{source_file}' is empty.")

        columns = list(df.columns)
        col_map = _resolve_columns(columns)
        self._validate_required_columns(col_map, columns)

        # Convert NaN → None for uniform null checking
        records: list[dict[str, Any]] = df.where(pd.notna(df), None).to_dict(orient="records")
        transactions: list[NormalizedTransaction] = []
        parse_errors: list[str] = []

        for row_index, row in enumerate(records, start=1):
            if _is_null_row(row):
                continue
            try:
                txn = _row_to_normalized(row, col_map, row_index, source_file)
                transactions.append(txn)
            except ParserError as exc:
                logger.warning("DataFrame row %d skipped: %s", row_index, exc)
                parse_errors.append(f"Row {row_index}: {exc}")

        if not transactions:
            detail = "\n".join(parse_errors[:10])
            raise ParserError(
                f"No valid transactions in DataFrame '{source_file}'.\n{detail}"
            )

        if self._persist:
            inserted, dupes = _persist_transactions(transactions)
            logger.info("DB write — %d inserted, %d duplicates", inserted, dupes)

        return transactions

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _read_to_bytes(source: str | bytes | io.IOBase) -> bytes:
        """Convert any supported source type to raw bytes."""
        if isinstance(source, bytes):
            return source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return path.read_bytes()
            # Treat as raw CSV string
            return source.encode("utf-8")
        # File-like object
        content = source.read()
        return content if isinstance(content, bytes) else content.encode("utf-8")

    @staticmethod
    def _load_json(
        source: str | bytes | io.IOBase | list[dict[str, Any]],
        source_file: str,
    ) -> Any:
        """Deserialise ``source`` to a Python object."""
        if isinstance(source, list):
            return source
        if isinstance(source, dict):
            return source

        raw_bytes: bytes
        if isinstance(source, bytes):
            raw_bytes = source
        elif isinstance(source, (str, Path)):
            path = Path(str(source))
            if path.exists():
                raw_bytes = path.read_bytes()
            else:
                raw_bytes = str(source).encode("utf-8")
        else:
            content = source.read()
            raw_bytes = content if isinstance(content, bytes) else content.encode("utf-8")

        encoding = _detect_encoding(raw_bytes)
        text = raw_bytes.decode(encoding, errors="replace").strip()
        if not text:
            raise ParserError(f"JSON file '{source_file}' is empty.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParserError(f"Invalid JSON in '{source_file}': {exc}") from exc

    @staticmethod
    def _unwrap_json(raw: Any, source_file: str) -> list[dict[str, Any]]:
        """
        Normalise the parsed JSON value to a flat list of dicts.

        Handles:
          - list        → returned as-is
          - dict with a list under a known wrapper key → unwrapped
          - single dict → wrapped in a list
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("transactions", "data", "records", "items", "results"):
                if key in raw and isinstance(raw[key], list):
                    logger.debug("JSON: unwrapped '%s' key from object.", key)
                    return raw[key]
            # Single transaction object
            return [raw]
        raise ParserError(
            f"JSON in '{source_file}' must be a list of objects or a dict "
            "with a 'transactions' / 'data' / 'records' key."
        )

    @staticmethod
    def _validate_required_columns(
        col_map: dict[str, str | None],
        available: list[str],
    ) -> None:
        """Raise MissingColumnError for the first required column not found."""
        for required in REQUIRED_COLUMNS:
            if col_map.get(required) is None:
                raise MissingColumnError(column=required, available=available)
