"""
tests/test_parser.py — Unit tests for parser.py

Coverage:
  - _parse_date():          11 formats, whitespace, invalid → DateParseError
  - _parse_amount():        negatives, commas, symbols, accounting notation, edge cases
  - _is_null_row():         all-None, all-empty, mixed
  - _compute_content_hash(): determinism, uniqueness
  - CSV parsing:            valid, missing columns, malformed dates, trailing spaces,
                            empty rows, empty file, encoding, column aliases
  - JSON parsing:           list input, bytes, file-like, wrapped dict, single object,
                            empty, invalid JSON, missing columns
  - DataFrame parsing:      basic happy path, empty DataFrame
"""
from __future__ import annotations

import io
import json
import os
import pytest
import pandas as pd
from datetime import date
from decimal import Decimal
from unittest.mock import patch

# Point at in-memory DB before importing any module that touches the DB
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FLASK_ENV", "testing")

from parser import (
    AmountParseError,
    DateParseError,
    MissingColumnError,
    ParserError,
    TransactionParser,
    _compute_content_hash,
    _is_null_row,
    _parse_amount,
    _parse_date,
    _resolve_column,
    _resolve_columns,
)


# ─────────────────────────────────────────────────────────────
# Raw CSV fixtures
# ─────────────────────────────────────────────────────────────

VALID_CSV = b"""date,description,amount,currency,merchant_name
2024-01-15,Starbucks Coffee,-5.75,USD,Starbucks
2024-01-16,Uber Ride,-12.50,USD,Uber
2024-01-20,Salary Deposit,3000.00,USD,Employer Inc
"""

ALIAS_COLUMN_CSV = b"""transaction_date,memo,amt,ccy,payee
2024-01-15,Starbucks Coffee,-5.75,USD,Starbucks
2024-01-16,Uber Ride,-12.50,USD,Uber
"""

MISSING_DATE_CSV = b"""description,amount,currency
Starbucks Coffee,-5.75,USD
"""

MISSING_AMOUNT_CSV = b"""date,description
2024-01-15,Starbucks Coffee
"""

MALFORMED_DATE_CSV = b"""date,description,amount
not-a-date,Starbucks Coffee,-5.75
2024-01-16,Uber Ride,-12.50
"""

TRAILING_SPACES_CSV = b"""date,description,amount,merchant_name
2024-01-15,  Starbucks Coffee  ,-5.75,  Starbucks  
2024-01-16,Uber Ride  ,-12.50,Uber
"""

EMPTY_ROWS_CSV = b"""date,description,amount
2024-01-15,Valid One,-5.75

,  ,
2024-01-16,Valid Two,-10.00
"""

ACCOUNTING_NOTATION_CSV = b"""date,description,amount
2024-01-15,Starbucks Coffee,(5.75)
2024-01-16,Uber Ride,(12.50)
"""

COMMA_AMOUNT_CSV = b"""date,description,amount
2024-01-15,Rent Payment,\"1,500.00\"
"""

CURRENCY_LOWERCASE_CSV = b"""date,description,amount,currency
2024-01-15,Coffee,-5.00,eur
"""

HEADERS_ONLY_CSV = b"date,description,amount\n"


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def parser():
    """Parser with DB persistence disabled."""
    return TransactionParser(persist=False)


# ─────────────────────────────────────────────────────────────
# _parse_date tests
# ─────────────────────────────────────────────────────────────

class TestParseDate:
    def test_iso_format(self):
        assert _parse_date("2024-01-15", 1) == date(2024, 1, 15)

    def test_us_slash_format(self):
        assert _parse_date("01/15/2024", 1) == date(2024, 1, 15)

    def test_eu_dash_format(self):
        assert _parse_date("15-01-2024", 1) == date(2024, 1, 15)

    def test_eu_slash_format(self):
        assert _parse_date("15/01/2024", 1) == date(2024, 1, 15)

    def test_year_slash_format(self):
        assert _parse_date("2024/01/15", 1) == date(2024, 1, 15)

    def test_eu_dot_format(self):
        assert _parse_date("15.01.2024", 1) == date(2024, 1, 15)

    def test_abbreviated_month(self):
        assert _parse_date("Jan 15, 2024", 1) == date(2024, 1, 15)

    def test_full_month_name(self):
        assert _parse_date("January 15, 2024", 1) == date(2024, 1, 15)

    def test_compact_yyyymmdd(self):
        assert _parse_date("20240115", 1) == date(2024, 1, 15)

    def test_leading_trailing_whitespace(self):
        assert _parse_date("  2024-01-15  ", 1) == date(2024, 1, 15)

    def test_invalid_raises_date_parse_error(self):
        with pytest.raises(DateParseError) as exc_info:
            _parse_date("not-a-date", 5)
        assert exc_info.value.row_index == 5
        assert exc_info.value.raw_value == "not-a-date"

    def test_empty_raises(self):
        with pytest.raises(DateParseError):
            _parse_date("", 1)

    def test_partial_date_raises(self):
        with pytest.raises(DateParseError):
            _parse_date("2024-13", 1)  # Incomplete / invalid month 13


# ─────────────────────────────────────────────────────────────
# _parse_amount tests
# ─────────────────────────────────────────────────────────────

class TestParseAmount:
    def test_negative_decimal_string(self):
        assert _parse_amount("-5.75", 1) == Decimal("-5.75")

    def test_positive_integer_string(self):
        assert _parse_amount("3000", 1) == Decimal("3000")

    def test_comma_thousands_separator(self):
        assert _parse_amount("1,234.56", 1) == Decimal("1234.56")

    def test_dollar_sign_stripped(self):
        assert _parse_amount("$99.99", 1) == Decimal("99.99")

    def test_euro_sign_stripped(self):
        assert _parse_amount("€ 200.00", 1) == Decimal("200.00")

    def test_accounting_notation_positive(self):
        # (50.00) means negative in accounting
        assert _parse_amount("(50.00)", 1) == Decimal("-50.00")

    def test_float_input(self):
        assert _parse_amount(-12.50, 1) == Decimal("-12.5")

    def test_int_input(self):
        assert _parse_amount(3000, 1) == Decimal("3000")

    def test_zero(self):
        assert _parse_amount("0.00", 1) == Decimal("0.00")

    def test_empty_string_raises(self):
        with pytest.raises(AmountParseError) as exc_info:
            _parse_amount("", 3)
        assert exc_info.value.row_index == 3

    def test_alpha_string_raises(self):
        with pytest.raises(AmountParseError):
            _parse_amount("abc", 1)

    def test_lone_minus_raises(self):
        with pytest.raises(AmountParseError):
            _parse_amount("-", 1)

    def test_large_amount(self):
        result = _parse_amount("1,000,000.00", 1)
        assert result == Decimal("1000000.00")


# ─────────────────────────────────────────────────────────────
# _is_null_row tests
# ─────────────────────────────────────────────────────────────

class TestIsNullRow:
    def test_all_none(self):
        assert _is_null_row({"a": None, "b": None, "c": None}) is True

    def test_all_empty_strings(self):
        assert _is_null_row({"a": "", "b": "   ", "c": "\t"}) is True

    def test_mixed_none_and_empty(self):
        assert _is_null_row({"a": None, "b": ""}) is True

    def test_has_data(self):
        assert _is_null_row({"a": "hello", "b": None}) is False

    def test_zero_amount(self):
        # "0" is a valid data value — not null
        assert _is_null_row({"a": "0"}) is False

    def test_empty_dict(self):
        assert _is_null_row({}) is True


# ─────────────────────────────────────────────────────────────
# _compute_content_hash tests
# ─────────────────────────────────────────────────────────────

class TestContentHash:
    def test_hash_is_64_hex_chars(self):
        h = _compute_content_hash(date(2024, 1, 15), "Starbucks", Decimal("-5.75"))
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_inputs_same_hash(self):
        h1 = _compute_content_hash(date(2024, 1, 15), "Starbucks", Decimal("-5.75"))
        h2 = _compute_content_hash(date(2024, 1, 15), "Starbucks", Decimal("-5.75"))
        assert h1 == h2

    def test_different_dates_different_hash(self):
        h1 = _compute_content_hash(date(2024, 1, 15), "Starbucks", Decimal("-5.75"))
        h2 = _compute_content_hash(date(2024, 1, 16), "Starbucks", Decimal("-5.75"))
        assert h1 != h2

    def test_description_case_insensitive(self):
        h1 = _compute_content_hash(date(2024, 1, 15), "STARBUCKS", Decimal("-5.75"))
        h2 = _compute_content_hash(date(2024, 1, 15), "starbucks", Decimal("-5.75"))
        assert h1 == h2

    def test_whitespace_normalised(self):
        h1 = _compute_content_hash(date(2024, 1, 15), "  Starbucks  ", Decimal("-5.75"))
        h2 = _compute_content_hash(date(2024, 1, 15), "Starbucks", Decimal("-5.75"))
        assert h1 == h2


# ─────────────────────────────────────────────────────────────
# _resolve_column tests
# ─────────────────────────────────────────────────────────────

class TestResolveColumn:
    def test_exact_match(self):
        assert _resolve_column("date", ["date", "amount", "description"]) == "date"

    def test_alias_match_transaction_date(self):
        assert _resolve_column("date", ["transaction_date", "amount"]) == "transaction_date"

    def test_alias_match_memo(self):
        assert _resolve_column("description", ["memo", "amount", "date"]) == "memo"

    def test_case_insensitive(self):
        assert _resolve_column("date", ["DATE", "AMOUNT"]) == "DATE"

    def test_no_match_returns_none(self):
        assert _resolve_column("date", ["foo", "bar"]) is None


# ─────────────────────────────────────────────────────────────
# CSV Parsing — happy path
# ─────────────────────────────────────────────────────────────

class TestCSVParsing:
    def test_valid_csv_count(self, parser):
        results = parser.parse_csv(VALID_CSV, source_file="test.csv")
        assert len(results) == 3

    def test_amounts_correct(self, parser):
        results = parser.parse_csv(VALID_CSV, source_file="test.csv")
        amounts = {r.description: r.amount for r in results}
        assert amounts["Starbucks Coffee"] == Decimal("-5.75")
        assert amounts["Uber Ride"] == Decimal("-12.50")
        assert amounts["Salary Deposit"] == Decimal("3000.00")

    def test_dates_parsed(self, parser):
        results = parser.parse_csv(VALID_CSV, source_file="test.csv")
        assert results[0].date == date(2024, 1, 15)
        assert results[1].date == date(2024, 1, 16)

    def test_source_file_set(self, parser):
        results = parser.parse_csv(VALID_CSV, source_file="bank_jan.csv")
        assert all(r.source_file == "bank_jan.csv" for r in results)

    def test_currency_defaults_to_usd(self, parser):
        csv = b"date,description,amount\n2024-01-15,Coffee,-5.00\n"
        results = parser.parse_csv(csv, source_file="t.csv")
        assert results[0].currency == "USD"

    def test_currency_normalized_uppercase(self, parser):
        results = parser.parse_csv(CURRENCY_LOWERCASE_CSV, source_file="t.csv")
        assert results[0].currency == "EUR"

    def test_content_hash_is_64_chars(self, parser):
        results = parser.parse_csv(VALID_CSV, source_file="t.csv")
        assert all(len(r.content_hash) == 64 for r in results)

    def test_hashes_are_consistent_across_calls(self, parser):
        r1 = parser.parse_csv(VALID_CSV, source_file="t.csv")
        r2 = parser.parse_csv(VALID_CSV, source_file="t.csv")
        assert r1[0].content_hash == r2[0].content_hash

    def test_hashes_are_unique_within_file(self, parser):
        results = parser.parse_csv(VALID_CSV, source_file="t.csv")
        hashes = [r.content_hash for r in results]
        assert len(hashes) == len(set(hashes))

    def test_column_aliases_resolved(self, parser):
        results = parser.parse_csv(ALIAS_COLUMN_CSV, source_file="alias.csv")
        assert len(results) == 2
        assert results[0].date == date(2024, 1, 15)
        assert results[0].description == "Starbucks Coffee"
        assert results[0].currency == "USD"

    def test_trailing_spaces_stripped(self, parser):
        results = parser.parse_csv(TRAILING_SPACES_CSV, source_file="t.csv")
        assert results[0].description == "Starbucks Coffee"
        assert results[0].merchant_name == "Starbucks"

    def test_empty_rows_skipped(self, parser):
        results = parser.parse_csv(EMPTY_ROWS_CSV, source_file="t.csv")
        assert len(results) == 2

    def test_accounting_notation_amounts(self, parser):
        results = parser.parse_csv(ACCOUNTING_NOTATION_CSV, source_file="t.csv")
        assert results[0].amount == Decimal("-5.75")
        assert results[1].amount == Decimal("-12.50")

    def test_comma_in_quoted_amount(self, parser):
        results = parser.parse_csv(COMMA_AMOUNT_CSV, source_file="t.csv")
        assert results[0].amount == Decimal("1500.00")


# ─────────────────────────────────────────────────────────────
# CSV Parsing — error cases
# ─────────────────────────────────────────────────────────────

class TestCSVParsingErrors:
    def test_empty_bytes_raises(self, parser):
        with pytest.raises(ParserError, match="empty"):
            parser.parse_csv(b"", source_file="t.csv")

    def test_whitespace_only_raises(self, parser):
        with pytest.raises(ParserError, match="empty"):
            parser.parse_csv(b"   \n  \t  ", source_file="t.csv")

    def test_missing_date_column_raises(self, parser):
        with pytest.raises(MissingColumnError) as exc_info:
            parser.parse_csv(MISSING_DATE_CSV, source_file="t.csv")
        assert exc_info.value.column == "date"
        assert "description" in exc_info.value.available

    def test_missing_amount_column_raises(self, parser):
        with pytest.raises(MissingColumnError) as exc_info:
            parser.parse_csv(MISSING_AMOUNT_CSV, source_file="t.csv")
        assert exc_info.value.column == "amount"

    def test_malformed_date_skips_row(self, parser):
        """Row with malformed date is skipped; valid rows still returned."""
        results = parser.parse_csv(MALFORMED_DATE_CSV, source_file="t.csv")
        assert len(results) == 1
        assert results[0].date == date(2024, 1, 16)

    def test_headers_only_raises(self, parser):
        with pytest.raises(ParserError):
            parser.parse_csv(HEADERS_ONLY_CSV, source_file="t.csv")

    def test_file_like_object_accepted(self, parser):
        file_like = io.BytesIO(VALID_CSV)
        results = parser.parse_csv(file_like, source_file="t.csv")
        assert len(results) == 3


# ─────────────────────────────────────────────────────────────
# JSON Parsing
# ─────────────────────────────────────────────────────────────

VALID_JSON_LIST = [
    {"date": "2024-01-15", "description": "Starbucks Coffee", "amount": -5.75, "currency": "USD"},
    {"date": "2024-01-16", "description": "Netflix", "amount": -15.99, "currency": "USD"},
]


class TestJSONParsing:
    def test_list_of_dicts(self, parser):
        results = parser.parse_json(VALID_JSON_LIST, source_file="t.json")
        assert len(results) == 2

    def test_raw_bytes(self, parser):
        raw = json.dumps(VALID_JSON_LIST).encode()
        results = parser.parse_json(raw, source_file="t.json")
        assert len(results) == 2

    def test_file_like_object(self, parser):
        raw = io.BytesIO(json.dumps(VALID_JSON_LIST).encode())
        results = parser.parse_json(raw, source_file="t.json")
        assert len(results) == 2

    def test_wrapped_transactions_key(self, parser):
        wrapped = {"transactions": VALID_JSON_LIST}
        results = parser.parse_json(wrapped, source_file="t.json")
        assert len(results) == 2

    def test_wrapped_data_key(self, parser):
        wrapped = {"data": VALID_JSON_LIST}
        results = parser.parse_json(wrapped, source_file="t.json")
        assert len(results) == 2

    def test_single_dict_object(self, parser):
        single = {"date": "2024-01-15", "description": "Coffee", "amount": -5.0}
        results = parser.parse_json(single, source_file="t.json")
        assert len(results) == 1

    def test_amounts_correct(self, parser):
        results = parser.parse_json(VALID_JSON_LIST, source_file="t.json")
        assert results[0].amount == Decimal("-5.75")
        assert results[1].amount == Decimal("-15.99")

    def test_empty_list_raises(self, parser):
        with pytest.raises(ParserError, match="no transactions"):
            parser.parse_json([], source_file="t.json")

    def test_invalid_json_bytes_raises(self, parser):
        with pytest.raises(ParserError, match="Invalid JSON"):
            parser.parse_json(b"not valid json {{{", source_file="t.json")

    def test_missing_date_column_raises(self, parser):
        data = [{"description": "Coffee", "amount": -5.0}]
        with pytest.raises(MissingColumnError) as exc_info:
            parser.parse_json(data, source_file="t.json")
        assert exc_info.value.column == "date"

    def test_missing_amount_column_raises(self, parser):
        data = [{"date": "2024-01-15", "description": "Coffee"}]
        with pytest.raises(MissingColumnError) as exc_info:
            parser.parse_json(data, source_file="t.json")
        assert exc_info.value.column == "amount"

    def test_currencies_normalized(self, parser):
        data = [
            {"date": "2024-01-15", "description": "Coffee", "amount": -5.0, "currency": "eur"}
        ]
        results = parser.parse_json(data, source_file="t.json")
        assert results[0].currency == "EUR"

    def test_null_rows_skipped(self, parser):
        data = [
            {"date": "2024-01-15", "description": "Coffee", "amount": -5.0},
            {"date": None, "description": None, "amount": None},
            {"date": "2024-01-16", "description": "Uber", "amount": -12.0},
        ]
        results = parser.parse_json(data, source_file="t.json")
        assert len(results) == 2


# ─────────────────────────────────────────────────────────────
# DataFrame Parsing
# ─────────────────────────────────────────────────────────────

class TestDataFrameParsing:
    def test_valid_dataframe(self, parser):
        df = pd.DataFrame([
            {"date": "2024-01-15", "description": "Starbucks", "amount": -5.75},
            {"date": "2024-01-16", "description": "Uber", "amount": -12.50},
        ])
        results = parser.parse_dataframe(df, source_file="df_test")
        assert len(results) == 2

    def test_empty_dataframe_raises(self, parser):
        df = pd.DataFrame()
        with pytest.raises(ParserError, match="empty"):
            parser.parse_dataframe(df, source_file="empty.df")

    def test_missing_column_raises(self, parser):
        df = pd.DataFrame([{"description": "Coffee", "amount": -5.0}])
        with pytest.raises(MissingColumnError):
            parser.parse_dataframe(df, source_file="df_test")


# ─────────────────────────────────────────────────────────────
# Persistence (with mocked DB)
# ─────────────────────────────────────────────────────────────

class TestPersistence:
    def test_persist_called_when_enabled(self):
        """Verify _persist_transactions is called when persist=True."""
        parser = TransactionParser(persist=True)
        with patch("parser._persist_transactions", return_value=(2, 1)) as mock_persist:
            results = parser.parse_csv(VALID_CSV, source_file="t.csv")
        mock_persist.assert_called_once()
        assert len(results) == 3

    def test_persist_not_called_when_disabled(self):
        """Verify _persist_transactions is NOT called when persist=False."""
        parser = TransactionParser(persist=False)
        with patch("parser._persist_transactions") as mock_persist:
            parser.parse_csv(VALID_CSV, source_file="t.csv")
        mock_persist.assert_not_called()
