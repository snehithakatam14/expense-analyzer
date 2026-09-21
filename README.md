# expense-analyzer

A REST API that processes bank transaction data, categorizes expenses using the OpenAI API, and flags unusual spending patterns. Built with Flask, SQLAlchemy, and PyTest.

---

## What it does

Upload a CSV or JSON export from your bank, run the categorizer, and get a breakdown of where your money went — by category, month-over-month changes, and any transactions that look out of place statistically.

The dashboard at `localhost:5000` shows charts and a transaction table so you can see everything at a glance.

---

## Tech stack

- Python 3.11
- Flask (REST API, Blueprint-organized)
- SQLite via SQLAlchemy 2.0
- OpenAI API — `gpt-4o-mini` with structured outputs (Pydantic-enforced response schema)
- Tenacity for retry logic
- Pandas for data processing
- PyTest — 179 tests across all four modules

---

## Setup

```bash
git clone https://github.com/snehithakatam14/expense-analyzer.git
cd expense-analyzer
pip install -r requirements.txt
cp .env.example .env
# add your OPENAI_API_KEY to .env
python app.py
```

Open `http://localhost:5000` in a browser.

---

## API endpoints

```
GET  /                              Dashboard UI
GET  /api/v1/health                 Server status
POST /api/v1/upload                 Upload a CSV or JSON file
POST /api/v1/categorize             Run AI categorization
GET  /api/v1/report?year=&month=    Monthly report
GET  /api/v1/anomalies              Flagged transactions
GET  /api/v1/transactions           Paginated list
GET  /api/v1/periods                Available months
```

Example upload:
```bash
curl -X POST http://localhost:5000/api/v1/upload -F "file=@transactions.csv"
```

---

## Project layout

```
expense_analyzer/
├── app.py              Flask API
├── parser.py           CSV/JSON ingestion and normalization
├── categorizer.py      OpenAI categorization with retry logic
├── reporter.py         Monthly reports, anomaly detection, health score
├── models.py           SQLAlchemy ORM + Pydantic schemas
├── database.py         DB setup and session management
├── config.py           Environment config via pydantic-settings
├── requirements.txt
├── .env.example
├── templates/
│   └── dashboard.html
└── tests/
    ├── conftest.py
    ├── test_parser.py
    ├── test_categorizer.py
    ├── test_reporter.py
    └── test_app.py
```

---

## CSV format

The parser handles column name variations automatically:

| Field | Accepted column names |
|---|---|
| date | `date`, `transaction_date`, `trans_date`, `posted_date` |
| description | `description`, `memo`, `narrative`, `details` |
| amount | `amount`, `amt`, `transaction_amount` |
| currency | `currency`, `ccy`, `currency_code` |
| merchant | `merchant_name`, `payee`, `merchant`, `vendor` |

Accounting notation like `(50.00)` is parsed as `-50.00`. Encoding is auto-detected.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `DATABASE_URL` | `sqlite:///./expense_analyzer.db` | |
| `CATEGORIZER_BATCH_SIZE` | `20` | Transactions per API call |
| `ANOMALY_ZSCORE_THRESHOLD` | `2.0` | |
| `MAX_UPLOAD_SIZE_MB` | `50` | |
