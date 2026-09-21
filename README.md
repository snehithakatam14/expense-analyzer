# 💰 AI-Powered Financial Expense Analyzer

> A production-grade REST API that ingests bank transaction data, categorizes it using **OpenAI GPT-4o-mini Structured Outputs**, detects anomalies with **Z-score statistical analysis**, and serves a live analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=flat&logo=flask)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat&logo=openai)
![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy%202.0-003B57?style=flat&logo=sqlite)
![Tests](https://img.shields.io/badge/Tests-179%20passed-4ade80?style=flat&logo=pytest)

---

## ✨ Features

- **CSV & JSON ingestion** — Handles 11 date formats, column aliases, accounting notation `(50.00) → -50.00`, encoding detection, and SHA-256 deduplication
- **AI Categorization** — OpenAI Structured Outputs with Pydantic schema enforcement, batch processing (20 tx/call), tenacity retry logic, and graceful fallback
- **Anomaly Detection** — Z-score (σ > 2.0) statistical engine merged with AI-flagged anomalies
- **Budget Intelligence** — Monthly drift %, 4-component health score (0–100), income/spend split
- **Live Dashboard** — Dark-themed analytics UI with donut chart, drift bar chart, health score ring, and transactions table
- **179 unit tests** — Full PyTest coverage across all 4 components

---

## 🏗️ Architecture

```
CSV/JSON Upload
      │
      ▼
  parser.py  ──────────────────► SQLite DB (TransactionORM)
  (TransactionParser)                     │
                                          ▼
                               categorizer.py (TransactionCategorizer)
                                    │   ▲
                                    │   └── OpenAI GPT-4o-mini
                                    ▼       (Structured Outputs)
                               SQLite DB (CategoryORM)
                                          │
                                          ▼
                               reporter.py (ExpenseReporter)
                               Z-score · Drift · Health Score
                                          │
                                          ▼
                                app.py (Flask REST API)
                                          │
                                          ▼
                               Live Dashboard (/)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API Framework | Flask + Blueprints |
| Database | SQLite via SQLAlchemy 2.0 ORM |
| AI / LLM | OpenAI GPT-4o-mini (Structured Outputs) |
| Data Validation | Pydantic v2 + pydantic-settings |
| Retry Logic | Tenacity (exponential backoff) |
| Data Processing | Pandas + chardet |
| Testing | PyTest + pytest-mock (179 tests) |
| Frontend | Vanilla JS + Chart.js |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/expense-analyzer.git
cd expense-analyzer
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the API
```bash
python app.py
# Server starts at http://localhost:5000
```

### 4. Open the dashboard
Navigate to **http://localhost:5000** in your browser.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live analytics dashboard |
| `GET` | `/api/v1/health` | Liveness probe |
| `POST` | `/api/v1/upload` | Upload CSV or JSON transactions |
| `POST` | `/api/v1/categorize` | AI-categorize all transactions |
| `GET` | `/api/v1/report?year=&month=` | Monthly expense report |
| `GET` | `/api/v1/anomalies?year=&month=` | Anomalous transactions |
| `GET` | `/api/v1/transactions?page=&per_page=` | Paginated transaction list |
| `GET` | `/api/v1/periods` | All available months |

### Upload example
```bash
curl -X POST http://localhost:5000/api/v1/upload -F "file=@transactions.csv"
```

### Response envelope
All responses follow a consistent structure:
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-01-15T10:30:00Z",
  "meta": { "message": "..." }
}
```

---

## 📁 Project Structure

```
expense_analyzer/
├── app.py              # Flask REST API (7 endpoints, Blueprint)
├── parser.py           # CSV/JSON/DataFrame ingestion
├── categorizer.py      # OpenAI AI categorization + retry
├── reporter.py         # Z-score anomaly, drift, health score
├── models.py           # SQLAlchemy ORM + Pydantic schemas
├── database.py         # DB engine, WAL mode, session factory
├── config.py           # pydantic-settings environment config
├── requirements.txt    # Pinned production dependencies
├── .env.example        # Environment variable template
├── templates/
│   └── dashboard.html  # Live analytics dashboard
└── tests/
    ├── conftest.py
    ├── test_parser.py       # 63 tests
    ├── test_categorizer.py  # 36 tests
    ├── test_reporter.py     # 47 tests
    └── test_app.py          # 33 tests
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# 179 passed
```

---

## 📊 Data Flow — CSV Column Support

The parser auto-resolves column aliases:

| Canonical | Accepted Aliases |
|---|---|
| `date` | `transaction_date`, `trans_date`, `posted_date` |
| `description` | `memo`, `narrative`, `details`, `transaction_description` |
| `amount` | `amt`, `debit_credit`, `transaction_amount` |
| `currency` | `ccy`, `currency_code` |
| `merchant_name` | `payee`, `merchant`, `vendor` |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI secret key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for categorization |
| `DATABASE_URL` | `sqlite:///./expense_analyzer.db` | SQLAlchemy connection string |
| `CATEGORIZER_BATCH_SIZE` | `20` | Transactions per API call |
| `ANOMALY_ZSCORE_THRESHOLD` | `2.0` | Z-score cutoff for anomaly flagging |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |

---

## 📄 License

MIT
