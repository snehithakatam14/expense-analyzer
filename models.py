"""
models.py — Data contracts and ORM definitions.

This module is the single source of truth for ALL data shapes in the system.
It defines:
  - SQLAlchemy ORM models  (persisted to SQLite)
  - Pydantic schemas       (validated Python types flowing between components)
  - TransactionCategory    (shared Enum across parser, categorizer, reporter, API)

Schema flow:
  parser.py       → NormalizedTransaction → DB + categorizer.py
  categorizer.py  → CategoryResponse      → DB + reporter.py
  reporter.py     → ReportSummary         → app.py (JSON response)
  OpenAI API      → BatchCategorizationResponse (structured output schema)
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


# ─────────────────────────────────────────────────────────────
# Shared Enum — used in ORM, Pydantic, and OpenAI schema
# ─────────────────────────────────────────────────────────────


class TransactionCategory(str, Enum):
    """
    Canonical category labels for all financial transactions.

    Using str-based Enum so values serialize naturally to JSON strings
    and are directly usable as SQLAlchemy column values.
    """

    FOOD = "Food"
    TRANSPORT = "Transport"
    HOUSING = "Housing"
    HEALTHCARE = "Healthcare"
    ENTERTAINMENT = "Entertainment"
    SHOPPING = "Shopping"
    UTILITIES = "Utilities"
    INCOME = "Income"
    TRAVEL = "Travel"
    EDUCATION = "Education"
    OTHER = "Other"


# ─────────────────────────────────────────────────────────────
# SQLAlchemy ORM Base
# ─────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    __allow_unmapped__ = True  # Allow string-quoted forward-ref annotations on relationships


# ─────────────────────────────────────────────────────────────
# ORM Models — persisted to SQLite
# ─────────────────────────────────────────────────────────────


class TransactionORM(Base):
    """
    Stores a single normalized financial transaction.

    content_hash is a SHA-256 fingerprint of (date + description + amount)
    used for idempotent de-duplication at ingestion time.
    """

    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    date = Column(Date, nullable=False, index=True)
    description = Column(Text, nullable=False)
    amount = Column(Numeric(precision=15, scale=2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    merchant_name = Column(String(255), nullable=True)
    source_file = Column(String(500), nullable=False)
    ingested_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)

    # One-to-one: a transaction may have zero or one Category record
    category: "CategoryORM" = relationship(
        "CategoryORM",
        back_populates="transaction",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_transactions_date_currency", "date", "currency"),
    )


class CategoryORM(Base):
    """
    Stores the AI-generated category for a single transaction.

    confidence is stored as a 4-digit decimal (e.g., 0.953).
    is_anomaly and anomaly_reason are set by both the AI categorizer
    and the statistical Z-score engine in reporter.py.
    """

    __tablename__ = "categories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(
        String(36),
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    category = Column(String(50), nullable=False, index=True)
    sub_category = Column(String(100), nullable=True)
    confidence = Column(Numeric(precision=4, scale=3), nullable=False)
    reasoning = Column(Text, nullable=True)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_reason = Column(Text, nullable=True)
    categorized_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    transaction: "TransactionORM" = relationship(
        "TransactionORM",
        back_populates="category",
    )


class ReportORM(Base):
    """
    Caches a generated monthly ReportSummary as a JSON blob.

    Keyed by period (YYYY-MM). Overwritten on re-generation.
    """

    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    period = Column(String(7), nullable=False, unique=True, index=True)  # YYYY-MM
    total_spend = Column(Numeric(precision=15, scale=2), nullable=False)
    budget_health_score = Column(Numeric(precision=5, scale=2), nullable=False)
    report_data = Column(Text, nullable=False)  # Full JSON blob of ReportSummary
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas — validated Python types (not persisted directly)
# ─────────────────────────────────────────────────────────────


class NormalizedTransaction(BaseModel):
    """
    Output contract from parser.py.
    Input contract to categorizer.py and the DB persistence layer.

    Produced by TransactionParser.parse_csv() / parse_json().
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: date
    description: str
    amount: Decimal
    currency: str = "USD"
    merchant_name: str = ""
    source_file: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute content_hash if not provided."""
        if not self.content_hash:
            raw = f"{self.date.isoformat()}|{self.description.strip().lower()}|{self.amount}"
            object.__setattr__(
                self,
                "content_hash",
                hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        stripped = v.strip().upper()
        return stripped[:3] if stripped else "USD"

    @field_validator("description", "merchant_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class CategoryResponse(BaseModel):
    """
    Output contract from categorizer.py.
    Wraps the raw OpenAI structured output and adds transaction_id linkage.
    Persisted to CategoryORM.
    """

    transaction_id: str
    category: TransactionCategory
    sub_category: str = Field(
        description="Specific sub-category, e.g. 'Coffee & Cafes' under Food."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        description="Brief human-readable rationale for the categorization."
    )
    is_anomaly: bool = Field(
        description="True if the transaction is flagged as anomalous by the AI."
    )
    anomaly_reason: str | None = Field(
        default=None,
        description="Explanation if is_anomaly is True.",
    )


class AnomalyRecord(BaseModel):
    """
    A single anomalous transaction — included in ReportSummary.anomalies
    and returned by GET /api/v1/anomalies.
    """

    transaction_id: str
    date: date
    description: str
    amount: Decimal
    category: str
    anomaly_reason: str
    z_score: float | None = Field(
        default=None,
        description="Statistical Z-score (None if detected by AI, not statistics).",
    )


class ReportSummary(BaseModel):
    """
    Output contract from reporter.py.
    Returned by GET /api/v1/report as a JSON response.
    """

    model_config = {"arbitrary_types_allowed": True}

    period: str = Field(description="Report period in YYYY-MM format.")
    total_spend: Decimal = Field(description="Sum of all expense transactions.")
    total_income: Decimal = Field(description="Sum of all income transactions.")
    net: Decimal = Field(description="total_income - total_spend.")
    by_category: dict[str, Decimal] = Field(
        description="Total spend per category name."
    )
    monthly_drift: dict[str, float] = Field(
        description="% change vs prior month per category. Positive = increased spend."
    )
    anomalies: list[AnomalyRecord] = Field(
        description="List of statistically or AI-flagged anomalous transactions."
    )
    budget_health_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Composite 0–100 score reflecting overall budget health.",
    )
    transaction_count: int = Field(
        description="Total number of transactions in this period."
    )
    categorized_count: int = Field(
        description="Number of transactions with a category assigned."
    )


# ─────────────────────────────────────────────────────────────
# OpenAI Structured Output Schemas
# These Pydantic models are passed as response_format to the OpenAI API.
# The API enforces the JSON schema, guaranteeing type-safe responses.
# ─────────────────────────────────────────────────────────────


class TransactionCategorization(BaseModel):
    """
    Single-transaction structured output schema for the OpenAI API.
    One of these is returned per transaction in the batch.
    """

    category: TransactionCategory
    sub_category: str = Field(
        description="Specific sub-category, e.g. 'Coffee & Cafes' under Food."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        description="Short explanation for the chosen category."
    )
    is_anomaly: bool = Field(
        description="True if the amount or merchant seems unusually suspicious."
    )
    anomaly_reason: str | None = Field(
        default=None,
        description="Explanation when is_anomaly is True.",
    )


class BatchCategorizationResponse(BaseModel):
    """
    Top-level structured output schema sent to the OpenAI API as response_format.
    The API guarantees results has exactly as many items as transactions sent.
    """

    results: list[TransactionCategorization] = Field(
        description="Categorizations in the same order as the input transactions."
    )
