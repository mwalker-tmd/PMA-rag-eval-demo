from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag_chatbot.domain.models import ReasoningEffort

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Test set generation defaults. Non-reasoning models: see README "Choosing models for SDG and judging".
DEFAULT_SDG_PROVIDER = "ragas"
DEFAULT_SDG_LLM_MODEL = "gpt-4.1"
DEFAULT_SDG_EMBEDDING_MODEL = "text-embedding-3-small"

# Evaluation (judge) defaults: non-reasoning, inexpensive, and not the answering model.
DEFAULT_EVAL_PROVIDER = "ragas"
DEFAULT_EVAL_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_EVAL_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EVAL_TIMEOUT_SECONDS = 360
DEFAULT_EVAL_ROW_DELAY_SECONDS = 2.0


class ConfigError(Exception):
    """The configuration (.env) is missing or invalid for what was asked."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    cohere_api_key: str | None
    llm_provider: str
    llm_model: str
    llm_reasoning_efforts: list[ReasoningEffort] | None  # None = let the adapter decide
    default_reasoning_effort: ReasoningEffort | None
    embedding_model: str
    data_dir: Path
    index_dir: Path
    datasets_dir: Path
    evals_dir: Path = PROJECT_ROOT / ".evals"
    sdg_provider: str = DEFAULT_SDG_PROVIDER
    sdg_llm_model: str = DEFAULT_SDG_LLM_MODEL
    sdg_embedding_model: str = DEFAULT_SDG_EMBEDDING_MODEL
    sdg_reasoning_effort: ReasoningEffort | None = None
    sdg_price_input_per_mtok: float | None = None  # USD per 1M tokens; both unset = no cost in estimates
    sdg_price_output_per_mtok: float | None = None
    eval_provider: str = DEFAULT_EVAL_PROVIDER
    eval_judge_model: str = DEFAULT_EVAL_JUDGE_MODEL
    eval_embedding_model: str = DEFAULT_EVAL_EMBEDDING_MODEL
    eval_judge_reasoning_effort: ReasoningEffort | None = None
    eval_timeout_seconds: int = DEFAULT_EVAL_TIMEOUT_SECONDS
    eval_row_delay_seconds: float = DEFAULT_EVAL_ROW_DELAY_SECONDS
    embedding_provider: str = "openai"  # openai | fake (tests only: deterministic, offline)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _parse_efforts(raw: str | None) -> list[ReasoningEffort] | None:
    if raw is None or not raw.strip():
        return None
    if raw.strip().lower() == "unsupported":
        return []
    return [ReasoningEffort(v.strip().lower()) for v in raw.split(",") if v.strip()]


def _optional_effort(raw: str | None) -> ReasoningEffort | None:
    raw = (raw or "").strip().lower()
    return ReasoningEffort(raw) if raw else None


def _optional_float(name: str) -> float | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = None
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number (USD per 1M tokens), got {raw!r}")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{name} must be a whole number of seconds (1 or more), got {raw!r}")
    return int(raw)


def _non_negative_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number of seconds, got {raw!r}")
    return value


def judge_matches_answer_model(settings: Settings) -> bool:
    """True when the judge would grade its own answers (EVAL_JUDGE_MODEL == LLM_MODEL), which inflates scores."""
    return settings.eval_judge_model.strip().casefold() == settings.llm_model.strip().casefold()


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    default_effort = os.getenv("DEFAULT_REASONING_EFFORT", "medium").strip().lower()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        cohere_api_key=os.getenv("COHERE_API_KEY") or None,
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        llm_reasoning_efforts=_parse_efforts(os.getenv("LLM_REASONING_EFFORTS")),
        default_reasoning_effort=ReasoningEffort(default_effort) if default_effort else None,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        data_dir=_resolve(os.getenv("DATA_DIR", "../data")),
        index_dir=_resolve(os.getenv("INDEX_DIR", ".index")),
        datasets_dir=_resolve(os.getenv("DATASETS_DIR", ".datasets")),
        evals_dir=_resolve(os.getenv("EVALS_DIR", ".evals")),
        sdg_provider=(os.getenv("SDG_PROVIDER") or DEFAULT_SDG_PROVIDER).strip().lower(),
        sdg_llm_model=(os.getenv("SDG_LLM_MODEL") or DEFAULT_SDG_LLM_MODEL).strip(),
        sdg_embedding_model=(os.getenv("SDG_EMBEDDING_MODEL") or DEFAULT_SDG_EMBEDDING_MODEL).strip(),
        sdg_reasoning_effort=_optional_effort(os.getenv("SDG_REASONING_EFFORT")),
        sdg_price_input_per_mtok=_optional_float("SDG_PRICE_INPUT_PER_MTOK"),
        sdg_price_output_per_mtok=_optional_float("SDG_PRICE_OUTPUT_PER_MTOK"),
        eval_provider=(os.getenv("EVAL_PROVIDER") or DEFAULT_EVAL_PROVIDER).strip().lower(),
        eval_judge_model=(os.getenv("EVAL_JUDGE_MODEL") or DEFAULT_EVAL_JUDGE_MODEL).strip(),
        eval_embedding_model=(os.getenv("EVAL_EMBEDDING_MODEL") or DEFAULT_EVAL_EMBEDDING_MODEL).strip(),
        eval_judge_reasoning_effort=_optional_effort(os.getenv("EVAL_JUDGE_REASONING_EFFORT")),
        eval_timeout_seconds=_positive_int("EVAL_TIMEOUT_SECONDS", DEFAULT_EVAL_TIMEOUT_SECONDS),
        eval_row_delay_seconds=_non_negative_float("EVAL_ROW_DELAY_SECONDS", DEFAULT_EVAL_ROW_DELAY_SECONDS),
        embedding_provider=(os.getenv("EMBEDDING_PROVIDER") or "openai").strip().lower(),
    )
