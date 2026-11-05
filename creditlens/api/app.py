"""
CreditLens FastAPI application.

Endpoints:
  POST /score              — Score a single applicant
  POST /score/batch        — Score multiple applicants
  POST /explain/{id}       — Get SHAP explanation for a decision
  GET  /model/metrics      — Model performance metrics
  GET  /fairness/report    — Fairness audit, on the held-out set and a large fresh draw
  GET  /health             — Health check
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creditlens.data.domain import (
    BALANCE_TO_MONTHLY_INCOME, INFLOW_TO_MONTHLY_INCOME, MAX_PAYMENT_BURDEN, OUTFLOW_TO_INFLOW,
    VALIDITY_DOMAIN, domain_warnings,
)
from creditlens.decision import (
    DECISION_REASONS, probability_to_score, score_band_table, score_to_decision,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CreditLens API",
    description="An explainable credit-scoring reference: every decision has an attribution that sums exactly to the served probability, and the fairness audit reports intervals and says INCONCLUSIVE when it should.",
    version="0.1.0",
)


# Largest request body accepted, in bytes. The largest VALID request — 1,000
# applicants (the batch maximum) with every field at full float precision — is
# 865,016 bytes (2026-11-04; tests/test_validation.py builds it), so 2 MiB never rejects a
# valid request while bounding what an unauthenticated caller can make the
# server buffer and parse.
MAX_BODY_BYTES = 2 * 1024 * 1024


class BodySizeLimitMiddleware:
    """Reject a request body over ``max_bytes`` with 413 BEFORE the app parses it.

    A declared Content-Length is checked up front (the server holds the body to
    the declared length). A body without one (chunked) is buffered here up to
    the limit and replayed to the app, so an oversized stream is cut off at
    ``max_bytes`` rather than read in full.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None:
            if not declared.isdigit() or int(declared) > self.max_bytes:
                await self._reject(send)
                return
            await self.app(scope, receive, send)
            return

        body, more = b"", True
        while more:
            message = await receive()
            if message["type"] != "http.request":  # client went away
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            more = message.get("more_body", False)

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send) -> None:
        payload = json.dumps(
            {"detail": f"Request body exceeds {self.max_bytes} bytes."}
        ).encode()
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode())],
        })
        await send({"type": "http.response.body", "body": payload})


app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
# Added last, so it is outermost: every response, a 413 included, carries CORS
# headers (a browser would otherwise report the 413 as an opaque CORS failure).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _json_safe(value: Any, cap: bool = True) -> Any:
    """A validation-error detail that can always be encoded: non-finite numbers
    become strings ("inf", "nan"), a long string is truncated when ``cap`` is set,
    and a lone surrogate is written out as text. JSON's "\\ud800" escape parses to a
    Python string that UTF-8 cannot encode; extra="forbid" echoes an unknown key in
    the error's location, so {"\\ud800": 1} failed while encoding its own 422."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        value = value.encode("utf-8", "backslashreplace").decode("utf-8")
        return f"{value[:100]}... ({len(value)} characters)" if cap and len(value) > 100 else value
    if isinstance(value, dict):
        return {_json_safe(k, cap): _json_safe(v, cap) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, cap) for v in value]
    return value


# What a validation error echoes from the caller, and so what is bounded. Everything
# else in it (the reason, its context) is this API's own wording and reaches the client
# whole, so the live page never cuts a reason off mid-sentence.
_ECHOED_ERROR_KEYS = {"input", "loc"}


@app.exception_handler(RequestValidationError)
async def _request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 echoes each offending input. When that input is NaN,
    +/-Infinity or a lone surrogate the echo itself cannot be encoded, which would
    turn the 422 into a 500, and a 100 kB invalid string would come back 100 kB long.
    This keeps the body shape, made safe to encode, with what it echoes bounded in size."""
    detail = [{key: _json_safe(value, cap=key in _ECHOED_ERROR_KEYS) for key, value in error.items()}
              for error in jsonable_encoder(exc.errors())]
    return JSONResponse(status_code=422, content={"detail": detail})

# In-memory, PER-PROCESS model state (trained model, feature engineer,
# evaluation, fairness report, SHAP explainer). This is a single-worker demo:
# the state lives in one process's memory and is NOT shared across workers. Run
# the API with a single worker (the default `uvicorn creditlens.api.app:app`).
# Under multiple workers (e.g. `--workers N` or gunicorn) each worker trains and
# holds its OWN copy, so a /train against one worker is invisible to the others
# and /explain on a fresh worker lazy-trains a different model. For a real
# multi-worker deployment, persist the fitted model to shared storage (load from
# file / object store) or a model server instead of this dict. See README
# "Deployment note (single-worker demo)".
_model_state: dict[str, Any] = {}

# ONE training configuration for every path that builds the served model: the
# startup hook, the lazy-train fallback, and a body-less POST /train, so a retrain
# cannot swap in a different model under the documented scores. Every documented
# number is derived from this. 10,000 rows is enough for the alternative-data
# fields' modest true effects to be learned in rough proportion (README "Model
# behaviour" measures it), and training on it takes about a second.
DEFAULT_TRAIN_SAMPLES = 10000
DEFAULT_SEED = 42
MIN_TRAIN_SAMPLES = 2000

# POST /train replaces, for every later caller, the model the API scores with,
# and the API has no authentication. It is therefore OFF unless the operator
# opts in by setting this environment variable to 1 / true / yes. Startup
# training does not need it.
TRAIN_ENV_FLAG = "CREDITLENS_ENABLE_TRAIN"


def _train_enabled() -> bool:
    return os.environ.get(TRAIN_ENV_FLAG, "").strip().lower() in {"1", "true", "yes"}


@app.on_event("startup")
def _startup_train() -> None:
    """Train a small synthetic model on startup so the demo works with zero
    manual steps. Kept intentionally small (fast boot); failures are non-fatal
    because scoring will lazy-train on first request as a fallback."""
    try:
        train_model(n_samples=DEFAULT_TRAIN_SAMPLES, seed=DEFAULT_SEED)
        logger.info("CreditLens startup training complete.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Startup training failed (will lazy-train on first request): {e}")



# Every request model is strict: finite numbers only (Python's JSON parser
# accepts NaN / Infinity), and unknown fields are rejected rather than ignored (a
# misspelled "num_delinquency": 3 must not score the applicant as never delinquent).
# strict=True: numbers must arrive as JSON numbers and integer fields as JSON
# integers, so "fico_score": "700" and "nsf_count": true are refused (a JSON integer
# is still accepted for a float field).
_STRICT = ConfigDict(allow_inf_nan=False, extra="forbid", strict=True)
_D = VALIDITY_DOMAIN


class ApplicantFeatures(BaseModel):
    """Input features for a single credit applicant.

    Every numeric field is finite and range-checked; SECURITY.md's table is the
    same bounds, and tests/test_validation.py fails if the two ever differ.

    Every model input is REQUIRED (422 when omitted): a default would assert something
    about the applicant, so omitting bad news cannot improve a decision. Net cash flow is
    not an input: the engine always derives it as ``monthly_inflow - monthly_outflow``,
    and sending it is a 422.

    There is no interest rate. The offered rate is the lender's own risk-based
    price (in the synthetic data it is priced from FICO, correlation -0.92), so
    a score that consumes it is circular.

    Every bound is the validity domain (data/domain.py): the default training
    data's range plus a 5% margin. Five joint
    rules bound what the per-field ranges cannot: inflow must match income, outflow
    must match inflow, the balance must match income, the monthly payment must be
    within the training range of income, and inflow minus outflow must stay in net
    cash flow's range.
    """

    model_config = _STRICT

    # Bounds = the validity domain (data/domain.py): the default training
    # data's range plus a 5% margin. Outside it the model has no data.
    annual_income: float = Field(..., ge=_D["annual_income"][0], le=_D["annual_income"][1],
                                 description="Annual income in USD")
    loan_amount: float = Field(..., ge=_D["loan_amount"][0], le=_D["loan_amount"][1],
                               description="Requested loan amount in USD")
    loan_term_months: int = Field(..., ge=_D["loan_term_months"][0], le=_D["loan_term_months"][1],
                                  description="Loan term in months")
    debt_to_income: float = Field(..., ge=_D["debt_to_income"][0], le=_D["debt_to_income"][1],
                                  description="Debt-to-income ratio")
    employment_years: float = Field(..., ge=_D["employment_years"][0], le=_D["employment_years"][1],
                                    description="Years at current employer")
    fico_score: int = Field(..., ge=_D["fico_score"][0], le=_D["fico_score"][1],
                            description="FICO credit score")
    credit_utilization: float = Field(..., ge=_D["credit_utilization"][0],
                                      le=_D["credit_utilization"][1])
    num_delinquencies: int = Field(..., ge=_D["num_delinquencies"][0],
                                   le=_D["num_delinquencies"][1])
    # The model caps recency at 24 months (features/engineering.py), where the
    # label's recency risk ends.
    months_since_delinquency: int = Field(
        ..., ge=-1, le=_D["months_since_delinquency"][1],
        description="Months since the most recent delinquency; -1 = never "
                    "delinquent. Must be -1 exactly when num_delinquencies is 0.",
    )
    num_hard_inquiries: int = Field(..., ge=_D["num_hard_inquiries"][0],
                                    le=_D["num_hard_inquiries"][1])
    credit_age_months: float = Field(..., ge=_D["credit_age_months"][0],
                                     le=_D["credit_age_months"][1])
    monthly_inflow: float = Field(..., ge=_D["monthly_inflow"][0], le=_D["monthly_inflow"][1])
    monthly_outflow: float = Field(..., ge=_D["monthly_outflow"][0], le=_D["monthly_outflow"][1])
    paycheck_regularity: float = Field(..., ge=_D["paycheck_regularity"][0],
                                       le=_D["paycheck_regularity"][1])
    bill_payment_consistency: float = Field(..., ge=_D["bill_payment_consistency"][0],
                                            le=_D["bill_payment_consistency"][1])
    nsf_count: int = Field(..., ge=_D["nsf_count"][0], le=_D["nsf_count"][1],
                           description="NSF incidents, last 12 months")
    overdraft_days: int = Field(..., ge=_D["overdraft_days"][0], le=_D["overdraft_days"][1],
                                description="Days overdrawn, last 12 months")
    rent_payment_score: float = Field(..., ge=_D["rent_payment_score"][0],
                                      le=_D["rent_payment_score"][1])
    utility_payment_score: float = Field(..., ge=_D["utility_payment_score"][0],
                                         le=_D["utility_payment_score"][1])
    banking_tenure_months: float = Field(..., ge=_D["banking_tenure_months"][0],
                                         le=_D["banking_tenure_months"][1])
    avg_account_balance: float = Field(..., ge=_D["avg_account_balance"][0],
                                       le=_D["avg_account_balance"][1])
    savings_rate: float = Field(..., ge=_D["savings_rate"][0], le=_D["savings_rate"][1])
    demographic_group: str = Field(
        "", max_length=64,
        description="Optional. Accepted and ignored: never a model feature, "
                    "never stored. The fairness audit uses the synthetic "
                    "training data's groups, not this field.",
    )

    @model_validator(mode="after")
    def _derived_net_cashflow_is_in_range(self):
        # Net cash flow is a model input the engine derives, so it answers to its
        # training range.
        expected = self.monthly_inflow - self.monthly_outflow
        lo, hi = _D["monthly_net_cashflow"]
        if not lo <= expected <= hi:
            raise ValueError(
                f"monthly_inflow - monthly_outflow must be {lo:,.0f} to {hi:,.0f} "
                f"(the training range); it is {expected:,.2f}"
            )
        return self

    @model_validator(mode="after")
    def _inflow_matches_income(self):
        # The generator draws inflow as annual_income / 12 x U(0.80, 1.20). A
        # higher stated income with unchanged deposits is not an applicant the
        # model has seen; the tolerance is that range plus a 5% margin.
        lo, hi = INFLOW_TO_MONTHLY_INCOME
        monthly_income = self.annual_income / 12
        ratio = self.monthly_inflow / monthly_income
        if not lo <= ratio <= hi:
            raise ValueError(
                f"monthly_inflow must be {lo:g}-{hi:g} x annual_income / 12 "
                f"({lo * monthly_income:,.0f}-{hi * monthly_income:,.0f}); "
                f"it is {ratio:.2f} x"
            )
        return self

    @model_validator(mode="after")
    def _outflow_matches_inflow(self):
        # The generator draws outflow as inflow x U(0.50, 0.95). Spending that
        # falls outside it is not an applicant the model has seen, and net cash flow
        # would reward it: the tolerance is that range plus a 5% margin.
        lo, hi = OUTFLOW_TO_INFLOW
        ratio = self.monthly_outflow / self.monthly_inflow
        if not lo <= ratio <= hi:
            raise ValueError(
                f"monthly_outflow must be {lo:g}-{hi:g} x monthly_inflow "
                f"({lo * self.monthly_inflow:,.0f}-{hi * self.monthly_inflow:,.0f}); "
                f"it is {ratio:.2f} x"
            )
        return self

    @model_validator(mode="after")
    def _balance_matches_income(self):
        # The generator draws the balance as annual_income / 12 x U(0.5, 3.0).
        lo, hi = BALANCE_TO_MONTHLY_INCOME
        monthly_income = self.annual_income / 12
        ratio = self.avg_account_balance / monthly_income
        if not lo <= ratio <= hi:
            raise ValueError(
                f"avg_account_balance must be {lo:g}-{hi:g} x annual_income / 12 "
                f"({lo * monthly_income:,.0f}-{hi * monthly_income:,.0f}); it is {ratio:.2f} x"
            )
        return self

    @model_validator(mode="after")
    def _payment_within_income(self):
        burden = (self.loan_amount / self.loan_term_months) / (self.annual_income / 12)
        if burden > MAX_PAYMENT_BURDEN:
            raise ValueError(
                f"the monthly loan payment (loan_amount / loan_term_months) must be at "
                f"most {MAX_PAYMENT_BURDEN:g} x monthly income (the training range); "
                f"it is {burden:.2f} x"
            )
        return self

    @model_validator(mode="after")
    def _delinquency_recency_matches_count(self):
        # In the training data -1 ("never") is exactly num_delinquencies == 0
        # (data/generator.py). A contradictory pair is off-distribution and the
        # model would read a delinquent applicant as never delinquent.
        never = self.months_since_delinquency == -1
        if self.num_delinquencies == 0 and not never:
            raise ValueError("months_since_delinquency must be -1 (never delinquent) "
                             "when num_delinquencies is 0")
        if self.num_delinquencies > 0 and never:
            raise ValueError("months_since_delinquency (0-120) is required "
                             "when num_delinquencies > 0")
        return self


class ScoreResponse(BaseModel):
    credit_score: int
    default_probability: float
    decision: str
    decision_reason: str
    # In-domain values few training applicants share (data/domain.py).
    warnings: list[str]


class BatchScoreRequest(BaseModel):
    model_config = _STRICT
    # Cap the batch length so an unauthenticated caller cannot submit an
    # arbitrarily huge list and exhaust CPU/memory. Over-limit -> 422.
    applicants: list[ApplicantFeatures] = Field(..., min_length=1, max_length=1000)


class TrainRequest(BaseModel):
    model_config = _STRICT
    # Bound n_samples so an unauthenticated caller cannot request a gigantic
    # synthetic dataset + model fit (CPU/memory DoS). Out-of-range -> 422.
    # The floor keeps the model and its audit meaningful: 2,000 rows -> a
    # 400-applicant audit set. Far fewer would audit fairness on a handful of
    # group-C applicants.
    n_samples: int = Field(default=DEFAULT_TRAIN_SAMPLES, ge=MIN_TRAIN_SAMPLES, le=50000)
    # numpy's seed range.
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**32 - 1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _model_state,
        "service": "CreditLens",
        "version": "0.1.0",
    }


@app.post("/score", response_model=ScoreResponse)
def score_applicant(applicant: ApplicantFeatures):
    """Score a single credit applicant."""
    _ensure_model_loaded()

    df = pd.DataFrame([applicant.model_dump()])
    model = _model_state["model"]
    engineer = _model_state["engineer"]

    try:
        X, _ = engineer.transform(df)
        proba = float(model.predict_proba(X)[0, 1])
        score = probability_to_score(proba)

        decision, reason = _get_decision(score, applicant)
        return ScoreResponse(
            credit_score=score,
            default_probability=round(proba, 4),
            decision=decision,
            decision_reason=reason,
            warnings=domain_warnings(applicant.model_dump()),
        )
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail="Internal scoring error.")


@app.post("/score/batch")
def score_batch(request: BatchScoreRequest):
    """Score multiple applicants in one call."""
    _ensure_model_loaded()

    df = pd.DataFrame([a.model_dump() for a in request.applicants])
    model = _model_state["model"]
    engineer = _model_state["engineer"]

    try:
        X, _ = engineer.transform(df)
        probas = model.predict_proba(X)[:, 1]
        scores = probability_to_score(probas)
        # One source of truth for the decision bands, so /score and /score/batch
        # cannot return different decisions for the same applicant.
        decisions = [
            _get_decision(int(s), a)[0] for s, a in zip(scores, request.applicants)
        ]
        return {
            "count": len(scores),
            "results": [
                {"index": i, "credit_score": int(s), "default_probability": round(float(p), 4),
                 "decision": d, "warnings": domain_warnings(a.model_dump())}
                for i, (s, p, d, a) in enumerate(zip(scores, probas, decisions,
                                                     request.applicants))
            ],
        }
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        raise HTTPException(status_code=500, detail="Internal scoring error.")


@app.post("/explain/{applicant_idx}")
def explain_decision(applicant_idx: int, applicant: ApplicantFeatures):
    """
    Get SHAP explanation for a credit decision.

    Returns each input's contribution to the served P(default).
    """
    _ensure_model_loaded()

    df = pd.DataFrame([applicant.model_dump()])

    try:
        explainer = _model_state.get("explainer")
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="No explainer for the current model: its setup failed (see the server log).",
            )

        explanation = explainer.explain_individual(df, applicant_id=applicant_idx)
        return {
            "applicant_id": applicant_idx,
            "credit_score": explanation.predicted_score,
            "default_probability": explanation.predicted_default_prob,
            "decision": explanation.decision,
            "warnings": domain_warnings(applicant.model_dump()),
            "explanation_text": explanation.explanation_text,
            "top_positive_factors": explanation.top_positive_factors[:5],
            "top_negative_factors": explanation.top_negative_factors[:5],
            # The full attribution, in the SAME space as default_probability:
            # base_value + sum(shap_values) == the served P(default) (before the
            # 4-dp rounding of default_probability). Empty when unavailable.
            "attribution_space": "probability_of_default",
            "attribution_method": explainer.method,
            # Each key of shap_values is one of the applicant's inputs, or a
            # group of inputs that move together (listed here).
            "attribution_players": {k: list(v) for k, v in explainer.players.items()},
            "base_value": explanation.base_value if explanation.status == "ok" else None,
            "shap_values": explanation.shap_values,
            # 'ok' when SHAP attribution succeeded; 'unavailable' (with a reason)
            # when it failed — the score/decision are still valid but the factor
            # lists will be empty, so the client shows "explanation unavailable"
            # instead of silently rendering no factors.
            "explanation_status": explanation.status,
            "explanation_status_reason": explanation.status_reason,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal explanation error.")


@app.get("/model/metrics")
def get_model_metrics():
    """Return model performance metrics from the last training run."""
    if "evaluation" not in _model_state:
        raise HTTPException(status_code=404, detail="No model trained yet. POST /train to train.")
    eval_ = _model_state["evaluation"]
    return {
        "model_name": eval_.model_name,
        "auc_roc": round(eval_.auc_roc, 4),
        "auc_pr": round(eval_.auc_pr, 4),
        "gini_coefficient": round(eval_.gini_coefficient, 4),
        "brier_score": round(eval_.brier_score, 4),
        "ks_statistic": round(eval_.ks_statistic, 4),
    }


def _finite(x: Any, digits: int = 4) -> float | None:
    """Round for the wire; undefined rates (NaN) become null — JSON has no NaN,
    and Starlette refuses to encode one (a NaN would otherwise be a 500)."""
    x = float(x)
    return None if not math.isfinite(x) else round(x, digits)


def _fairness_payload(report: Any) -> dict[str, Any]:
    """Wire shape of a FairnessReport: the legacy point fields, plus every
    criterion's definition, interval, verdict and per-pair breakdown."""
    return {
        "overall_verdict": report.overall_verdict,
        "overall_pass": report.overall_pass,
        "decision_rule": report.decision_rule,
        "confidence": report.confidence,
        "interval_method": (
            "stratified bootstrap of each group's approved x repaid table, "
            f"{report.n_bootstrap} replicates; percentile intervals of signed differences, "
            "Bonferroni-adjusted over group pairs (simultaneous per criterion). Equalized "
            "odds adjusts over its two gaps as well and adds their bounds, which is conservative"
        ),
        "demographic_parity_diff": _finite(report.demographic_parity_diff),
        "equal_opportunity_diff": _finite(report.equal_opportunity_diff),
        "equalized_odds_diff": _finite(report.equalized_odds_diff),
        "disparate_impact_ratio": _finite(report.disparate_impact_ratio),
        # Reported, not gated: the worst pair's gap in P(default | not approved).
        "predictive_parity_diff": _finite(report.predictive_parity_diff),
        "passes_demographic_parity": report.passes_demographic_parity,
        "passes_equal_opportunity": report.passes_equal_opportunity,
        "passes_disparate_impact": report.passes_disparate_impact,
        "passes_equalized_odds": report.passes_equalized_odds,
        "checks": {
            name: {
                "definition": c.definition,
                "value": _finite(c.value),
                "ci": [_finite(c.ci_low), _finite(c.ci_high)],
                "threshold": c.threshold,
                "passes_when": c.passes_when,
                "verdict": c.verdict,
                "reason": c.reason,
                "worst_pair": list(c.worst_pair) if c.worst_pair else None,
                "pairs": [
                    {"groups": list(pg.groups), "estimate": _finite(pg.estimate),
                     "ci": [_finite(pg.ci_low), _finite(pg.ci_high)], "verdict": pg.verdict,
                     "reason": pg.reason}
                    for pg in c.pairs
                ],
            }
            for name, c in report.checks.items()
        },
        "recommendations": report.recommendations,
        "group_metrics": {
            g: {
                "n_samples": m.n_samples,
                "n_repaid": m.n_repaid,
                "n_defaulted": m.n_positive,
                "n_approved": m.n_approved,
                "approval_rate": _finite(m.approval_rate),
                "approval_rate_repaid": _finite(m.approval_rate_repaid),
                "approval_rate_defaulted": _finite(m.approval_rate_defaulted),
                "true_positive_rate": _finite(m.true_positive_rate),
                "avg_score": _finite(m.avg_score, 1),
                "default_rate": _finite(m.default_rate),
                "mean_predicted_default": _finite(m.mean_predicted_default),
                "calibration_in_the_large": _finite(m.calibration_in_the_large),
                "calibration_ci": [_finite(m.calibration_ci[0]), _finite(m.calibration_ci[1])],
            }
            for g, m in report.group_metrics.items()
        },
    }


# The audit set is small and one draw. This engine can draw as many applicants as
# it likes from the world it was trained on, so /fairness/report also audits the
# served model on a large fresh draw: what the small set's interval cannot show.
POPULATION_N = 400_000
POPULATION_SEED = 2026
# Rows scored at a time. The calculation is row by row, and transforming the whole
# draw at once would hold hundreds of MB of copies (measured: README "Deploy").
POPULATION_CHUNK = 10_000
# The audit is computed once, and the page calls it on load. Without this, callers
# arriving during the first computation would each run their own and hold the draw.
_population_lock = threading.Lock()


def _population_report() -> Any:
    """Audit of the current model on the large fresh draw. Computed on first use and
    kept until the next training run."""
    with _population_lock:
        if _model_state.get("fairness_population") is None:
            from creditlens.data.generator import DatasetConfig, generate_credit_dataset
            from creditlens.fairness.metrics import FairnessAuditor

            frame = generate_credit_dataset(DatasetConfig(n_samples=POPULATION_N, seed=POPULATION_SEED))
            engineer, model = _model_state["engineer"], _model_state["model"]
            p = np.concatenate([model.predict_proba(engineer.transform(frame.iloc[at:at + POPULATION_CHUNK])[0])[:, 1]
                                for at in range(0, POPULATION_N, POPULATION_CHUNK)])
            _model_state["fairness_population"] = FairnessAuditor().audit(
                frame["default"].to_numpy(), p, frame["demographic_group"].to_numpy())
        return _model_state["fairness_population"]


def _population_statement(report: Any) -> str:
    """One recommendation: what the large draw says the audit set could not."""
    parts = [f"Large fresh draw ({POPULATION_N:,} applicants from the same generator): "
             f"{report.overall_verdict}."]
    for name, check in report.checks.items():
        if check.verdict != "PASS":
            parts.append(f"{name} {check.value:.3f} (95% CI {check.ci_low:.3f}-{check.ci_high:.3f}, "
                         f"threshold {check.threshold:g}) reads {check.verdict}.")
    excludes_zero = {g: m for g, m in report.group_metrics.items()
                     if m.calibration_ci[0] > 0 or m.calibration_ci[1] < 0}
    if excludes_zero:
        g, m = max(excludes_zero.items(), key=lambda kv: abs(kv[1].calibration_in_the_large))
        direction = "under-predicts" if m.calibration_in_the_large > 0 else "over-predicts"
        parts.append(f"Calibration: the model {direction} group {g}'s risk, observed "
                     f"{m.default_rate:.3f} vs mean predicted {m.mean_predicted_default:.3f} "
                     f"(gap {m.calibration_in_the_large:+.3f}; 95% CI {m.calibration_ci[0]:+.3f} to "
                     f"{m.calibration_ci[1]:+.3f}).")
    return " ".join(parts)


@app.get("/fairness/report")
def get_fairness_report():
    """Return the fairness audit from the last training run, on its held-out set,
    with the same audit on a large fresh draw under ``population``."""
    if "fairness_report" not in _model_state:
        raise HTTPException(status_code=404, detail="No fairness report available.")
    payload = _fairness_payload(_model_state["fairness_report"])
    payload["score_bands"] = _model_state["score_bands"]
    population = _population_report()
    payload["recommendations"] = [*payload["recommendations"], _population_statement(population)]
    payload["population"] = {"n_samples": POPULATION_N, "seed": POPULATION_SEED,
                             **_fairness_payload(population)}
    return payload


@app.post("/train")
def train_endpoint(request: TrainRequest = TrainRequest()):
    """
    Train the credit scoring model on synthetic data.

    Disabled (403) unless the server was started with CREDITLENS_ENABLE_TRAIN=1:
    it is unauthenticated and replaces the model for every later caller.

    Thin HTTP wrapper that enforces bounded inputs (n_samples in [2000, 50000])
    via TrainRequest before delegating to the internal train_model() callable.
    With no body it rebuilds exactly the startup model (DEFAULT_TRAIN_SAMPLES,
    DEFAULT_SEED), so documented scores reproduce after a retrain.
    An out-of-range n_samples is rejected with 422 by Pydantic, preventing an
    unauthenticated CPU/memory DoS via a huge synthetic dataset + model fit.
    """
    if not _train_enabled():
        raise HTTPException(
            status_code=403,
            detail=(f"POST /train is disabled. It is unauthenticated and replaces the served "
                    f"model for every caller; start the server with {TRAIN_ENV_FLAG}=1 to enable it."),
        )
    return train_model(n_samples=request.n_samples, seed=request.seed)


def train_model(n_samples: int = DEFAULT_TRAIN_SAMPLES, seed: int = DEFAULT_SEED):
    """
    Train the credit scoring model on synthetic data.

    This is the internal callable used by the /train endpoint, the startup
    hook, and the lazy-train fallback. It generates synthetic data, trains the
    logistic scorer, evaluates, runs fairness audit, and initializes the SHAP explainer.
    """
    from creditlens.data.generator import generate_credit_dataset, DatasetConfig, train_test_split
    from creditlens.features.engineering import FeatureEngineer
    from creditlens.models.scorer import LogisticScorer, evaluate_model
    from creditlens.fairness.metrics import FairnessAuditor

    config = DatasetConfig(n_samples=n_samples, seed=seed)
    df = generate_credit_dataset(config)
    train_df, test_df = train_test_split(df, test_size=0.20, seed=seed)

    engineer = FeatureEngineer(scale=True)
    X_train, y_train = engineer.fit_transform(train_df)
    X_test, y_test = engineer.transform(test_df)

    # Sign-constrained logistic regression, the correctly specified form for the
    # synthetic label (a logistic model additive in these features). Depth-1 XGBoost
    # is a step function that lumps FICO's whole effect at split points (on Linux x86-64,
    # FICO 487 scores 414 / Denied, 488 scores 597 / Review, and 350 scores the same as 487).
    # README "Why logistic regression" compares them.
    model = LogisticScorer().fit(X_train, y_train)

    evaluation = evaluate_model(model, X_test, y_test, "LogisticScorer")

    # No threshold argument: audit exactly the decision rule /score applies.
    auditor = FairnessAuditor()
    y_proba_test = model.predict_proba(X_test)[:, 1]
    fairness_report = auditor.audit(
        y_test.values, y_proba_test,
        test_df["demographic_group"].values
    )

    # Explainer for THIS model. If it cannot be built, the new model serves
    # without one (/explain -> 503), never next to the previous model's explainer,
    # which would explain a model that does not produce the score.
    explainer = None
    try:
        from creditlens.explainability.shap_explainer import SHAPExplainer
        # Attribution over the applicant's inputs (features derived inside the
        # value function), against 1,000 of the raw training rows.
        explainer = SHAPExplainer.over_inputs(model, engineer)
        explainer.fit(train_df)
    except Exception as e:
        explainer = None
        logger.warning(f"Explainer setup failed; /explain unavailable for this model: {e}")

    # Replace everything at once: model, engineer, reports and explainer
    # always describe the same training run.
    _model_state.update({
        "model": model,
        "engineer": engineer,
        "evaluation": evaluation,
        "fairness_report": fairness_report,
        "fairness_population": None,
        "score_bands": score_band_table(probability_to_score(y_proba_test), y_test.values),
        "explainer": explainer,
    })

    return {
        "status": "trained",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc_roc": round(evaluation.auc_roc, 4),
        "fairness_pass": fairness_report.overall_pass,
        "fairness_verdict": fairness_report.overall_verdict,
    }


def _ensure_model_loaded():
    if "model" not in _model_state:
        # Auto-train on first request
        train_model()


def _get_decision(score: int, applicant: ApplicantFeatures) -> tuple[str, str]:
    decision = score_to_decision(score)
    return decision, DECISION_REASONS[decision]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("creditlens.api.app:app", host="0.0.0.0", port=8002, reload=True)
