"""MicroShield FastAPI application entry point.

The simulation algorithms live in `app/core/` as pure Python modules that can
be imported and tested without starting this server (see `backend/tests/`).
This module only performs HTTP concerns: validation (via Pydantic schemas),
CORS, and routing.
"""

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import MODEL_VERSION, __version__
from app.config import CORS_ALLOW_ALL, CORS_ALLOW_ORIGINS
from app.core.branch_predictors import BranchConfig, BranchEvent, simulate
from app.core.cache import CacheConfig, parse_accesses, simulate_cache
from app.core.pipeline import PipelineConfig, simulate_pipeline

app = FastAPI(
    title="MicroShield API",
    description=(
        "CPU microarchitecture simulation services: 5-stage pipeline, branch "
        "predictors, cache memory, and microarchitectural security demos. "
        "All results are produced by simplified educational models — never "
        "measurements from real hardware."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ALL else CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Liveness probe used by deployment checks and the frontend status dot."""
    return {
        "status": "ok",
        "version": __version__,
        "model_version": MODEL_VERSION,
    }


class BranchRequest(BaseModel):
    outcomes: List[Literal["T", "N"]] = Field(max_length=500)
    predictor: Literal["always_taken", "always_not_taken", "one_bit", "two_bit", "ghr"] = "two_bit"
    initial_prediction: Literal["taken", "not_taken"] = "taken"
    initial_counter: int = Field(default=2, ge=0, le=3)
    initial_ghr: int = Field(default=0, ge=0)
    history_bits: int = Field(default=4, ge=1, le=8)
    misprediction_penalty: int = Field(default=2, ge=0, le=20)


class BranchComparisonRequest(BaseModel):
    outcomes: List[Literal["T", "N"]] = Field(max_length=500)
    predictors: List[Literal["always_taken", "always_not_taken", "one_bit", "two_bit", "ghr"]] = ["always_taken", "always_not_taken", "one_bit", "two_bit"]
    initial_prediction: Literal["taken", "not_taken"] = "taken"
    initial_counter: int = Field(default=2, ge=0, le=3)
    misprediction_penalty: int = Field(default=2, ge=0, le=20)


class PipelineRequest(BaseModel):
    program: str = Field(max_length=20000)
    forwarding: bool = True
    branch_policy: Literal["static_not_taken", "static_taken", "one_bit", "two_bit", "ghr"] = "static_not_taken"
    memory_latency: int = Field(default=1, ge=1, le=8)


class CacheRequest(BaseModel):
    accesses: str = Field(max_length=30000)
    capacity: int = Field(default=256, ge=4, le=65536)
    line_size: int = Field(default=16, ge=4, le=1024)
    associativity: int = Field(default=4, ge=1, le=64)
    replacement: Literal["lru", "fifo"] = "lru"
    hit_latency: int = Field(default=1, ge=1, le=100)
    miss_latency: int = Field(default=20, ge=1, le=1000)


def _simulate_or_400(call):
    try:
        return call()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/branch-predictor/simulate", tags=["simulation"])
def branch_predictor(request: BranchRequest) -> dict:
    events = [BranchEvent(label="B0", outcome=value == "T") for value in request.outcomes]
    config = BranchConfig(
        predictor=request.predictor,
        initial_prediction=request.initial_prediction,
        initial_counter=request.initial_counter,
        initial_ghr=request.initial_ghr,
        history_bits=request.history_bits,
        misprediction_penalty=request.misprediction_penalty,
    )
    return _simulate_or_400(lambda: simulate(events, config).to_dict())


@app.post("/api/branch-predictor/compare", tags=["simulation"])
def compare_predictors(request: BranchComparisonRequest) -> dict:
    events = [BranchEvent(label="B0", outcome=value == "T") for value in request.outcomes]
    def run():
        results = []
        for predictor in request.predictors:
            config = BranchConfig(predictor=predictor, initial_prediction=request.initial_prediction,
                                  initial_counter=request.initial_counter,
                                  misprediction_penalty=request.misprediction_penalty)
            results.append(simulate(events, config).to_dict())
        return {"results": results}
    return _simulate_or_400(run)


@app.post("/api/pipeline/simulate", tags=["simulation"])
def pipeline(request: PipelineRequest) -> dict:
    config = PipelineConfig(forwarding=request.forwarding, branch_policy=request.branch_policy,
                            memory_latency=request.memory_latency)
    return _simulate_or_400(lambda: simulate_pipeline(request.program, config).to_dict())


@app.post("/api/cache/simulate", tags=["simulation"])
def cache(request: CacheRequest) -> dict:
    config = CacheConfig(capacity=request.capacity, line_size=request.line_size,
                         associativity=request.associativity, replacement=request.replacement)
    def run():
        result = simulate_cache(parse_accesses(request.accesses), config).to_dict()
        steps = result["steps"]
        total = sum(request.hit_latency if step["hit"] else request.miss_latency for step in steps)
        result["hit_latency"] = request.hit_latency
        result["miss_latency"] = request.miss_latency
        result["total_cycles"] = total
        result["amat"] = total / len(steps) if steps else None
        return result
    return _simulate_or_400(run)


@app.post("/api/security/cache-timing", tags=["security"])
def cache_timing(hit_latency: int = 1, miss_latency: int = 20, samples: int = 8) -> dict:
    if not 1 <= hit_latency <= 100 or not hit_latency <= miss_latency <= 1000 or not 1 <= samples <= 100:
        raise HTTPException(status_code=400, detail="Require 1<=hit_latency<=100, hit_latency<=miss_latency<=1000, and 1<=samples<=100.")
    warm = [hit_latency] * samples
    mixed = [hit_latency if i % 2 == 0 else miss_latency for i in range(samples)]
    return {"model": "synthetic cache timing", "warm": warm, "mixed": mixed,
            "warm_mean": sum(warm) / len(warm), "mixed_mean": sum(mixed) / len(mixed),
            "notes": ["Synthetic educational timings; no hardware is measured."]}


@app.post("/api/security/speculation", tags=["security"])
def speculation(barrier: bool = False) -> dict:
    return {"barrier": barrier, "architectural_state": "wrong-path writes rolled back",
            "cache_signal": not barrier, "overhead_cycles": 1 if barrier else 0,
            "timeline": ["branch predicted", "synthetic speculative work", "architectural rollback",
                         "modeled cache signal suppressed by barrier" if barrier else "synthetic cache footprint persists"],
            "notes": ["Toy model only; the barrier is not a complete real-world defense."]}
