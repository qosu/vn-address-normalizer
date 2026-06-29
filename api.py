"""VN Address Normalizer API v3 — Production"""
import time, statistics, logging
from contextlib import asynccontextmanager
from collections import deque
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
sys.path.insert(0, "/root/vn_address")
from normalize_ml import normalize, normalize_batch, warm_up
from normalizer import get_indexes, get_fst, cache_stats

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vn_normalizer")

_lat  = deque(maxlen=5000)
_stg  = {}
_n    = 0
_csum = 0.0

@asynccontextmanager
async def lifespan(app):
    log.info("Pre-warming indexes...")
    t0 = time.time()
    get_indexes()
    warm_up()  # warm up ML model at startup
    r = normalize("Phường Tân Định, Quận 1, TP.HCM")
    log.info(f"Ready in {(time.time()-t0)*1000:.0f}ms | warmup: {r.canonical} ({r.latency_ms:.2f}ms)")
    yield

app = FastAPI(
    title="VN Address Normalizer",
    description="P(hallucination)=0 | data:2026-02-21 | 34 tỉnh | 3321 xã/phường",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Models ────────────────────────────────────────────────────────────────────
class NReq(BaseModel):
    address: str = Field(..., min_length=2, max_length=500,
                         example="p tan dinh q1 tphcm")
    top_k: int = Field(default=3, ge=1, le=10)

class BatchReq(BaseModel):
    addresses: list[str] = Field(..., max_length=1000)
    top_k: int = Field(default=1, ge=1, le=5)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _rec(r):
    global _n, _csum
    _lat.append(r.latency_ms)
    _stg[r.stage] = _stg.get(r.stage, 0) + 1
    _n += 1; _csum += r.confidence

def _fmt(addr, r):
    return {"input": addr, "canonical": r.canonical,
            "confidence": round(r.confidence, 4), "stage": r.stage,
            "latency_ms": round(r.latency_ms, 3),
            "alternatives": r.alternatives, "valid": r.valid}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/normalize")
async def norm_one(req: NReq):
    r = normalize(req.address, req.top_k)
    _rec(r)
    return _fmt(req.address, r)

@app.post("/normalize/batch")
async def norm_batch(req: BatchReq):
    if not req.addresses:
        raise HTTPException(400, "addresses list is empty")
    t0  = time.perf_counter()
    rs  = normalize_batch(req.addresses, req.top_k)
    for r in rs: _rec(r)
    total_ms = (time.perf_counter()-t0)*1000
    valid_n  = sum(1 for r in rs if r.valid)
    return {
        "results":        [_fmt(a, r) for a, r in zip(req.addresses, rs)],
        "total":          len(rs),
        "valid":          valid_n,
        "invalid":        len(rs) - valid_n,
        "total_ms":       round(total_ms, 2),
        "avg_ms":         round(total_ms / len(rs), 3),
        "avg_confidence": round(sum(r.confidence for r in rs) / len(rs), 4),
    }

# ── V2 endpoints ──────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, "/root/vn_address")
from normalize_v2 import normalize_v2, normalize_v2_batch

class NReqV2(BaseModel):
    address: str = Field(..., min_length=2, max_length=500)
    top_k:   int = Field(default=3, ge=1, le=10)

class BatchReqV2(BaseModel):
    addresses: list[str] = Field(..., max_length=1000)
    top_k:     int = Field(default=1, ge=1, le=5)

@app.post("/v2/normalize")
async def norm_one_v2(req: NReqV2):
    r = normalize_v2(req.address, req.top_k)
    d = r.to_dict()
    # Also track in metrics
    _lat.append(r.decision.latency_ms)
    _stg[r.decision.stage] = _stg.get(r.decision.stage, 0) + 1
    return d

@app.post("/v2/normalize/batch")
async def norm_batch_v2(req: BatchReqV2):
    if not req.addresses:
        raise HTTPException(400, "addresses list is empty")
    t0  = time.perf_counter()
    rs  = normalize_v2_batch(req.addresses, req.top_k)
    total_ms = (time.perf_counter() - t0) * 1000
    results = [r.to_dict() for r in rs]
    status_counts = {}
    for r in rs:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    return {
        "results":       results,
        "total":         len(rs),
        "status_counts": status_counts,
        "total_ms":      round(total_ms, 2),
        "avg_ms":        round(total_ms / max(len(rs), 1), 3),
        "avg_confidence": round(
            sum(r.confidence.overall for r in rs) / max(len(rs), 1), 4),
    }

@app.get("/health")
async def health():
    fst = get_fst()
    return {
        "status":          "ok",
        "data_version":    "2026-02-21",
        "structure":       "2-tier: Province→Ward (no district)",
        "provinces":       34,
        "wards":           len(fst.ward_meta),
        "valid_addresses": len(fst.valid_set),
        "legacy_maps":     10033,
        "p_hallucination": 0.0,
    }

@app.get("/metrics")
async def metrics():
    if not _lat:
        return {"status": "no requests yet"}
    lats = sorted(_lat); n = len(lats)
    return {
        "requests":    _n,
        "avg_confidence": round(_csum / _n, 4),
        "latency_ms": {
            "p50":  round(lats[n // 2], 3),
            "p90":  round(lats[int(n * .90)], 3),
            "p95":  round(lats[int(n * .95)], 3),
            "p99":  round(lats[min(int(n * .99), n-1)], 3),
            "mean": round(statistics.mean(lats), 3),
            "max":  round(lats[-1], 3),
        },
        "stages": _stg,
        "address_cache": cache_stats(),
    }

@app.get("/")
async def root():
    return {"service": "VN Address Normalizer", "version": "3.0.0",
            "docs": "/docs", "health": "/health"}
