"""
ML-enhanced normalize — used by api.py, normalize_v2.py, test_regression.py.
Single source of truth for the ML fallback logic.
"""
import time, logging, sys
sys.path.insert(0, "/root/vn_address")
from normalizer import normalize as _rb_normalize, normalize_batch as _rb_batch

log = logging.getLogger("vn_normalizer")

_ml_model   = None
_ML_THRESH  = 0.75
_ML_STAGES  = {"fuzzy", "fuzzy_low_conf", "street_only"}

def _get_ml():
    global _ml_model
    if _ml_model is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ml_inference_v3", "/root/vn_address/ml_inference_v3.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ml_model = mod
        log.info("ML model loaded.")
    return _ml_model

class _MLResult:
    def __init__(self, canonical, confidence, latency_ms, alternatives):
        self.canonical    = canonical
        self.confidence   = confidence
        self.stage        = "ml"
        self.latency_ms   = latency_ms
        self.alternatives = alternatives
        self.valid        = bool(canonical)

def normalize(address: str, top_k: int = 3):
    t0 = time.perf_counter()
    r  = _rb_normalize(address, top_k)
    should_ml = (
        not r.valid or
        r.confidence < _ML_THRESH or
        r.stage in _ML_STAGES
    )
    if should_ml:
        try:
            ml = _get_ml().normalize(address)
            if ml.get("valid") and ml.get("canonical"):
                ms = (time.perf_counter() - t0) * 1000
                return _MLResult(
                    canonical   = ml["canonical"],
                    confidence  = 0.88 if ml.get("ward_hint") else 0.78,
                    latency_ms  = ms,
                    alternatives= r.alternatives,
                )
        except Exception as e:
            log.warning(f"ML fallback error: {e}")
    return r

def normalize_batch(addresses, top_k: int = 1):
    return [normalize(a, top_k) for a in addresses]

def warm_up():
    _get_ml()
    normalize("Phường Tân Định, Quận 1, TP.HCM")
    log.info("ML warm-up done.")
