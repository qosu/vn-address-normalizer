"""
Phase 0 — V2 Output Schema Layer
Wraps existing normalizer.py with structured output.
Import: from normalize_v2 import normalize_v2, NormalizeV2Result
"""
import re, time, sys
from dataclasses import dataclass, field
from typing import Optional
sys.path.insert(0, "/root/vn_address")

from normalize_ml import normalize as _rb_normalize
from normalize_ml import normalize_batch as _rb_batch_inner

# ── V2 Schema ─────────────────────────────────────────────────────────────────
@dataclass
class AddressComponents:
    house_number: Optional[str] = None
    street:       Optional[str] = None
    ward:         Optional[str] = None
    province:     Optional[str] = None

@dataclass
class LegacyDetected:
    ward:     Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None

@dataclass
class ComponentConfidence:
    overall:  float = 0.0
    province: float = 0.0
    ward:     float = 0.0
    street:   float = 0.0

@dataclass
class Candidates:
    ward:   list = field(default_factory=list)
    street: list = field(default_factory=list)

@dataclass
class Decision:
    stage:           str = ""
    reason:          str = ""
    latency_ms:      float = 0.0
    stage_latencies: dict = field(default_factory=dict)

@dataclass
class NormalizeV2Result:
    status:           str = "out_of_coverage"
    input:            str = ""
    normalized_text:  str = ""
    current:          AddressComponents = field(default_factory=AddressComponents)
    legacy_detected:  LegacyDetected    = field(default_factory=LegacyDetected)
    confidence:       ComponentConfidence = field(default_factory=ComponentConfidence)
    candidates:       Candidates         = field(default_factory=Candidates)
    decision:         Decision           = field(default_factory=Decision)
    warnings:         list = field(default_factory=list)

    def to_dict(self):
        return {
            "status": self.status,
            "input":  self.input,
            "normalized_text": self.normalized_text,
            "current": {
                "house_number": self.current.house_number,
                "street":       self.current.street,
                "ward":         self.current.ward,
                "province":     self.current.province,
            },
            "legacy_detected": {
                "ward":     self.legacy_detected.ward,
                "district": self.legacy_detected.district,
                "province": self.legacy_detected.province,
            },
            "confidence": {
                "overall":  round(self.confidence.overall,  4),
                "province": round(self.confidence.province, 4),
                "ward":     round(self.confidence.ward,     4),
                "street":   round(self.confidence.street,   4),
            },
            "candidates": {
                "ward":   self.candidates.ward,
                "street": self.candidates.street,
            },
            "decision": {
                "stage":           self.decision.stage,
                "reason":          self.decision.reason,
                "latency_ms":      round(self.decision.latency_ms, 3),
                "stage_latencies": {k: round(v,3)
                                    for k,v in self.decision.stage_latencies.items()},
            },
            "warnings": self.warnings,
        }

# ── Helpers ───────────────────────────────────────────────────────────────────
_NUM_HOUSE = re.compile(r'^(\d+[a-zA-Z]?(?:/\d+[a-zA-Z]?)?)\s+(.+)', re.I)
_LEGACY_STAGES = {"legacy_match"}
_FUZZY_STAGES  = {"fuzzy", "fuzzy_low_conf"}
_ML_STAGE      = {"ml"}
_INVALID_STAGES = {"failed"}

def _parse_canonical(canonical: str):
    """Parse 'Street, Ward, Province' → (house, street, ward, province)."""
    if not canonical:
        return None, None, None, None
    parts = [p.strip() for p in canonical.split(",")]
    if len(parts) == 3:
        street_raw, ward, province = parts
        m = _NUM_HOUSE.match(street_raw)
        if m:
            return m.group(1), m.group(2), ward, province
        return None, street_raw, ward, province
    elif len(parts) == 2:
        ward, province = parts
        return None, None, ward, province
    return None, None, None, None

def _extract_legacy_info(raw: str) -> LegacyDetected:
    """Try to detect old district/ward from raw input."""
    from unidecode import unidecode
    def slug(s): return unidecode(s).lower().strip()
    s = slug(raw)
    ld = LegacyDetected()
    # Old districts
    dist_m = re.search(r'\b(q\.?\s*\d+|quan\s*\d+|h\.\s*\w+|huyen\s+\w+|'
                       r'district\s*\d+)\b', s, re.I)
    if dist_m:
        ld.district = dist_m.group(0).strip()
    # Old ward names (anything after phuong/xa that didn't survive)
    ward_m = re.search(r'\b(?:phuong|p\.|p\s|xa|x\.)\s*([a-z0-9][a-z0-9\s]{1,30}?)(?=\s+(?:quan|q|huyen|tp|tinh)|$)', s, re.I)
    if ward_m:
        ld.ward = ward_m.group(1).strip()
    return ld

def _compute_confidence(r, house, street, ward, province) -> ComponentConfidence:
    """Derive per-component confidence from rule-based result."""
    base = r.confidence

    # Province confidence: high for exact/component/legacy, lower for fuzzy
    if r.stage in ("exact_match", "component_match", "legacy_match"):
        prov_conf = 0.98
    elif r.stage in _ML_STAGE:
        prov_conf = 0.95 if province else 0.0
    elif r.stage in _FUZZY_STAGES:
        prov_conf = min(0.85, base * 1.1)
    else:
        prov_conf = 0.0

    # Ward confidence
    if r.stage in ("exact_match", "legacy_match"):
        ward_conf = 0.96
    elif r.stage == "component_match":
        ward_conf = 0.93
    elif r.stage in _ML_STAGE:
        ward_conf = 0.88 if ward else 0.0
    elif r.stage in _FUZZY_STAGES:
        ward_conf = base
    else:
        ward_conf = 0.0

    # Street confidence: only meaningful if street is present
    if street:
        if r.stage == "exact_match":
            street_conf = 0.95
        elif r.stage in _ML_STAGE:
            street_conf = 0.82
        elif r.stage in _FUZZY_STAGES:
            street_conf = base * 0.85
        else:
            street_conf = base * 0.7
    else:
        street_conf = 0.0

    overall = base

    return ComponentConfidence(
        overall=overall, province=prov_conf,
        ward=ward_conf, street=street_conf
    )

def _compute_status(r, house, street, ward, province, conf: ComponentConfidence) -> tuple:
    """Returns (status, reason, warnings)."""
    warnings = []

    # Invalid
    if r.stage in _INVALID_STAGES or (not r.valid and not r.canonical):
        return "invalid", "no_valid_administrative_match", warnings

    if not province and not ward:
        return "out_of_coverage", "insufficient_information", warnings

    # Province-only result
    if province and not ward:
        return "out_of_coverage", "province_only_no_ward", warnings

    # Has ward + province — check confidence
    if ward and province:
        HIGH = 0.85  # threshold for "confident enough"
        if conf.ward >= HIGH and conf.province >= HIGH:
            if street:
                if conf.street >= 0.78:
                    return "exact", "high_confidence_all_components", warnings
                else:
                    warnings.append("street_score_below_threshold")
                    return "partial", "ward_province_exact_street_uncertain", warnings
            else:
                # ward-only: exact regardless of stage if confidence is high
                return "exact", "ward_province_exact", warnings
        elif conf.ward >= 0.65:
            # Only ambiguous if truly multiple close candidates AND low confidence
            if r.alternatives and len(r.alternatives) > 0 and conf.ward < 0.75:
                return "ambiguous", "multiple_ward_candidates", warnings
            return "partial", "ward_below_high_confidence", warnings
        else:
            if r.alternatives:
                return "ambiguous", "low_confidence_multiple_candidates", warnings
            return "partial", "low_confidence_result", warnings

    return "out_of_coverage", "incomplete_resolution", warnings

# ── Main function ──────────────────────────────────────────────────────────────
def normalize_v2(raw: str, top_k: int = 3) -> NormalizeV2Result:
    """Full v2 normalization with structured output."""
    t_total = time.perf_counter()
    result = NormalizeV2Result(input=raw)

    # Call existing normalizer (includes ML fallback)
    t_rb = time.perf_counter()
    r = _rb_normalize(raw, top_k)
    rb_ms = (time.perf_counter() - t_rb) * 1000

    result.decision.stage_latencies["normalizer"] = rb_ms

    # Parse canonical into components
    house, street, ward, province = _parse_canonical(r.canonical)
    result.current = AddressComponents(
        house_number=house, street=street,
        ward=ward,          province=province
    )
    result.normalized_text = r.canonical or ""

    # Extract legacy info from raw input
    result.legacy_detected = _extract_legacy_info(raw)

    # Per-component confidence
    result.confidence = _compute_confidence(r, house, street, ward, province)

    # Status + reason
    status, reason, warnings = _compute_status(
        r, house, street, ward, province, result.confidence)
    result.status   = status
    result.warnings = warnings

    # Candidates
    if r.alternatives:
        # Parse alternatives into structured candidates
        for alt in r.alternatives[:top_k]:
            _, a_street, a_ward, a_province = _parse_canonical(alt)
            if a_ward:
                result.candidates.ward.append({
                    "canonical": alt,
                    "ward": a_ward,
                    "province": a_province,
                })

    # Decision
    total_ms = (time.perf_counter() - t_total) * 1000
    result.decision = Decision(
        stage=r.stage,
        reason=reason,
        latency_ms=total_ms,
        stage_latencies={"normalizer": round(rb_ms, 3)},
    )

    return result


def normalize_v2_batch(addresses: list, top_k: int = 1) -> list:
    return [normalize_v2(a, top_k) for a in addresses]


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    tests = [
        "45 tran binh trong p tan dinh q1 hcm",
        "P. Ben Nghe Q1 HCM",
        "Phuong 14 Quan 5 HCM",
        "phuong ba dinh ha noi",
    ]
    for t in tests:
        r = normalize_v2(t)
        print(f"\n[{r.status.upper()}] {t!r}")
        print(f"  ward    : {r.current.ward}")
        print(f"  province: {r.current.province}")
        print(f"  street  : {r.current.street}")
        print(f"  conf    : province={r.confidence.province:.2f} "
              f"ward={r.confidence.ward:.2f} street={r.confidence.street:.2f}")
        print(f"  stage   : {r.decision.stage} {r.decision.latency_ms:.1f}ms")
        if r.warnings: print(f"  warns   : {r.warnings}")
