"""VN Address Normalizer — Production v4 FINAL"""
import re, time, pickle, json, sys
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, OrderedDict
from pathlib import Path
from unidecode import unidecode
from rapidfuzz import fuzz, process as rf_process

sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_DIR = Path(__file__).resolve().parent
FST_CACHE = CACHE_DIR / "fst_cache.pkl"
IDX_CACHE = CACHE_DIR / "idx_cache.pkl"

@dataclass
class NormalizeResult:
    canonical:    str
    confidence:   float
    stage:        str
    latency_ms:   float
    alternatives: list = field(default_factory=list)
    valid:        bool = True

PROVINCE_PRIOR = {
    79:0.280,1:0.160,31:0.065,48:0.045,75:0.040,
    91:0.035,92:0.030,82:0.025,38:0.022,40:0.020,
    52:0.018,80:0.017,86:0.016,25:0.015,37:0.015,
    24:0.014,33:0.013,66:0.012,68:0.012,96:0.011,
}
def _prior(c): return PROVINCE_PRIOR.get(c, 0.008)

def _slug(s): return unidecode(s).lower().strip()

def _clean(t):
    t = t.strip()
    t = re.sub(r'[\t\n\r]+', ' ', t)
    t = re.sub(r'(?<=[\w\u00C0-\u024F])-(?=[\w\u00C0-\u024F])', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t

_OLD_PROV = {
    "ha giang":"tuyen quang","yen bai":"lao cai","bac kan":"thai nguyen",
    "vinh phuc":"phu tho","hoa binh":"phu tho","bac giang":"bac ninh",
    "thai binh":"hung yen","hai duong":"hai phong","ha nam":"ninh binh",
    "nam dinh":"ninh binh","quang binh":"quang tri","quang nam":"da nang",
    "kon tum":"quang ngai","binh dinh":"gia lai","phu yen":"dak lak",
    "ninh thuan":"khanh hoa","dak nong":"dak lak","binh phuoc":"dong nai",
    "binh duong":"ho chi minh","ba ria vung tau":"ho chi minh",
    "long an":"tay ninh","tien giang":"tay ninh",
    "ben tre":"vinh long","tra vinh":"vinh long",
    "dong thap":"an giang","kien giang":"an giang",
    "hau giang":"can tho","soc trang":"ca mau","bac lieu":"ca mau",
    "hcm":"ho chi minh","tphcm":"ho chi minh","saigon":"ho chi minh",
    "sai gon":"ho chi minh","hanoi":"ha noi","thua thien hue":"hue",
    "tt hue":"hue","brvt":"ho chi minh","vung tau":"ho chi minh",
}

# ── Regexes (all operate on SLUG/ASCII text) ──────────────────────────────────
# NOTE: _WARD_PFX and _PROV_PFX operate on raw Vietnamese text (comma-split path)
_WARD_PFX = re.compile(
    r'^(ph\u01b0\u1eddng|phuong|ph\.|p\.|x\xe3|xa|x\.|'
    r'\u0111\u1eb7c\s*khu|dk\.?)\s*', re.I)
_PROV_PFX = re.compile(
    r'^(t\u1ec9nh|tinh|th\xe0nh\s*ph\u1ed1|thanh\s*pho|tp\.?|t\.p\.?)\s*', re.I)
_DIST_PFX = re.compile(
    r'^(qu\u1eadn|quan|q\.?|huy\u1ec7n|huyen|h\.?|tx\.?)\s*', re.I)
_NUM_STR  = re.compile(r'^(\d+[a-z]?(?:/\d+[a-z]?)*)[\s,]+(.+)', re.I)

# No-comma patterns — operate on SLUG text (unidecode+lower)
_NC_PROV = re.compile(
    r'\b(tphcm|hcm|hanoi|saigon|sai gon|'
    r'ho chi minh|hai phong|da nang|can tho|hue|'
    r'tp\s+[\w\s]{1,20}|tinh\s+[\w\s]{1,20})\b', re.I)
_NC_DIST = re.compile(
    r'\b(q\.?\s*\d+|quan\s*\d+|h\.\s*\w+|huyen\s+\w+)\b', re.I)
# Operates on slug text — no diacritics issue
_NC_WARD = re.compile(
    r'^(phuong|xa|tt|p\.\s*|x\.\s*)([\w][\w\s]*)', re.I)

# Numbered ward guard — catches "14", "14 quan 5", "phuong 14 ..."
_NUMBERED = re.compile(r'^(phuong\s+)?\d{1,3}(\s.*)?$')


def _extract(raw: str) -> dict:
    """Parse comma-separated address."""
    parts = [p.strip() for p in re.split(r'[,;]', raw) if p.strip()]
    r = {"ward": None, "province": None, "district_hint": None}
    if parts:
        m = _NUM_STR.match(parts[0])
        if m: parts = [m.group(2)] + parts[1:]
    for part in parts:
        if   _PROV_PFX.match(part): r["province"]      = _PROV_PFX.sub('',part).strip()
        elif _DIST_PFX.match(part): r["district_hint"] = part
        elif _WARD_PFX.match(part): r["ward"]          = _WARD_PFX.sub('',part).strip()
        elif not r["ward"]:         r["ward"]           = part
    if not r["province"] and len(parts) >= 2:
        r["province"] = parts[-1]
    return r


def _parse_no_comma(raw: str) -> dict:
    """Parse space-only address — works on slug to avoid diacritic issues."""
    r = {"ward": None, "province": None, "district_hint": None}
    # Slug first: removes diacritics → regex handles cleanly
    text = _slug(raw)

    m = _NC_PROV.search(text)
    if m:
        r["province"] = m.group(0)
        text = (text[:m.start()] + " " + text[m.end():]).strip()

    m = _NC_DIST.search(text)
    if m:
        r["district_hint"] = m.group(0)
        text = (text[:m.start()] + " " + text[m.end():]).strip()

    text = text.strip()
    m = _NC_WARD.match(text)
    # group(2) = ward name after prefix
    r["ward"] = m.group(2).strip() if m else text
    return r


# ── FST ───────────────────────────────────────────────────────────────────────
_FST = None
def get_fst():
    global _FST
    if _FST is None:
        from fst import load
        _FST = load()
    return _FST


# ── Indexes ───────────────────────────────────────────────────────────────────
class Indexes:
    __slots__ = ('fst','slug_to_canon','short_to_canon','prov_slug',
                 'prov_wards','all_canons','all_canons_slug',
                 'legacy_idx','_c2pc')

    def __init__(self, fst):
        self.fst            = fst
        self.slug_to_canon  = defaultdict(list)
        self.short_to_canon = defaultdict(list)
        self.prov_slug      = {}
        self.prov_wards     = defaultdict(list)
        self._c2pc          = {}   # canonical → province_code  O(1)

        for wc, m in fst.ward_meta.items():
            c  = m["canonical"]
            pc = m["province_code"]
            sh = m.get("short_codename","").replace("_"," ")
            self._c2pc[c] = pc

            full = _slug(m["name"])              # e.g. "xa thuong lam"
            self.slug_to_canon[full].append(c)
            self.prov_wards[pc].append((full, c))

            # Also index by name-only (strip ward type prefix)
            name_only = re.sub(r'^(xa|phuong|thi tran|dac khu)\s+',
                               '', full).strip()
            if name_only != full:
                self.slug_to_canon[name_only].append(c)
                self.prov_wards[pc].append((name_only, c))

            if sh: self.short_to_canon[_slug(sh)].append(c)

        for pc, m in fst.prov_meta.items():
            for sl in [_slug(m["name"]),
                       m.get("codename","").replace("_"," ")]:
                self.prov_slug[sl] = pc

        self.all_canons      = list(fst.valid_set)
        self.all_canons_slug = [_slug(c) for c in self.all_canons]
        self.legacy_idx      = self._build_legacy()

    def _build_legacy(self):
        from vietnam_provinces.legacy import Ward as LW, WardCode as LWC
        from vietnam_provinces._ward_conversion_2025 import OLD_TO_NEW
        from vietnam_provinces import NESTED_DIVISIONS_JSON_PATH
        with open(NESTED_DIVISIONS_JSON_PATH) as f:
            data = json.load(f)
        wm = {w["code"]: f"{w['name']}, {p['name']}"
              for p in data for w in p.get("wards",[])}
        idx = defaultdict(list)
        for wc in LWC:
            lw    = LW.from_code(wc)
            entry = OLD_TO_NEW.get(wc.value)
            if not entry: continue
            new_canons = [wm[nw.code] for nw in entry.new_wards
                          if nw.code in wm]
            if new_canons:
                full = _slug(str(lw))            # "xa thuong lam"
                idx[full].extend(new_canons)
                name_only = re.sub(
                    r'^(phuong|xa|thi tran|p\.|x\.)\s*',
                    '', full).strip()
                if name_only != full:
                    idx[name_only].extend(new_canons)
        return dict(idx)

    def resolve_province(self, ts: str) -> Optional[int]:
        # Strip prefix: "tinh ha giang" → "ha giang"
        ts2 = re.sub(r'^(tinh|tp\.?\s*|thanh pho)\s+', '', ts).strip()
        for key in ([ts, ts2] if ts2 != ts else [ts]):
            if key in self.prov_slug: return self.prov_slug[key]
            alias = _OLD_PROV.get(key)
            if alias: return self.prov_slug.get(alias)
        for k, v in self.prov_slug.items():
            if ts2 and (ts2 in k or k in ts2): return v
        return None

    def pc(self, canon: str) -> int:
        return self._c2pc.get(canon, -1)

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({k: (dict(getattr(self,k))
                             if isinstance(getattr(self,k), defaultdict)
                             else getattr(self,k))
                         for k in self.__slots__ if k != 'fst'}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, fst):
        with open(path, 'rb') as f: d = pickle.load(f)
        obj = object.__new__(cls)
        obj.fst            = fst
        obj.slug_to_canon  = defaultdict(list, d["slug_to_canon"])
        obj.short_to_canon = defaultdict(list, d["short_to_canon"])
        obj.prov_slug      = d["prov_slug"]
        obj.prov_wards     = defaultdict(list, d["prov_wards"])
        obj.all_canons     = d["all_canons"]
        obj.all_canons_slug= d["all_canons_slug"]
        obj.legacy_idx     = d["legacy_idx"]
        obj._c2pc          = d.get("_c2pc", {})
        return obj


_IDX: Optional[Indexes] = None

def get_indexes() -> Indexes:
    global _IDX
    if _IDX is not None: return _IDX
    fst = get_fst()
    if IDX_CACHE.exists():
        try:
            t0 = time.time()
            _IDX = Indexes.load(IDX_CACHE, fst)
            print(f"Indexes loaded {(time.time()-t0)*1000:.0f}ms "
                  f"({len(_IDX.legacy_idx)} legacy)")
            return _IDX
        except Exception as e:
            print(f"Cache miss ({e}), rebuilding")
    t0 = time.time()
    _IDX = Indexes(fst)
    _IDX.save(IDX_CACHE)
    print(f"Indexes built {(time.time()-t0)*1000:.0f}ms "
          f"({len(_IDX.legacy_idx)} legacy)")
    return _IDX


# ── LRU ───────────────────────────────────────────────────────────────────────
class LRU:
    def __init__(self, n=50_000):
        self._c = OrderedDict(); self._n = n
        self.hits = self.miss = 0
    def get(self, k):
        if k in self._c:
            self._c.move_to_end(k); self.hits += 1; return self._c[k]
        self.miss += 1; return None
    def put(self, k, v):
        self._c[k] = v; self._c.move_to_end(k)
        if len(self._c) > self._n: self._c.popitem(last=False)
    def stats(self):
        t = self.hits + self.miss
        return {"size": len(self._c), "hits": self.hits, "misses": self.miss,
                "hit_rate": round(self.hits/t, 4) if t else 0}

_LRU = LRU(50_000)


# ── Core normalize ────────────────────────────────────────────────────────────
def normalize(raw: str, top_k: int = 3) -> NormalizeResult:
    t0  = time.perf_counter()
    raw = _clean(raw)

    cached = _LRU.get(raw)
    if cached:
        return NormalizeResult(
            cached.canonical, cached.confidence, "cache",
            (time.perf_counter()-t0)*1000, cached.alternatives, cached.valid)

    fst = get_fst()
    idx = get_indexes()

    def ret(canon, conf, stage, alts=None):
        r = NormalizeResult(canon, conf, stage,
                            (time.perf_counter()-t0)*1000,
                            alts or [], bool(canon and conf > 0.5))
        if r.valid: _LRU.put(raw, r)
        return r

    # S1: exact
    if fst.accepts(raw): return ret(raw, 1.0, "exact")

    # S2: slug exact  (handles full-form slugs)
    rs   = _slug(raw)
    hits = idx.slug_to_canon.get(rs, [])
    if hits: return ret(hits[0], 0.99, "slug_exact", hits[1:top_k])

    # Parse
    comps = _extract(raw) if ',' in raw else _parse_no_comma(raw)

    # Province resolution
    rpc = None
    for fv in [comps.get("province"), comps.get("district_hint")]:
        if fv and not rpc:
            rpc = idx.resolve_province(_slug(fv))

    # Ward hint — already slug (from _parse_no_comma) or raw text (from _extract)
    ward_raw = comps.get("ward") or raw
    wh = _slug(ward_raw) if ward_raw else rs

    # Guard: numbered ward = pre-2025 invalid (e.g. "14", "phuong 14", "14 quan 5")
    if _NUMBERED.match(wh):
        return ret("", 0.0, "failed")

    # S3: multi-source O(1) lookup
    d = idx.slug_to_canon.get(wh, [])
    s = idx.short_to_canon.get(wh, [])
    l = idx.legacy_idx.get(wh, []) if len(wh) > 2 else []
    all_h = list(dict.fromkeys(d + s + l))

    if all_h:
        stage = "legacy_match" if (l and not d) else "component_match"
        if rpc:
            in_p = [c for c in all_h if idx.pc(c) == rpc]
            if in_p: return ret(in_p[0], 0.95, stage, in_p[1:top_k])
        ranked = sorted(all_h, key=lambda c: _prior(idx.pc(c)), reverse=True)
        return ret(ranked[0], 0.88, stage+"_prior", ranked[1:top_k])

    # S4: rapidfuzz on narrowed pool
    if rpc and idx.prov_wards.get(rpc):
        pool_s = [w for w, _ in idx.prov_wards[rpc]]
        pool_c = [c for _, c in idx.prov_wards[rpc]]
        # Deduplicate (prov_wards has both "xa thuong lam" and "thuong lam" entries)
        seen = {}
        for w, c in zip(pool_s, pool_c):
            if c not in seen: seen[c] = w
        pool_c = list(seen.keys())
        pool_s = [seen[c] for c in pool_c]
    else:
        pool_s, pool_c = idx.all_canons_slug, idx.all_canons

    if not pool_s: return ret("", 0.0, "failed")

    res = rf_process.extract(wh, pool_s,
                             scorer=fuzz.token_sort_ratio, limit=top_k+1)
    if res:
        bs   = res[0][1] / 100.0
        bc   = pool_c[res[0][2]]
        alts = [pool_c[r[2]] for r in res[1:top_k]]
        conf = min(bs * _prior(idx.pc(bc))**0.15, 0.95)
        if conf > 0.50: return ret(bc, conf, "fuzzy", alts)
        return ret(bc, conf * 0.5, "fuzzy_low_conf")

    return ret("", 0.0, "failed")


def normalize_batch(addrs: list, top_k: int = 1) -> list:
    return [normalize(a, top_k) for a in addrs]

def cache_stats() -> dict:
    return _LRU.stats()
