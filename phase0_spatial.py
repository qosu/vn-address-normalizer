"""
Phase 0: OSM Vietnam → street_ward_map.json
Strategy 1: addr: tag extraction (no spatial computation needed)
Strategy 2: Spatial join for streets not covered by Strategy 1

Run on server: python3 /root/vn_address/phase0_spatial.py
"""
import osmium, json, sys, time, re
from collections import defaultdict
from unidecode import unidecode
from pathlib import Path

PBF  = '/root/vietnam-latest.osm.pbf'
OUT  = '/root/vn_address/street_ward_map.json'
STAT = '/root/vn_address/street_ward_stats.json'
sys.path.insert(0, '/root/vn_address')

def slug(s): return unidecode(s).lower().strip()

# ── Load VBHC ground truth ────────────────────────────────────────────────────
print("Loading VBHC ground truth...", flush=True)
from fst import load as fst_load
fst = fst_load()

# Index 1: ward_slug → list of (canonical, province_code)
# Multiple wards can share the same name in different provinces
ward_slug_idx = defaultdict(list)
for wc, meta in fst.ward_meta.items():
    ws = slug(meta['name'])
    ward_slug_idx[ws].append((meta['canonical'], meta['province_code']))
    # Also without type prefix: "phuong tan dinh" → "tan dinh"
    name_only = re.sub(r'^(xa|phuong|thi tran|dac khu)\s+', '', ws).strip()
    if name_only != ws:
        ward_slug_idx[name_only].append((meta['canonical'], meta['province_code']))

# Index 2: province slug → province_code
prov_slug_idx = {}
for pc, meta in fst.prov_meta.items():
    for sl in [slug(meta['name']),
               meta.get('codename','').replace('_',' '),
               re.sub(r'^(tinh|thanh pho|tp\.?)\s*','', slug(meta['name'])).strip()]:
        prov_slug_idx[sl] = pc

# Legacy ward name lookup (pre-2025 names → new canonical)
from vietnam_provinces.legacy import Ward as LW, WardCode as LWC
from vietnam_provinces._ward_conversion_2025 import OLD_TO_NEW
legacy_name_idx = defaultdict(list)
for wc in LWC:
    lw    = LW.from_code(wc)
    entry = OLD_TO_NEW.get(wc.value)
    if not entry: continue
    new_canons = [fst.ward_meta[nw.code]['canonical']
                  for nw in entry.new_wards if nw.code in fst.ward_meta]
    if new_canons:
        key = re.sub(r'^(phuong|xa|thi tran|p\.|x\.)\s*', '', slug(str(lw))).strip()
        legacy_name_idx[key].extend(new_canons)

print(f"  Ward slug entries : {sum(len(v) for v in ward_slug_idx.values()):,}")
print(f"  Province entries  : {len(prov_slug_idx):,}")
print(f"  Legacy name keys  : {len(legacy_name_idx):,}")

# ── Province resolver (reuse logic from normalizer) ───────────────────────────
_OLD_PROV = {
    "ha giang":"tuyen quang","yen bai":"lao cai","bac kan":"thai nguyen",
    "vinh phuc":"phu tho","hoa binh":"phu tho","bac giang":"bac ninh",
    "thai binh":"hung yen","hai duong":"hai phong","ha nam":"ninh binh",
    "nam dinh":"ninh binh","quang binh":"quang tri","quang nam":"da nang",
    "kon tum":"quang ngai","binh dinh":"gia lai","phu yen":"dak lak",
    "ninh thuan":"khanh hoa","dak nong":"dak lak","binh phuoc":"dong nai",
    "binh duong":"ho chi minh","ba ria vung tau":"ho chi minh",
    "long an":"tay ninh","tien giang":"tay ninh","ben tre":"vinh long",
    "tra vinh":"vinh long","dong thap":"an giang","kien giang":"an giang",
    "hau giang":"can tho","soc trang":"ca mau","bac lieu":"ca mau",
    "hcm":"ho chi minh","tphcm":"ho chi minh","saigon":"ho chi minh",
    "sai gon":"ho chi minh","hanoi":"ha noi",
}

def resolve_province(raw_city: str):
    """raw city string → province_code or None"""
    ts = slug(raw_city)
    # Try progressively stripped versions
    ts2 = re.sub(r'^(tinh|tp\.?\s*|thanh pho)\s+', '', ts).strip()
    ts3 = re.sub(r'^tp\.?\s*', '', ts).strip()   # "tp.hcm" → "hcm"
    ts3 = re.sub(r'\.', '', ts3).strip()           # "tp.hcm" → "tphcm"
    keys = list(dict.fromkeys([ts, ts2, ts3, ts.replace('.','').replace(' ','')]))
    for key in keys:
        if key in prov_slug_idx: return prov_slug_idx[key]
        alias = _OLD_PROV.get(key)
        if alias and alias in prov_slug_idx: return prov_slug_idx[alias]
    for k, v in prov_slug_idx.items():
        if ts2 and len(ts2) > 2 and (ts2 in k or k in ts2): return v
    return None

def resolve_ward(ward_raw: str, province_code=None):
    """ward string + optional province_code → canonical or None"""
    ws = slug(ward_raw)
    # Strip type prefix
    name_only = re.sub(r'^(xa|phuong|thi tran|p\.|x\.)\s*', '', ws).strip()

    for key in ([ws, name_only] if name_only != ws else [ws]):
        candidates = ward_slug_idx.get(key, [])
        if candidates:
            if province_code:
                in_prov = [c for c, pc in candidates if pc == province_code]
                if in_prov: return in_prov[0]
            # Fallback: return all (will be disambiguated later)
            return candidates[0][0]

        # Try legacy
        legacy = legacy_name_idx.get(key, [])
        if legacy:
            if province_code:
                # Filter by province
                filtered = [c for c in legacy
                            if any(v['canonical'] == c and v['province_code'] == province_code
                                   for v in fst.ward_meta.values())]
                if filtered: return filtered[0]
            return legacy[0]

    return None

# ── Pass 1: addr: tag extraction ──────────────────────────────────────────────
class AddrHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        # street_slug → set of ward_canonicals
        self.mapping   = defaultdict(set)
        # street_slug → canonical street name (preserve original case/diacritics)
        self.street_canon = {}

        self.n_total   = 0
        self.n_matched = 0
        self.n_no_ward = 0
        self.n_unmatch = 0

    def _process(self, tags):
        street = tags.get('addr:street','').strip()
        if not street: return

        # Various ward tag keys used in Vietnam OSM
        ward_raw = (tags.get('addr:suburb','') or
                    tags.get('addr:ward','')   or
                    tags.get('addr:quarter','') or
                    tags.get('addr:hamlet','') or
                    tags.get('addr:subdistrict','')).strip()

        city_raw = (tags.get('addr:city','')   or
                    tags.get('addr:province','') or
                    tags.get('addr:state','')).strip()

        self.n_total += 1
        street_slug = slug(street)
        # Preserve canonical street name (Vietnamese with tones)
        if street_slug not in self.street_canon:
            self.street_canon[street_slug] = street

        if not ward_raw:
            self.n_no_ward += 1
            return

        # Resolve province first (helps disambiguate)
        prov_code = resolve_province(city_raw) if city_raw else None

        # Resolve ward
        ward_canon = resolve_ward(ward_raw, prov_code)
        if ward_canon:
            self.mapping[street_slug].add(ward_canon)
            self.n_matched += 1
        else:
            self.n_unmatch += 1

    def node(self, n): self._process(n.tags)
    def way(self,  w): self._process(w.tags)

t0 = time.time()
print("\nPass 1: addr: tag extraction (no spatial needed)...", flush=True)
h = AddrHandler()
h.apply_file(PBF, locations=False)
t1 = time.time()

print(f"  Time             : {t1-t0:.1f}s")
print(f"  Total addr nodes : {h.n_total:,}")
print(f"  Matched (ward)   : {h.n_matched:,}")
print(f"  No ward tag      : {h.n_no_ward:,}")
print(f"  Ward unmatched   : {h.n_unmatch:,}")
print(f"  Unique streets   : {len(h.mapping):,}")

# ── Coverage analysis ─────────────────────────────────────────────────────────
with open('/root/vn_address/streets_clean.json', encoding='utf-8') as f:
    clean_streets = json.load(f)   # slug → canonical

covered     = sum(1 for s in clean_streets if s in h.mapping)
not_covered = sum(1 for s in clean_streets if s not in h.mapping)

print(f"\nCoverage of clean street list:")
print(f"  Clean streets total    : {len(clean_streets):,}")
print(f"  Covered by addr: tags  : {covered:,}  ({covered/len(clean_streets)*100:.1f}%)")
print(f"  Not covered (need join): {not_covered:,}  ({not_covered/len(clean_streets)*100:.1f}%)")

# ── Build output: prefer addr: mapping, fill gaps from streets_clean ──────────
# For streets in clean_streets but NOT in addr: mapping:
# → Still include them with empty ward list (will need spatial join OR
#   model learns from street name pattern alone)
result = {}
for street_slug, street_canon in clean_streets.items():
    wards = list(h.mapping.get(street_slug, set()))
    # Use preserved street canonical if available (better Vietnamese)
    best_name = h.street_canon.get(street_slug, street_canon)
    result[street_slug] = {
        "canonical": best_name,
        "wards":     sorted(wards),
        "source":    "addr_tag" if wards else "osm_name_only"
    }

# Also add any streets found via addr: tags but NOT in clean list
# (may have been filtered out — add them back if they have ward coverage)
extra = 0
for street_slug, wards in h.mapping.items():
    if street_slug not in result and len(street_slug) > 3:
        best_name = h.street_canon.get(street_slug, street_slug)
        result[street_slug] = {
            "canonical": best_name,
            "wards":     sorted(wards),
            "source":    "addr_tag_extra"
        }
        extra += 1

print(f"\nFinal street_ward_map.json:")
print(f"  Total entries          : {len(result):,}")
print(f"  With ward mapping      : {sum(1 for v in result.values() if v['wards']):,}")
print(f"  Without ward mapping   : {sum(1 for v in result.values() if not v['wards']):,}")
print(f"  Extra from addr: only  : {extra:,}")

# Save
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)
print(f"\nSaved: {OUT}")
print(f"File size: {Path(OUT).stat().st_size/1024:.0f} KB")

# Stats
stats = {
    "total_streets": len(result),
    "with_ward_mapping": sum(1 for v in result.values() if v["wards"]),
    "without_ward_mapping": sum(1 for v in result.values() if not v["wards"]),
    "addr_tag_matched": h.n_matched,
    "addr_tag_total": h.n_total,
    "coverage_pct": round(covered/len(clean_streets)*100, 1),
}
with open(STAT, 'w') as f: json.dump(stats, f, indent=2)

# Sample output
print("\nSample with ward mapping:")
shown = 0
for slug_key, v in result.items():
    if v['wards'] and shown < 8:
        print(f"  {v['canonical']!r:35} → {v['wards'][:2]}")
        shown += 1

print("\nSample without ward mapping:")
shown = 0
for slug_key, v in result.items():
    if not v['wards'] and shown < 5:
        print(f"  {v['canonical']!r}")
        shown += 1
