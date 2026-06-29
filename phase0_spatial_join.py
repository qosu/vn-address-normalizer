"""
Phase 0b: Spatial join — street midpoints × ward boundaries
Fills the 91.1% gap not covered by addr: tags.

Algorithm:
  Pass 1: Extract admin_level=8 ward boundaries → shapely polygons
  Pass 2: Extract street ways + coordinates → midpoints
  Pass 3: STRtree spatial join → street midpoint in which ward polygon
  Pass 4: Merge with existing addr: mapping
"""
import osmium, osmium.geom, json, sys, re, time
from collections import defaultdict
from pathlib import Path
from unidecode import unidecode

try:
    from shapely import wkb as shapely_wkb
    from shapely.geometry import Point, MultiPolygon, Polygon
    from shapely.strtree import STRtree
    SHAPELY = True
except ImportError:
    print("shapely not installed — run: pip install shapely --break-system-packages")
    sys.exit(1)

PBF      = '/root/vietnam-latest.osm.pbf'
MAP_IN   = '/root/vn_address/street_ward_map.json'   # from Phase 0a
MAP_OUT  = '/root/vn_address/street_ward_map.json'   # overwrite with merged
LOC_IDX  = '/tmp/vn_osm_nodes'                       # disk-based node index
sys.path.insert(0, '/root/vn_address')

def slug(s): return unidecode(s).lower().strip()

# ── Load VBHC + build ward name matchers ─────────────────────────────────────
print("Loading VBHC...", flush=True)
from fst import load as fst_load
fst = fst_load()

# ward_slug → list of (canonical, prov_code)
ward_idx = defaultdict(list)
for wc, m in fst.ward_meta.items():
    for key in [slug(m['name']),
                re.sub(r'^(xa|phuong|thi tran|dac khu)\s+','',slug(m['name'])).strip()]:
        entry = (m['canonical'], m['province_code'])
        if entry not in ward_idx[key]:
            ward_idx[key].append(entry)

# province centroid → code (for province-of-boundary lookup)
# We'll determine this from the ward canonical string
def canonical_from_name(osm_name, prov_hint=None):
    """OSM boundary name → VBHC canonical ward string"""
    ws = slug(osm_name)
    name_only = re.sub(r'^(xa|phuong|thi tran|p\.|x\.)\s*','',ws).strip()
    for key in ([ws, name_only] if name_only != ws else [ws]):
        candidates = ward_idx.get(key, [])
        if not candidates: continue
        if prov_hint:
            filtered = [c for c,pc in candidates if pc == prov_hint]
            if filtered: return filtered[0]
        # Return most likely (highest logistics prior province)
        from normalizer import PROVINCE_PRIOR
        best = max(candidates, key=lambda x: PROVINCE_PRIOR.get(x[1], 0.008))
        return best[0]
    # Try legacy
    try:
        from vietnam_provinces.legacy import Ward as LW, WardCode as LWC
        from vietnam_provinces._ward_conversion_2025 import OLD_TO_NEW
        for wc in LWC:
            if name_only in slug(str(LW.from_code(wc))):
                entry = OLD_TO_NEW.get(wc.value)
                if entry:
                    for nw in entry.new_wards:
                        if nw.code in fst.ward_meta:
                            canon = fst.ward_meta[nw.code]['canonical']
                            if not prov_hint or fst.ward_meta[nw.code]['province_code'] == prov_hint:
                                return canon
    except Exception:
        pass
    return None

# ── Pass 1: Extract ward boundaries ──────────────────────────────────────────
wkbfab = osmium.geom.WKBFactory()

class WardBoundaryHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.wards = []       # list of (canonical, shapely_geom)
        self.n_ok  = 0
        self.n_err = 0

    def area(self, a):
        level = a.tags.get('admin_level','')
        if level != '8': return
        name = a.tags.get('name','').strip()
        if not name: return
        try:
            wkb_data = wkbfab.create_multipolygon(a)
            geom     = shapely_wkb.loads(wkb_data, hex=False)
            canon    = canonical_from_name(name)
            if canon and not geom.is_empty:
                self.wards.append((canon, geom))
                self.n_ok += 1
        except Exception:
            self.n_err += 1

t0 = time.time()
print(f"\nPass 1: Extracting ward boundaries (admin_level=8)...", flush=True)
wb = WardBoundaryHandler()
wb.apply_file(PBF, locations=True, idx='flex_mem')
t1 = time.time()

print(f"  Ward polygons extracted: {wb.n_ok:,}")
print(f"  Extraction errors      : {wb.n_err:,}")
print(f"  Time: {t1-t0:.1f}s", flush=True)

if wb.n_ok == 0:
    print("ERROR: No ward boundaries extracted. Check OSM data / admin_level tags.")
    sys.exit(1)

# ── Build spatial index ───────────────────────────────────────────────────────
print("\nBuilding STRtree spatial index...", flush=True)
ward_geoms  = [g for _, g in wb.wards]
ward_canons = [c for c, _ in wb.wards]
tree = STRtree(ward_geoms)
print(f"  Index built: {len(ward_geoms)} polygons", flush=True)

# ── Pass 2: Extract street ways + coordinates ─────────────────────────────────
KEEP_HW = {
    'motorway','trunk','primary','secondary','tertiary',
    'unclassified','residential','living_street','service',
    'motorway_link','trunk_link','primary_link','secondary_link','tertiary_link',
    'road','pedestrian','busway',
}

class StreetGeoHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.streets = {}   # slug → (canonical_name, midpoint_lon, midpoint_lat)
        self.n_ok = self.n_err = 0

    def way(self, w):
        hw   = w.tags.get('highway','')
        name = w.tags.get('name','') or w.tags.get('name:vi','')
        if not name or hw not in KEEP_HW: return
        name = name.strip()
        key  = slug(name)
        if key in self.streets: return   # already have it
        try:
            nodes = [(n.lon, n.lat) for n in w.nodes
                     if n.location.valid()]
            if len(nodes) < 2: return
            mid = nodes[len(nodes)//2]
            self.streets[key] = (name, mid[0], mid[1])
            self.n_ok += 1
        except Exception:
            self.n_err += 1

print("\nPass 2: Extracting street geometries...", flush=True)
sg = StreetGeoHandler()
# Reuse the same location index
sg.apply_file(PBF, locations=True, idx='flex_mem')
t2 = time.time()
print(f"  Streets with coords: {sg.n_ok:,}")
print(f"  Errors             : {sg.n_err:,}")
print(f"  Time: {t2-t1:.1f}s", flush=True)

# ── Pass 3: Spatial join ──────────────────────────────────────────────────────
print("\nPass 3: Spatial join...", flush=True)
spatial_mapping = defaultdict(set)   # street_slug → set of ward_canonicals
n_matched = n_unmatched = 0

for i, (street_slug, (name, lon, lat)) in enumerate(sg.streets.items()):
    if i % 5000 == 0:
        print(f"  {i:,}/{len(sg.streets):,}", flush=True)
    pt = Point(lon, lat)
    # Query R-tree (returns indices of bboxes that contain pt)
    candidates = list(tree.query(pt))
    for idx in candidates:
        if ward_geoms[idx].contains(pt):
            spatial_mapping[street_slug].add(ward_canons[idx])
            n_matched += 1
            break
    else:
        n_unmatched += 1

t3 = time.time()
print(f"  Matched   : {n_matched:,}")
print(f"  Unmatched : {n_unmatched:,}")
print(f"  Time: {t3-t2:.1f}s", flush=True)

# ── Pass 4: Merge with addr: results ─────────────────────────────────────────
print("\nPass 4: Merging with addr: mapping...", flush=True)
with open(MAP_IN, encoding='utf-8') as f:
    existing = json.load(f)

merged  = 0
new_add = 0
for slug_key, data in existing.items():
    spatial_wards = list(spatial_mapping.get(slug_key, set()))
    combined = list(dict.fromkeys(data['wards'] + spatial_wards))
    if spatial_wards and not data['wards']:
        merged += 1
    if combined != data['wards']:
        data['wards']  = combined
        if spatial_wards: data['source'] = 'spatial_join'
    existing[slug_key] = data

# Add streets found in spatial join but not in existing map
for slug_key, ward_set in spatial_mapping.items():
    if slug_key not in existing and ward_set:
        name = sg.streets.get(slug_key, (slug_key,))[0]
        existing[slug_key] = {
            "canonical": name,
            "wards":     sorted(ward_set),
            "source":    "spatial_only"
        }
        new_add += 1

print(f"  Filled gaps (addr→spatial): {merged:,}")
print(f"  New entries from spatial   : {new_add:,}")

# Save
with open(MAP_OUT, 'w', encoding='utf-8') as f:
    json.dump(existing, f, ensure_ascii=False)

# Final stats
total      = len(existing)
with_wards = sum(1 for v in existing.values() if v['wards'])
coverage   = with_wards / total * 100

print(f"\n=== FINAL RESULTS ===")
print(f"Total streets     : {total:,}")
print(f"With ward mapping : {with_wards:,}  ({coverage:.1f}%)")
print(f"Without mapping   : {total - with_wards:,}  ({100-coverage:.1f}%)")
print(f"Saved: {MAP_OUT}")
print(f"Total time: {time.time()-t0:.1f}s")
