"""
Batch reverse geocoding: 18,454 streets → ward via Nominatim
Rate: 1 req/s | Est: ~5.5h | Run overnight
"""
import json, time, re, sys, urllib.request, urllib.parse
from pathlib import Path
from unidecode import unidecode

sys.path.insert(0, '/root/vn_address')

MAP_FILE  = '/root/vn_address/street_ward_map.json'
OSM_FILE  = '/root/vietnam-latest.osm.pbf'
SAVE_FILE = '/root/vn_address/street_ward_map.json'
PROG_FILE = '/root/vn_address/nominatim_progress.json'

def slug(s): return unidecode(s).lower().strip()

ROAD_PATTERNS = re.compile(
    r'^(quoc lo|tinh lo|duong tinh|duong huyen|duong xa|'
    r'cao toc|quoc|tinh|huyen|xa lo|tl |ql |dt |dh )', re.I)

# Load existing map
with open(MAP_FILE, encoding='utf-8') as f:
    street_map = json.load(f)

# Load FST for ward resolution
from fst import load as fst_load
fst = fst_load()

# Build ward name index (slug → canonical)
from collections import defaultdict
ward_idx = defaultdict(list)
for wc, meta in fst.ward_meta.items():
    ward_idx[slug(meta['name'])].append(meta['canonical'])
    # also legacy
    for new_canon in fst.legacy_ward(wc):
        ward_idx[slug(meta['name'])].append(new_canon)

# Load OSM street GPS coords
print("Loading GPS coords from OSM...", flush=True)
import osmium
class GpsHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.coords = {}  # slug → (lon, lat)
    def way(self, w):
        name = w.tags.get('name','') or w.tags.get('name:vi','')
        if not name: return
        key = slug(name)
        if key in self.coords: return
        try:
            nodes = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if nodes:
                mid = nodes[len(nodes)//2]
                self.coords[key] = mid
        except: pass

gps = GpsHandler()
gps.apply_file(OSM_FILE, locations=True, idx='flex_mem')
print(f"GPS coords loaded: {len(gps.coords):,}", flush=True)

# Load progress
progress = {}
if Path(PROG_FILE).exists():
    with open(PROG_FILE) as f:
        progress = json.load(f)
print(f"Already done: {len(progress):,}", flush=True)

# Target: streets without wards that are real streets
targets = {
    k: v for k, v in street_map.items()
    if not v['wards']
    and not ROAD_PATTERNS.match(slug(v['canonical']))
    and slug(v['canonical']) in gps.coords
    and k not in progress
}
print(f"Targets to geocode: {len(targets):,}", flush=True)

def nominatim_reverse(lon, lat):
    url = (f"https://nominatim.openstreetmap.org/reverse"
           f"?lat={lat}&lon={lon}&format=json&zoom=16"
           f"&accept-language=vi")
    req = urllib.request.Request(url, headers={
        "User-Agent": "VNAddressNormalizer/1.0 research@example.com"
    })
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def extract_ward(result):
    """Extract ward name from Nominatim reverse result."""
    addr = result.get('address', {})
    # Try fields in priority order
    for field in ['suburb', 'quarter', 'neighbourhood',
                  'village', 'hamlet', 'town', 'city_district']:
        val = addr.get(field, '')
        if val:
            # Try to match to VBHC ward
            key = slug(val)
            if key in ward_idx:
                return ward_idx[key][0]
            # Try legacy
            legacy = fst.fuzzy(val, top_k=1)
            if legacy:
                return legacy[0][1]
    return None

# Run batch
done = skipped = failed = 0
save_every = 100

for i, (slug_key, data) in enumerate(targets.items()):
    canon = data['canonical']
    coords = gps.coords.get(slug_key)
    if not coords:
        skipped += 1
        continue

    lon, lat = coords
    try:
        result = nominatim_reverse(lon, lat)
        ward   = extract_ward(result)
        progress[slug_key] = ward or ""
        if ward:
            street_map[slug_key]['wards'].append(ward)
            street_map[slug_key]['source'] = 'nominatim'
            done += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1
        progress[slug_key] = ""

    if (i+1) % 100 == 0:
        # Save progress
        with open(PROG_FILE, 'w') as f: json.dump(progress, f)
        with open(SAVE_FILE, 'w', encoding='utf-8') as f:
            json.dump(street_map, f, ensure_ascii=False)
        pct = (i+1)/len(targets)*100
        print(f"  [{pct:.1f}%] {i+1}/{len(targets)} | "
              f"matched={done} skipped={skipped} failed={failed}", flush=True)

    time.sleep(1.0)  # Nominatim rate limit: 1 req/s

# Final save
with open(PROG_FILE, 'w') as f: json.dump(progress, f)
with open(SAVE_FILE, 'w', encoding='utf-8') as f:
    json.dump(street_map, f, ensure_ascii=False)

with_ward = sum(1 for v in street_map.values() if v['wards'])
print(f"\nDone: {done} matched, {skipped} skipped, {failed} failed")
print(f"Coverage: {with_ward:,}/{len(street_map):,} = {with_ward/len(street_map)*100:.1f}%")
