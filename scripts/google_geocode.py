"""
Google Geocoding API — reverse geocode 18,454 streets → ward
Rate: 50 req/s | Est: ~6 minutes | Free tier: 40K/month
"""
import json, time, re, sys, urllib.request, urllib.parse
from pathlib import Path
from unidecode import unidecode
from collections import defaultdict

sys.path.insert(0, '/root/vn_address')

API_KEY  = os.getenv('GOOGLE_GEOCODE_API_KEY', '')
MAP_FILE = '/root/vn_address/street_ward_map.json'
OSM_FILE = '/root/vietnam-latest.osm.pbf'
PROG_FILE= '/root/vn_address/google_progress.json'

def slug(s): return unidecode(s).lower().strip()

ROAD_PAT = re.compile(
    r'^(quoc lo|tinh lo|duong tinh|duong huyen|cao toc|'
    r'quoc|tinh|huyen|xa lo|tl |ql |dt |dh )', re.I)

# Load map
with open(MAP_FILE, encoding='utf-8') as f:
    street_map = json.load(f)

# Load FST for ward resolution
from fst import load as fst_load
fst = fst_load()

ward_idx = defaultdict(list)
for wc, meta in fst.ward_meta.items():
    for key in [slug(meta['name']),
                re.sub(r'^(xa|phuong|thi tran|dac khu)\s+','',slug(meta['name'])).strip()]:
        ward_idx[key].append(meta['canonical'])

prov_idx = {}
for pc, meta in fst.prov_meta.items():
    prov_idx[slug(meta['name'])] = pc
    prov_idx[meta.get('codename','').replace('_',' ')] = pc

# Load GPS coords from OSM
print("Loading GPS coords...", flush=True)
import osmium

class GpsHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.coords = {}
    def way(self, w):
        name = w.tags.get('name','') or w.tags.get('name:vi','')
        if not name: return
        key = slug(name)
        if key in self.coords: return
        try:
            nodes = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if nodes:
                self.coords[key] = nodes[len(nodes)//2]
        except: pass

gps = GpsHandler()
gps.apply_file(OSM_FILE, locations=True, idx='flex_mem')
print(f"GPS loaded: {len(gps.coords):,}", flush=True)

# Load progress
progress = {}
if Path(PROG_FILE).exists():
    with open(PROG_FILE) as f: progress = json.load(f)
print(f"Already done: {len(progress):,}", flush=True)

# Targets: streets without ward, not roads, have GPS
targets = {
    k: v for k, v in street_map.items()
    if not v['wards']
    and not ROAD_PAT.match(slug(v['canonical']))
    and k in gps.coords
    and k not in progress
}
print(f"Targets: {len(targets):,}", flush=True)

def google_reverse(lat, lon):
    url = (f"https://maps.googleapis.com/maps/api/geocode/json"
           f"?latlng={lat},{lon}&language=vi&result_type=sublocality"
           f"&key={API_KEY}")
    req = urllib.request.Request(url, headers={"User-Agent":"VNAddress/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def extract_ward(result):
    if result.get('status') != 'OK': return None
    for res in result.get('results', []):
        for comp in res.get('address_components', []):
            if 'sublocality' in comp.get('types', []):
                name = comp.get('long_name', '')
                key  = slug(name)
                name_only = re.sub(r'^(phuong|xa|thi tran|p\.|x\.)\s*','',key).strip()
                for k in [key, name_only]:
                    if ward_idx.get(k): return ward_idx[k][0]
    return None

# Run
done = failed = skipped = 0
BATCH = 50   # 50 req/s max
t_batch = time.time()

items = list(targets.items())
for i, (slug_key, data) in enumerate(items):
    lon, lat = gps.coords[slug_key]
    try:
        result = google_reverse(lat, lon)
        ward   = extract_ward(result)
        progress[slug_key] = ward or ""
        if ward:
            street_map[slug_key]['wards'].append(ward)
            street_map[slug_key]['source'] = 'google'
            done += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1
        progress[slug_key] = ""

    # Rate limiting: 50 req/s
    if (i+1) % BATCH == 0:
        elapsed = time.time() - t_batch
        if elapsed < 1.0: time.sleep(1.0 - elapsed)
        t_batch = time.time()

    if (i+1) % 500 == 0:
        with open(PROG_FILE,'w') as f: json.dump(progress, f)
        with open(MAP_FILE,'w',encoding='utf-8') as f:
            json.dump(street_map, f, ensure_ascii=False)
        pct = (i+1)/len(items)*100
        print(f"[{pct:.1f}%] {i+1}/{len(items)} | matched={done} failed={failed}", flush=True)

# Final save
with open(PROG_FILE,'w') as f: json.dump(progress, f)
with open(MAP_FILE,'w',encoding='utf-8') as f:
    json.dump(street_map, f, ensure_ascii=False)

with_ward = sum(1 for v in street_map.values() if v['wards'])
print(f"\nDone: matched={done} failed={failed}")
print(f"Coverage: {with_ward:,}/{len(street_map):,} = {with_ward/len(street_map)*100:.1f}%")
