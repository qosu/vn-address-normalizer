"""
Phase 1: Generate training data from street_ward_map.json
Target: ~3M pairs covering all noise patterns logistics addresses encounter
"""
import json, re, random, time, sys
from pathlib import Path
from unidecode import unidecode

sys.path.insert(0, '/root/vn_address')
random.seed(42)

def slug(s): return unidecode(s).lower().strip()

MAP  = '/root/vn_address/street_ward_map.json'
OUT  = '/root/vn_address/train.jsonl'
VAL  = '/root/vn_address/val.jsonl'

with open(MAP, encoding='utf-8') as f:
    street_map = json.load(f)

from fst import load as fst_load
fst = fst_load()

# ── Corruption functions ──────────────────────────────────────────────────────

def drop_all_tones(s):
    return unidecode(s)

def drop_some_tones(s, p=0.5):
    return ' '.join(unidecode(w) if random.random()<p else w for w in s.split())

def abbrev_street_type(s):
    m = {'Đường':'D.','Phố':'P.','Kiệt':'Kt.','Ngõ':'Ng.','Hẻm':'H.'}
    for full, short in m.items():
        if s.startswith(full) and random.random() < 0.7:
            return s.replace(full, random.choice([short, full.lower(), slug(full)]), 1)
    return s

def abbrev_ward(s):
    m = {'Phường':'P.','Xã':'X.','Thị trấn':'TT.'}
    r = s
    for full, short in m.items():
        if full in r and random.random() < 0.7:
            r = r.replace(full, random.choice([short, full.lower(), '']), 1)
    return r.strip()

def abbrev_province(s):
    m = {'Thành phố':'TP.','Tỉnh':''}
    r = s
    for full, short in m.items():
        if full in r and random.random() < 0.6:
            r = r.replace(full, random.choice([short, 'tp', full.lower()]), 1)
    return r.strip()

def random_case(s):
    f = random.randint(0, 3)
    if f == 0: return s.upper()
    if f == 1: return s.lower()
    if f == 2: return s.title()
    return ' '.join(w.upper() if random.random()<0.4 else w.lower() for w in s.split())

def add_noise_punct(s):
    ops = [
        lambda t: t.replace(', ', ','),
        lambda t: t.replace(', ', ' '),
        lambda t: re.sub(r'(?<=\w)-(?=\w)', ' ', t),
        lambda t: t.replace(' ', '-', 1),
    ]
    return random.choice(ops)(s)

def inject_old_district(s):
    """Simulate pre-2025 address with district injected between ward and province"""
    parts = [p.strip() for p in s.split(',')]
    if len(parts) >= 2 and random.random() < 0.3:
        fakes = ['Quận 1','Q.1','Q.3','Quận 5','Huyện X','H. Bình Chánh',
                 'Quận Bình Thạnh','Huyện Hóc Môn','Q.Tân Bình']
        parts.insert(-1, random.choice(fakes))
    return ', '.join(parts)

def add_house_number(s):
    """Add realistic house number prefix (model should ignore it)"""
    if random.random() < 0.4:
        num = random.choice([
            f"{random.randint(1,300)}",
            f"{random.randint(1,50)}/{random.randint(1,20)}",
            f"{random.randint(1,300)}B",
        ])
        return f"{num} {s}"
    return s

def reorder_components(s):
    """Swap street/ward/province order"""
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) >= 2 and random.random() < 0.3:
        random.shuffle(parts)
    return ', '.join(parts)

def ocr_errors(s):
    m = {'l':'1','1':'l','O':'0','0':'O','I':'l','ơ':'o','ư':'u'}
    out = list(s)
    for i,c in enumerate(out):
        if c in m and random.random() < 0.06:
            out[i] = m[c]
    return ''.join(out)

CORRUPTIONS = [
    (drop_all_tones,      0.30),
    (drop_some_tones,     0.25),
    (abbrev_street_type,  0.55),
    (abbrev_ward,         0.55),
    (abbrev_province,     0.45),
    (random_case,         0.40),
    (add_noise_punct,     0.20),
    (inject_old_district, 0.25),
    (add_house_number,    0.40),
    (reorder_components,  0.20),
    (ocr_errors,          0.10),
]

def corrupt(full_address, n=None):
    if n is None: n = random.randint(2, 5)
    ops = list(CORRUPTIONS); random.shuffle(ops)
    out = full_address; applied = 0
    for fn, prob in ops:
        if applied >= n: break
        if random.random() < prob:
            out = fn(out); applied += 1
    return out.strip()

# ── Generate pairs ────────────────────────────────────────────────────────────
t0 = time.time()
train_pairs = []
val_pairs   = []
VARIANTS_PER = 60   # per (street, ward) canonical pair
VAL_RATIO    = 0.05

n_full = n_ward_only = 0

print(f"Generating training pairs...", flush=True)

# 1. Full address pairs (street + ward + province)
for slug_key, data in street_map.items():
    if not data['wards']: continue
    street_canon = data['canonical']
    for ward_canon in data['wards']:
        full_canon = f"{street_canon}, {ward_canon}"
        # Generate VARIANTS_PER noisy versions
        for _ in range(VARIANTS_PER):
            noisy = corrupt(full_canon)
            pair = {
                "input":          noisy,
                "street":         street_canon,
                "ward_canonical": ward_canon,
                "full_canonical": full_canon,
            }
            if random.random() < VAL_RATIO:
                val_pairs.append(pair)
            else:
                train_pairs.append(pair)
        n_full += 1

print(f"  Full address pairs: {n_full:,}", flush=True)

# 2. Ward-only pairs (admin unit normalization — already works well but reinforce)
for wc, meta in fst.ward_meta.items():
    ward_canon = meta['canonical']
    for _ in range(30):  # fewer variants since we already have good coverage
        noisy = corrupt(ward_canon)
        pair = {
            "input":          noisy,
            "street":         "",
            "ward_canonical": ward_canon,
            "full_canonical": ward_canon,
        }
        if random.random() < VAL_RATIO:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)
    n_ward_only += 1

print(f"  Ward-only pairs   : {n_ward_only:,}", flush=True)

# Shuffle
random.shuffle(train_pairs)
random.shuffle(val_pairs)

# Write
with open(OUT, 'w', encoding='utf-8') as f:
    for p in train_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

with open(VAL, 'w', encoding='utf-8') as f:
    for p in val_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

elapsed = time.time() - t0
print(f"\n=== DONE ===")
print(f"Train pairs : {len(train_pairs):,}")
print(f"Val pairs   : {len(val_pairs):,}")
print(f"Total       : {len(train_pairs)+len(val_pairs):,}")
print(f"Train file  : {Path(OUT).stat().st_size/1024/1024:.0f} MB")
print(f"Val file    : {Path(VAL).stat().st_size/1024/1024:.0f} MB")
print(f"Time        : {elapsed:.1f}s")

# Sample
print("\nSample pairs:")
for p in random.sample(train_pairs, 5):
    print(f"  INPUT: {p['input']!r}")
    print(f"  FULL:  {p['full_canonical']!r}")
    print()
