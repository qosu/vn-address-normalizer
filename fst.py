import json, pickle
from pathlib import Path
from collections import defaultdict
from unidecode import unidecode
from vietnam_provinces import NESTED_DIVISIONS_JSON_PATH
from vietnam_provinces._province_conversion_2025 import OLD_TO_NEW as PROV_OLD_TO_NEW
from vietnam_provinces._ward_conversion_2025 import OLD_TO_NEW as WARD_OLD_TO_NEW

def slug(s): return unidecode(s).lower().strip()

class AddressFST:
    def __init__(self):
        self.valid_set = set()
        self.tree = {}
        self.ward_meta = {}
        self.prov_meta = {}
        self._sorted = []
        self.slug_to_wards = defaultdict(list)
        self.slug_to_provs = defaultdict(list)
        self.old_ward_to_new = defaultdict(list)
        self.old_prov_to_new = defaultdict(list)

    def build(self):
        with open(NESTED_DIVISIONS_JSON_PATH, encoding="utf-8") as f:
            provinces_raw = json.load(f)
        for p in provinces_raw:
            pc = p["code"]
            self.prov_meta[pc] = p
            self.tree[pc] = {}
            for sl in [slug(p["name"]), p["codename"].replace("_"," ")]:
                self.slug_to_provs[sl].append(pc)
            for w in p.get("wards", []):
                wc = w["code"]
                canon = f"{w['name']}, {p['name']}"
                self.tree[pc][wc] = canon
                self.valid_set.add(canon)
                self.ward_meta[wc] = {**w, "province_code": pc,
                                      "province_name": p["name"], "canonical": canon}
                for sl in [slug(w["name"]), w["short_codename"].replace("_"," ")]:
                    self.slug_to_wards[sl].append(wc)
        for old_wc, entry in WARD_OLD_TO_NEW.items():
            for nw in entry.new_wards:
                self.old_ward_to_new[old_wc].append(nw.code)
        for old_pc, entry in PROV_OLD_TO_NEW.items():
            for np in entry.new_provinces:
                self.old_prov_to_new[old_pc].append(np.code)
        self._sorted = sorted(self.valid_set)
        return self

    def accepts(self, s): return s in self.valid_set

    def legacy_ward(self, old_code):
        return [self.ward_meta[c]["canonical"]
                for c in self.old_ward_to_new.get(old_code, [])
                if c in self.ward_meta]

    def legacy_province(self, old_code):
        return [self.prov_meta[c]["name"]
                for c in self.old_prov_to_new.get(old_code, [])
                if c in self.prov_meta]

def load():
    cache = Path("/root/vn_address/fst_cache.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, 'valid_set') and len(obj.valid_set) > 3000:
            return obj
    fst = AddressFST().build()
    with open(cache, "wb") as f:
        pickle.dump(fst, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fst
