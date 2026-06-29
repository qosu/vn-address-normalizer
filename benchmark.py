import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalizer import normalize, get_indexes
get_indexes()

TESTS = [
    ("Phường Ba Đình, Thành phố Hà Nội",          "Phường Ba Đình, Thành phố Hà Nội",          1,"exact"),
    ("Phường Phú Mỹ, Thành phố Hồ Chí Minh",      "Phường Phú Mỹ, Thành phố Hồ Chí Minh",      1,"exact"),
    ("Phuong Ba Dinh, Thanh pho Ha Noi",           "Phường Ba Đình, Thành phố Hà Nội",          2,"no_tones"),
    ("Xa Cu Chi, TP Ho Chi Minh",                  "Xã Củ Chi, Thành phố Hồ Chí Minh",          2,"abbrev"),
    ("P Phu My, TP.HCM",                           "Phường Phú Mỹ, Thành phố Hồ Chí Minh",      2,"abbrev"),
    ("Phường Tân Định, Quận 1, TP.HCM",            "Phường Tân Định, Thành phố Hồ Chí Minh",    3,"old_dist"),
    ("Xã Củ Chi, Huyện Củ Chi, TP Hồ Chí Minh",   "Xã Củ Chi, Thành phố Hồ Chí Minh",          3,"old_dist"),
    ("P. Bến Nghé, Q.1, TP HCM",                  "Phường Sài Gòn, Thành phố Hồ Chí Minh",     3,"legacy_ward"),
    ("Xã Thượng Lâm, Tỉnh Hà Giang",              "Xã Thượng Lâm, Tỉnh Tuyên Quang",           4,"old_prov"),
    ("Phường Lào Cai, Tỉnh Yên Bái",              "Phường Lào Cai, Tỉnh Lào Cai",              4,"old_prov"),
    ("p tan dinh q1 tphcm",                        "Phường Tân Định, Thành phố Hồ Chí Minh",    5,"no_comma"),
    ("XA CU CHI TP HO CHI MINH",                   "Xã Củ Chi, Thành phố Hồ Chí Minh",          5,"allcaps"),
    ("phuong-ba-dinh, ha-noi",                     "Phường Ba Đình, Thành phố Hà Nội",          5,"hyphen"),
    ("Phường 14, Quận 5, TP.HCM",                 None,                                         6,"invalid"),
    ("Xã ABC, Tỉnh Hà Giang",                     None,                                         6,"nonexist"),
]

tier_res = {}
print(f"\n{'INPUT':<46} {'STAGE':<18} {'CONF':>6}  RESULT")
print("-"*100)
for inp, expected, tier, desc in TESTS:
    r = normalize(inp)
    if expected is None:
        ok = not r.valid
        res = f"{'✓ REJECTED' if ok else '✗ HALLUCINATED: '+r.canonical}"
    else:
        ok = r.canonical == expected
        res = f"{'✓' if ok else '✗ GOT: '+r.canonical}"
    tier_key = f"T{tier}"
    d = tier_res.setdefault(tier_key, {"ok":0,"total":0})
    d["total"] += 1
    if ok: d["ok"] += 1
    print(f"{inp[:45]:<46} {r.stage:<18} {r.confidence:>6.3f}  {res}")

tdesc = {"T1":"Clean new format","T2":"No tones/abbrev","T3":"Old district format",
         "T4":"Old province name","T5":"Heavy noise","T6":"Adversarial"}
print(f"\n{'TIER':<5} {'DESCRIPTION':<28} SCORE")
tok = tn = 0
for t,d in sorted(tier_res.items()):
    tok+=d["ok"]; tn+=d["total"]
    print(f"  {t}  {tdesc.get(t,''):<28} {d['ok']}/{d['total']}  {d['ok']/d['total']*100:.0f}%")
print(f"\n  OVERALL {tok}/{tn}  {tok/tn*100:.0f}%")
