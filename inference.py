"""
VN Address Normalizer — Standalone Inference
============================================
No FST, no vietnam-provinces. Runs standalone on any machine with:
    pip install -r requirements.txt

Usage (CLI):
    python inference.py "p tan dinh q1 tphcm"

Usage (import):
    from inference import normalize
    result = normalize("p tan dinh q1 tphcm")
    print(result["canonical"])
"""

import json, re, time, sys
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from unidecode import unidecode

MODEL_DIR = Path(__file__).resolve().parent / "model_v3_final"

def slug(s: str) -> str:
    return unidecode(s).lower().strip()

# ── Load artifacts ────────────────────────────────────────────────────────────
cfg       = json.load(open(MODEL_DIR / "config.json"))
src_vocab = json.load(open(MODEL_DIR / "src_vocab.json", encoding="utf-8"))
tgt_vocab = json.load(open(MODEL_DIR / "tgt_vocab.json", encoding="utf-8"))
clean      = json.load(open(MODEL_DIR / "clean_canonicals.json", encoding="utf-8"))
legacy_idx = json.load(open(MODEL_DIR / "legacy_ward_idx.json", encoding="utf-8"))

src_ch2id = {c: i for i, c in enumerate(src_vocab)}
tgt_ch2id = {c: i for i, c in enumerate(tgt_vocab)}
SRC_PAD, SRC_UNK, SRC_BOS, SRC_EOS = 0, 1, 2, 3
TGT_PAD, TGT_UNK, TGT_BOS, TGT_EOS = 0, 1, 2, 3

print(f"Canonicals: {len(clean):,}", flush=True)

# ── Build indexes from clean_canonicals.json (no FST) ─────────────────────────
prov_to_c = defaultdict(list)   # province_name → [canonical, ...]
pw_to_c   = defaultdict(list)   # (prov, ward_slug) → [canonical, ...]
ward_idx  = defaultdict(list)   # ward_slug → [canonical, ...]
ps        = {}                  # province_slug → canonical_province_name

for _c in clean:
    _parts = [p.strip() for p in _c.split(",")]
    if len(_parts) < 2:
        continue
    _prov      = _parts[-1]
    _ward_part = _parts[-2]
    _ps        = slug(_prov)

    ps[_ps] = _prov
    _stripped = re.sub(r"^(tinh|thanh pho|tp\.?)\s*", "", _ps).strip()
    if _stripped != _ps:
        ps[_stripped] = _prov

    prov_to_c[_prov].append(_c)

    for _ws in [slug(_ward_part),
                re.sub(r"^(phuong|xa|thi tran|dac khu)\s+", "", slug(_ward_part)).strip()]:
        pw_to_c[(_prov, _ws)].append(_c)
        ward_idx[_ws].append(_c)

# ── Province aliases (historical / colloquial names) ──────────────────────────
_OLD = {
    "hcm": "ho chi minh",       "tphcm": "ho chi minh",
    "saigon": "ho chi minh",    "sai gon": "ho chi minh",
    "hanoi": "ha noi",
    "ha giang": "tuyen quang",  "yen bai": "lao cai",
    "bac kan": "thai nguyen",   "vinh phuc": "phu tho",
    "hoa binh": "phu tho",      "bac giang": "bac ninh",
    "thai binh": "hung yen",    "hai duong": "hai phong",
    "ha nam": "ninh binh",      "nam dinh": "ninh binh",
    "quang binh": "quang tri",  "quang nam": "da nang",
    "kon tum": "quang ngai",    "binh dinh": "gia lai",
    "phu yen": "dak lak",       "ninh thuan": "khanh hoa",
    "dak nong": "dak lak",      "binh phuoc": "dong nai",
    "binh duong": "ho chi minh","ba ria vung tau": "ho chi minh",
    "long an": "tay ninh",      "tien giang": "tay ninh",
    "ben tre": "vinh long",     "tra vinh": "vinh long",
    "dong thap": "an giang",    "kien giang": "an giang",
    "hau giang": "can tho",     "soc trang": "ca mau",
    "bac lieu": "ca mau",       "thua thien hue": "hue",
    "tt hue": "hue",            "brvt": "ho chi minh",
    "vung tau": "ho chi minh",
}


def _resolve_prov(ts: str):
    ts2 = re.sub(r"^(tinh|tp\.?\s*|thanh pho)\s+", "", ts).strip()
    ts3 = re.sub(r"[.\s]", "", ts)
    for key in [ts, ts2, ts3]:
        if key in ps:
            return ps[key]
        alias = _OLD.get(key)
        if alias:
            for k, v in ps.items():
                if alias in k:
                    return v
    for k, v in ps.items():
        if ts2 and len(ts2) > 2 and (ts2 in k or k in ts2):
            return v
    return None


# ── Address component parser (inlined — no normalizer.py dependency) ──────────
# _WARD_PFX / _PROV_PFX operate on raw Vietnamese text (comma-split)
_WARD_PFX = re.compile(
    r"^(phường|phuong|ph\.|p\.|x\xe3|xa|x\."
    r"|đặc\s*khu|dk\.?)\s*", re.I)
_PROV_PFX = re.compile(
    r"^(tỉnh|tinh|th\xe0nh\s*phố|thanh\s*pho|tp\.?|t\.p\.?)\s*", re.I)
_DIST_PFX = re.compile(
    r"^(quận|quan|q\.?|huyện|huyen|h\.?|tx\.?)\s*", re.I)
_NUM_STR  = re.compile(r"^(\d+[a-z]?(?:/\d+[a-z]?)*)[\s,]+(.+)", re.I)

# _NC_* operate on slug text (unidecode+lower — no diacritics)
_NC_PROV = re.compile(
    r"\b(tphcm|hcm|hanoi|saigon|sai gon"
    r"|ho chi minh|hai phong|da nang|can tho|hue"
    r"|tp\s+[\w\s]{1,20}|tinh\s+[\w\s]{1,20})\b", re.I)
_NC_DIST = re.compile(r"\b(q\.?\s*\d+|quan\s*\d+|h\.\s*\w+|huyen\s+\w+)\b", re.I)
_NC_WARD = re.compile(r"^(phuong|xa|tt|p\.\s*|x\.\s*)([\w][\w\s]*)", re.I)


def _extract(raw: str) -> dict:
    """Parse comma-separated address into components."""
    parts = [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
    r = {"ward": None, "province": None, "district_hint": None}
    if parts:
        m = _NUM_STR.match(parts[0])
        if m:
            parts = [m.group(2)] + parts[1:]
    for part in parts:
        if   _PROV_PFX.match(part): r["province"]      = _PROV_PFX.sub("", part).strip()
        elif _DIST_PFX.match(part): r["district_hint"] = part
        elif _WARD_PFX.match(part): r["ward"]          = _WARD_PFX.sub("", part).strip()
        elif not r["ward"]:         r["ward"]           = part
    if not r["province"] and len(parts) >= 2:
        r["province"] = parts[-1]
    return r


def _parse_no_comma(raw: str) -> dict:
    """Parse space-only address on slug text."""
    r = {"ward": None, "province": None, "district_hint": None}
    text = slug(raw)
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
    r["ward"] = m.group(2).strip() if m else text
    return r


def detect_prov(raw: str):
    comps = _extract(raw) if "," in raw else _parse_no_comma(raw)
    for field in ["province", "district_hint"]:
        v = comps.get(field)
        if v:
            r = _resolve_prov(slug(v))
            if r:
                return r
    return _resolve_prov(slug(raw))


# ── Ward hint extractor ───────────────────────────────────────────────────────
_WS  = re.compile(r"\b(?:phuong|p\.|p\s|xa|x\.)\s*([a-z0-9][a-z0-9\s]{1,40})", re.I)
_NUM = re.compile(r"^\d{1,3}$")


def detect_ward(raw: str, prov: str):
    m = _WS.search(slug(raw))
    if not m:
        return None, None
    words = m.group(1).strip().split()
    for n in range(min(4, len(words)), 0, -1):
        cand = " ".join(words[:n])
        lead = cand.split()[0] if cand.split() else cand
        if _NUM.match(lead):
            return None, "numbered"
        for ws in [cand,
                   re.sub(r"^(phuong|xa|thi tran)\s+", "", cand).strip()]:
            if prov:
                canons = pw_to_c.get((prov, ws), [])
                if canons:
                    return ws, canons
            rb = ward_idx.get(ws, []) + legacy_idx.get(ws, [])
            if rb:
                pf = [c for c in rb if prov and prov in c] if prov else rb
                if pf:
                    return ws, pf
    return None, None


# ── Trie ──────────────────────────────────────────────────────────────────────
class TrieNode:
    __slots__ = ("children", "is_terminal")
    def __init__(self):
        self.children = {}
        self.is_terminal = False


class Trie:
    def __init__(self, strings=None):
        self.root = TrieNode()
        if strings:
            for s in strings:
                self.insert(s)

    def insert(self, s: str):
        n = self.root
        for c in s:
            if c not in n.children:
                n.children[c] = TrieNode()
            n = n.children[c]
        n.is_terminal = True

    def valid_next(self, p: str):
        n = self.root
        for c in p:
            if c not in n.children:
                return frozenset(), False
            n = n.children[c]
        return frozenset(n.children.keys()), n.is_terminal

    def accepts(self, s: str) -> bool:
        n = self.root
        for c in s:
            if c not in n.children:
                return False
            n = n.children[c]
        return n.is_terminal


full_trie = Trie(clean)
_pt: dict = {}


def get_pt(prov: str) -> Trie:
    if prov not in _pt:
        _pt[prov] = Trie(prov_to_c.get(prov, []))
    return _pt[prov]


print("Tries built.", flush=True)


# ── Seq2Seq model ─────────────────────────────────────────────────────────────
class S2S(nn.Module):
    def __init__(self):
        super().__init__()
        D = cfg["D_MODEL"]
        self.src_emb  = nn.Embedding(cfg["SRC_VOCAB"], D, padding_idx=0)
        self.src_pos  = nn.Embedding(cfg["MAX_SRC"], D)
        el = nn.TransformerEncoderLayer(
            D, cfg["N_HEADS"], cfg["D_FF"], .1,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder  = nn.TransformerEncoder(el, cfg["ENC_LAYERS"])
        self.enc_norm = nn.LayerNorm(D)
        self.tgt_emb  = nn.Embedding(cfg["TGT_VOCAB"], D, padding_idx=0)
        self.tgt_pos  = nn.Embedding(cfg["MAX_TGT"], D)
        dl = nn.TransformerDecoderLayer(
            D, cfg["N_HEADS"], cfg["D_FF"], .1,
            batch_first=True, norm_first=True, activation="gelu")
        self.decoder  = nn.TransformerDecoder(dl, cfg["DEC_LAYERS"])
        self.dec_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, cfg["TGT_VOCAB"])

    def encode(self, src):
        B, L = src.shape
        h = (self.src_emb(src)
             + self.src_pos(torch.arange(L, device=src.device)))
        h = self.encoder(h, src_key_padding_mask=(src == 0))
        return self.enc_norm(h), (src == 0)

    def step(self, tgt, mem, sp):
        L = tgt.shape[1]
        cm = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        h = (self.tgt_emb(tgt)
             + self.tgt_pos(torch.arange(L, device=tgt.device)))
        h = self.decoder(h, mem, tgt_mask=cm, memory_key_padding_mask=sp)
        return self.out_proj(self.dec_norm(h))[:, -1, :]


def _load_model() -> S2S:
    m = S2S()
    sf = MODEL_DIR / "model.safetensors"
    pt = MODEL_DIR / "model_best.pt"
    if sf.exists():
        try:
            from safetensors.torch import load_file
            m.load_state_dict(load_file(str(sf)))
            print("Model loaded (safetensors).", flush=True)
            return m
        except Exception as e:
            print(f"safetensors failed ({e}), trying .pt", flush=True)
    if pt.exists():
        m.load_state_dict(
            torch.load(str(pt), map_location="cpu", weights_only=True))
        print("Model loaded (.pt).", flush=True)
        return m
    raise FileNotFoundError(
        f"No model weights in {MODEL_DIR}. "
        "Expected model.safetensors or model_best.pt.")


model = _load_model()
model.eval()


def enc_src(text: str) -> list:
    ids = ([SRC_BOS]
           + [src_ch2id.get(c, SRC_UNK) for c in text[:cfg["MAX_SRC"] - 2]]
           + [SRC_EOS])
    return ids + [SRC_PAD] * (cfg["MAX_SRC"] - len(ids))


def beam_search(mem, sp, trie: Trie, B: int = 5, maxs: int = 96):
    dev   = mem.device
    beams = [(0., "", [TGT_BOS])]
    done  = []
    for _ in range(maxs - 1):
        if not beams:
            break
        nb = []
        for sc, cs, ids in beams:
            vc, it = trie.valid_next(cs)
            if it and not vc:
                done.append((sc, cs))
                continue
            tgt = torch.tensor([ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                lp = F.log_softmax(model.step(tgt, mem, sp)[0], dim=-1)
            cands = []
            if it:
                cands.append((sc + lp[TGT_EOS].item(), cs, ids + [TGT_EOS], True))
            for c in vc:
                if c in tgt_ch2id:
                    cid = tgt_ch2id[c]
                    cands.append((sc + lp[cid].item(), cs + c, ids + [cid], False))
            if not cands:
                if it:
                    done.append((sc, cs))
                continue
            cands.sort(key=lambda x: x[0], reverse=True)
            for ns, nss, ni, d in cands[:B]:
                if d:
                    done.append((ns, nss))
                else:
                    nb.append((ns, nss, ni))
        nb.sort(key=lambda x: x[0], reverse=True)
        beams = nb[:B]
    for sc, s, _ in beams:
        _, it = trie.valid_next(s)
        if it:
            done.append((sc, s))
    if not done:
        return "", 0.
    done.sort(key=lambda x: x[0], reverse=True)
    return done[0][1], done[0][0]


# ── Public API ────────────────────────────────────────────────────────────────
def normalize(raw: str, beam_size: int = 5) -> dict:
    """
    Normalize a Vietnamese address string.

    Args:
        raw:       Raw address string, e.g. "p tan dinh q1 tphcm".
                   Accepts Vietnamese diacritics or ASCII-slugified input.
                   Truncated to 300 characters if longer.
        beam_size: Beam width. Higher = better accuracy, slower (default 5).

    Returns:
        dict:
            canonical    (str)   — normalized address; empty if not found
            valid        (bool)  — True if canonical is in the address database
            confidence   (float) — raw log-prob score (higher = more confident)
            province     (str)   — resolved province name, or None
            ward_hint    (str)   — detected ward slug, or None
            search_space (int)   — number of trie candidates searched
            latency_ms   (float) — wall-clock time in milliseconds
    """
    if not raw or not raw.strip():
        return {
            "canonical": "", "valid": False, "confidence": 0.,
            "province": None, "ward_hint": None,
            "search_space": 0, "latency_ms": 0.,
        }

    raw = raw.strip()[:300]

    t0   = time.perf_counter()
    src  = torch.tensor([enc_src(raw)], dtype=torch.long)
    with torch.no_grad():
        mem, sp = model.encode(src)

    prov      = detect_prov(raw)
    ward_hint = None
    ward_c    = None

    if prov:
        ward_hint, ward_c = detect_ward(raw, prov)
        if ward_c == "numbered":
            return {
                "canonical": "", "valid": False, "confidence": 0.,
                "province": prov, "ward_hint": None,
                "search_space": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1e3, 1),
            }

    if ward_hint and isinstance(ward_c, list) and ward_c:
        trie = Trie(ward_c)
        n    = len(ward_c)
    elif prov and prov_to_c.get(prov):
        trie = get_pt(prov)
        n    = len(prov_to_c[prov])
    else:
        trie = full_trie
        n    = len(clean)

    res, sc = beam_search(mem, sp, trie, B=beam_size)
    ms      = round((time.perf_counter() - t0) * 1e3, 1)

    return {
        "canonical":    res,
        "valid":        bool(res and full_trie.accepts(res)),
        "confidence":   round(float(sc), 4),
        "province":     prov,
        "ward_hint":    ward_hint,
        "search_space": n,
        "latency_ms":   ms,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py \"địa chỉ cần normalize\"")
        sys.exit(1)

    address = " ".join(sys.argv[1:])
    r = normalize(address)
    print(f"Input:       {address}")
    print(f"Canonical:   {r['canonical'] or '(not found)'}")
    print(f"Valid:       {r['valid']}")
    print(f"Province:    {r['province'] or '(unknown)'}")
    print(f"Ward hint:   {r['ward_hint'] or '(none)'}")
    print(f"Space:       {r['search_space']:,} candidates")
    print(f"Latency:     {r['latency_ms']} ms")
