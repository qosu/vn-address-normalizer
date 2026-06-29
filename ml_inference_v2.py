"""
VN Address ML Inference v2 — Province-anchored beam search
3 fixes vs v1:
  1. Clean valid_canonicals (remove dirty Google geocoding results)
  2. Province-anchored trie (filter to 1 province = 1K strings instead of 41K)
  3. Beam search size=5 (vs greedy = beam size 1)

No retraining needed.
"""
import json, torch, time, re, sys
import torch.nn as nn
from collections import defaultdict
from unidecode import unidecode
from pathlib import Path

MODEL_DIR = Path('/root/vn_address/model_v3_final')
sys.path.insert(0, '/root/vn_address')

def slug(s): return unidecode(s).lower().strip()

# ── Load artifacts ────────────────────────────────────────────────────────────
cfg       = json.load(open(MODEL_DIR/'config.json'))
src_vocab = json.load(open(MODEL_DIR/'src_vocab.json', encoding='utf-8'))
tgt_vocab = json.load(open(MODEL_DIR/'tgt_vocab.json', encoding='utf-8'))
raw_canonicals = json.load(open(MODEL_DIR/'valid_canonicals.json', encoding='utf-8'))

src_ch2id = {c:i for i,c in enumerate(src_vocab)}
tgt_ch2id = {c:i for i,c in enumerate(tgt_vocab)}
tgt_id2ch = {i:c for c,i in tgt_ch2id.items()}
SRC_PAD,SRC_UNK,SRC_BOS,SRC_EOS = 0,1,2,3
TGT_PAD,TGT_UNK,TGT_BOS,TGT_EOS = 0,1,2,3

# ── Fix 1: Clean valid canonicals ─────────────────────────────────────────────
# Load VBHC ground truth for validation
from fst import load as fst_load
fst = fst_load()

valid_ward_set = set()
for wc, meta in fst.ward_meta.items():
    valid_ward_set.add(meta['canonical'])  # "Phường X, Tỉnh Y"

valid_prov_set = set()
for pc, meta in fst.prov_meta.items():
    valid_prov_set.add(meta['name'])  # "Thành phố Hồ Chí Minh"

def is_clean_canonical(s):
    """A canonical must be VBHC-validated structure."""
    parts = [p.strip() for p in s.split(',')]
    if len(parts) < 2 or len(parts) > 3: return False
    if len(s) > 90: return False
    # Last part must be a valid province
    prov = parts[-1].strip()
    if prov not in valid_prov_set: return False
    # Second-to-last must be a valid ward (as standalone)
    ward_part = parts[-2].strip()
    ward_canon = f"{ward_part}, {prov}"
    if ward_canon not in valid_ward_set: return False
    # If 3 parts: first is street name (no admin keywords, reasonable length)
    if len(parts) == 3:
        street = parts[0].strip()
        if len(street) > 60: return False
        if re.search(r'\b(block|căn hộ|chung cư|ecogreen|toà nhà|tầng)\b',
                     street, re.I): return False
    return True

clean_canonicals = [c for c in raw_canonicals if is_clean_canonical(c)]
print(f"Canonicals: {len(raw_canonicals):,} → {len(clean_canonicals):,} clean")

# ── Province index ────────────────────────────────────────────────────────────
# province_name → list of canonicals
prov_canonicals = defaultdict(list)
for c in clean_canonicals:
    parts = [p.strip() for p in c.split(',')]
    prov = parts[-1]
    prov_canonicals[prov].append(c)

print(f"Provinces with canonicals: {len(prov_canonicals)}")

# ── Trie ──────────────────────────────────────────────────────────────────────
class TrieNode:
    __slots__ = ('children', 'is_terminal')
    def __init__(self): self.children = {}; self.is_terminal = False

class Trie:
    def __init__(self, strings=None):
        self.root = TrieNode()
        if strings:
            for s in strings: self.insert(s)

    def insert(self, s):
        n = self.root
        for c in s:
            if c not in n.children: n.children[c] = TrieNode()
            n = n.children[c]
        n.is_terminal = True

    def valid_next(self, prefix):
        n = self.root
        for c in prefix:
            if c not in n.children: return frozenset(), False
            n = n.children[c]
        return frozenset(n.children.keys()), n.is_terminal

    def accepts(self, s):
        n = self.root
        for c in s:
            if c not in n.children: return False
            n = n.children[c]
        return n.is_terminal

# Full trie (fallback)
print("Building full trie...", flush=True)
full_trie = Trie(clean_canonicals)

# Per-province tries (cached)
_prov_tries = {}
def get_prov_trie(prov_name):
    if prov_name not in _prov_tries:
        _prov_tries[prov_name] = Trie(prov_canonicals.get(prov_name, []))
    return _prov_tries[prov_name]

print(f"Trie built. {len(clean_canonicals):,} valid strings", flush=True)

# ── Province resolver (from normalizer) ──────────────────────────────────────
from normalizer import get_indexes, _parse_no_comma, _extract, _slug, _OLD_PROV

def detect_province(raw):
    """Use existing rule-based to detect province from input."""
    idx = get_indexes()
    comps = _extract(raw) if ',' in raw else _parse_no_comma(raw)
    rpc = None
    for fv in [comps.get("province"), comps.get("district_hint")]:
        if fv and not rpc:
            rpc = idx.resolve_province(_slug(fv))
    if rpc:
        meta = fst.prov_meta.get(rpc)
        if meta: return meta['name']
    return None

# ── Model ─────────────────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        D = cfg['D_MODEL']
        self.src_emb  = nn.Embedding(cfg['SRC_VOCAB'], D, padding_idx=0)
        self.src_pos  = nn.Embedding(cfg['MAX_SRC'], D)
        el = nn.TransformerEncoderLayer(D, cfg['N_HEADS'], cfg['D_FF'], 0.1,
                                        batch_first=True, norm_first=True, activation='gelu')
        self.encoder  = nn.TransformerEncoder(el, cfg['ENC_LAYERS'])
        self.enc_norm = nn.LayerNorm(D)
        self.tgt_emb  = nn.Embedding(cfg['TGT_VOCAB'], D, padding_idx=0)
        self.tgt_pos  = nn.Embedding(cfg['MAX_TGT'], D)
        dl = nn.TransformerDecoderLayer(D, cfg['N_HEADS'], cfg['D_FF'], 0.1,
                                        batch_first=True, norm_first=True, activation='gelu')
        self.decoder  = nn.TransformerDecoder(dl, cfg['DEC_LAYERS'])
        self.dec_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, cfg['TGT_VOCAB'])

    def encode(self, src):
        B, L = src.shape
        h = self.src_emb(src) + self.src_pos(torch.arange(L, device=src.device))
        h = self.encoder(h, src_key_padding_mask=(src == 0))
        return self.enc_norm(h), (src == 0)

    def decode_step(self, tgt_ids, memory, src_pad):
        L = tgt_ids.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tgt_ids.device)
        h = self.tgt_emb(tgt_ids) + self.tgt_pos(torch.arange(L, device=tgt_ids.device))
        h = self.decoder(h, memory,
                         tgt_mask=causal,
                         memory_key_padding_mask=src_pad)
        return self.out_proj(self.dec_norm(h))[:, -1, :]  # (B, TGT_VOCAB)

model = Seq2Seq()
state = torch.load(MODEL_DIR/'model_best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()
print("Model loaded.", flush=True)

def encode_src(text):
    ids = [SRC_BOS]+[src_ch2id.get(c, SRC_UNK) for c in text[:cfg['MAX_SRC']-2]]+[SRC_EOS]
    return ids + [SRC_PAD] * (cfg['MAX_SRC'] - len(ids))

# ── Fix 3: Beam search with FST constraint ────────────────────────────────────
import torch.nn.functional as F

def beam_search_fst(memory, src_pad, trie, beam_size=5, max_steps=96):
    """
    Beam search with FST constraint.
    Returns best complete canonical string.
    """
    device = memory.device

    # Each beam: (log_prob, current_string, tgt_ids_list)
    beams = [(0.0, "", [TGT_BOS])]
    completed = []

    for step in range(max_steps - 1):
        if not beams: break
        new_beams = []

        # Batch all beam tgt_ids for efficiency
        # (expand memory for each beam)
        B = len(beams)
        mem_exp   = memory.expand(B, -1, -1)
        pad_exp   = src_pad.expand(B, -1)
        max_len   = max(len(b[2]) for b in beams)
        tgt_batch = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, (_, _, ids) in enumerate(beams):
            tgt_batch[i, :len(ids)] = torch.tensor(ids, device=device)

        # Decode step
        with torch.no_grad():
            logits_batch = []
            for i, (_, _, ids) in enumerate(beams):
                tgt = torch.tensor([ids], dtype=torch.long, device=device)
                logit = model.decode_step(tgt, memory, src_pad)  # (1, TGT_VOCAB)
                logits_batch.append(logit[0])
        logits_batch = torch.stack(logits_batch)  # (B, TGT_VOCAB)

        # Log probs
        log_probs = F.log_softmax(logits_batch, dim=-1)  # (B, TGT_VOCAB)

        for i, (score, current_str, ids) in enumerate(beams):
            valid_chars, is_terminal = trie.valid_next(current_str)

            if is_terminal and not valid_chars:
                completed.append((score, current_str))
                continue

            candidates = []
            # Option: EOS if terminal
            if is_terminal:
                eos_score = score + log_probs[i, TGT_EOS].item()
                candidates.append((eos_score, current_str, ids + [TGT_EOS], True))

            # Expand valid next chars
            for c in valid_chars:
                if c not in tgt_ch2id: continue
                cid = tgt_ch2id[c]
                new_score = score + log_probs[i, cid].item()
                candidates.append((new_score, current_str + c, ids + [cid], False))

            if not candidates:
                # Dead end — add as-is if terminal
                if is_terminal: completed.append((score, current_str))
                continue

            # Take top beam_size
            candidates.sort(key=lambda x: x[0], reverse=True)
            for new_score, new_str, new_ids, is_done in candidates[:beam_size]:
                if is_done:
                    completed.append((new_score, new_str))
                else:
                    new_beams.append((new_score, new_str, new_ids))

        if not new_beams and not completed:
            break

        # Keep top beam_size active beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

    # Also add any incomplete beams that are terminal
    for score, s, ids in beams:
        _, is_term = trie.valid_next(s)
        if is_term:
            completed.append((score, s))

    if not completed:
        return "", 0.0

    completed.sort(key=lambda x: x[0], reverse=True)
    best_score, best_str = completed[0]
    # Convert log_prob to confidence (0-1)
    conf = min(1.0, max(0.0, (best_score + 100) / 100))  # rough normalization
    return best_str, conf


def normalize_ml(raw_text, beam_size=5):
    """Full normalization with province anchoring + beam search."""
    t0 = time.perf_counter()

    # Fix 2: Province anchoring
    detected_prov = detect_province(raw_text)
    if detected_prov and prov_canonicals.get(detected_prov):
        trie = get_prov_trie(detected_prov)
        n_candidates = len(prov_canonicals[detected_prov])
    else:
        trie = full_trie
        n_candidates = len(clean_canonicals)

    # Encode input
    src = torch.tensor([encode_src(raw_text)], dtype=torch.long)

    with torch.no_grad():
        memory, src_pad = model.encode(src)

    # Beam search with FST
    result, conf = beam_search_fst(memory, src_pad, trie, beam_size=beam_size)

    ms = (time.perf_counter() - t0) * 1000
    return {
        "canonical":  result,
        "confidence": round(conf, 3),
        "province":   detected_prov,
        "n_candidates": n_candidates,
        "valid":      bool(result and full_trie.accepts(result)),
        "latency_ms": round(ms, 1),
    }
