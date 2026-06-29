"""
ML Inference Module — integrates with normalizer.py
Replaces Stage 4 (fuzzy) with trained CharBERT model.
Fallback to rule-based if confidence < threshold.
"""
import json, torch, sys, time
import torch.nn as nn
from pathlib import Path
from unidecode import unidecode

MODEL_DIR = Path("/root/vn_address/model_output")
sys.path.insert(0, "/root/vn_address")

def slug(s): return unidecode(s).lower().strip()

# ── Load config ───────────────────────────────────────────────────────────────
def load_artifacts():
    cfg = json.load(open(MODEL_DIR/"config.json"))

    vocab        = json.load(open(MODEL_DIR/"vocab.json",    encoding="utf-8"))
    street_cls   = json.load(open(MODEL_DIR/"street_classes.json", encoding="utf-8"))
    ward_cls     = json.load(open(MODEL_DIR/"ward_classes.json",   encoding="utf-8"))
    prov_cls     = json.load(open(MODEL_DIR/"prov_classes.json",   encoding="utf-8"))
    s_mask       = torch.load(MODEL_DIR/"street_masks.pt",  map_location="cpu", weights_only=True)
    w_mask       = torch.load(MODEL_DIR/"ward_masks.pt",    map_location="cpu", weights_only=True)
    w2p          = torch.load(MODEL_DIR/"ward_to_prov.pt",  map_location="cpu", weights_only=True)

    ch2id = {c:i for i,c in enumerate(vocab)}
    street_map = {c:i for i,c in enumerate(street_cls)}
    ward_map   = {c:i for i,c in enumerate(ward_cls)}
    prov_map   = {c:i for i,c in enumerate(prov_cls)}

    return cfg, ch2id, street_cls, ward_cls, prov_cls, s_mask, w_mask, w2p

# ── Model (must match training architecture) ──────────────────────────────────
class AddressModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V   = cfg["VOCAB"]
        D   = cfg["D_MODEL"]
        L   = cfg["MAX_LEN"]
        N_P = cfg["N_PROV"]
        N_S = cfg["N_STREET"]
        N_W = cfg["N_WARD"]
        H   = cfg["N_HEADS"]
        NL  = cfg["N_LAYERS"]
        FF  = cfg["D_FF"]

        self.emb  = nn.Embedding(V, D, padding_idx=0)
        self.pos  = nn.Embedding(L, D)
        layer = nn.TransformerEncoderLayer(D,H,FF,0.1,
                    batch_first=True,norm_first=True,activation="gelu")
        self.enc  = nn.TransformerEncoder(layer, NL)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(0.1)
        self.hp   = nn.Linear(D, N_P)
        self.hs   = nn.Linear(D, N_S)
        self.prov_emb = nn.Embedding(N_P, D//4)
        self.hw   = nn.Sequential(
            nn.Linear(D + D//4, D), nn.GELU(),
            nn.LayerNorm(D), nn.Linear(D, N_W)
        )

    def forward(self, x, pid=None):
        B,L = x.shape
        h   = self.drop(self.emb(x) + self.pos(torch.arange(L,device=x.device)))
        h   = self.enc(h, src_key_padding_mask=(x==0))
        cls = self.norm(h)[:,0]
        lp  = self.hp(cls)
        ls  = self.hs(cls)
        if pid is None: pid = lp.argmax(1)
        lw  = self.hw(torch.cat([cls, self.prov_emb(pid)], dim=1))
        return lp, ls, lw

# ── Inference engine ──────────────────────────────────────────────────────────
class MLNormalizer:
    CONF_THRESHOLD = 0.60   # below this → fallback to rule-based

    def __init__(self):
        t0 = time.time()
        self.cfg, self.ch2id, self.street_cls, self.ward_cls, \
            self.prov_cls, self.s_mask, self.w_mask, self.w2p = load_artifacts()

        self.device = torch.device("cpu")   # server is CPU-only
        self.model  = AddressModel(self.cfg)
        state = torch.load(MODEL_DIR/"model_best.pt",
                           map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.MAX_LEN = self.cfg["MAX_LEN"]
        print(f"MLNormalizer loaded in {(time.time()-t0)*1000:.0f}ms")

    def _encode(self, text: str) -> torch.Tensor:
        BOS, EOS, UNK, PAD = 2, 3, 1, 0
        ids = [BOS] + [self.ch2id.get(c, UNK) for c in text[:self.MAX_LEN-2]] + [EOS]
        ids += [PAD] * (self.MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def predict(self, raw: str) -> dict:
        """
        Returns:
          street:     canonical street name (may be empty)
          ward:       canonical ward+province string
          confidence: float 0-1
          valid:      bool (ward in FST)
        """
        t0 = time.time()
        x  = self._encode(raw)

        with torch.no_grad():
            lp, ls, lw = self.model(x, pid=None)

        # Province
        pp       = lp.argmax(1)
        prov_str = self.prov_cls[pp.item()]

        # Apply masks
        ls_m = ls.masked_fill(~self.s_mask[pp], -1e9)
        lw_m = lw.masked_fill(~self.w_mask[pp], -1e9)

        sp = ls_m.argmax(1)
        wp = lw_m.argmax(1)

        # Confidence = ward probability (main metric)
        ward_prob  = torch.softmax(lw_m, dim=1)[0, wp].item()
        street_str = self.street_cls[sp.item()] if sp.item() > 0 else ""
        ward_str   = self.ward_cls[wp.item()]   if wp.item() > 0 else ""

        # Validate: ward must exist and province must match
        valid = (ward_str != "" and
                 ward_prob >= self.CONF_THRESHOLD and
                 prov_str in ward_str)

        return {
            "street":     street_str,
            "ward":       ward_str,
            "province":   prov_str,
            "confidence": round(ward_prob, 4),
            "valid":      valid,
            "latency_ms": round((time.time()-t0)*1000, 2),
        }

    def batch_predict(self, texts: list[str], batch_size=128) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            xs    = torch.cat([self._encode(t) for t in batch])
            with torch.no_grad():
                lp, ls, lw = self.model(xs)
            pp   = lp.argmax(1)
            ls_m = torch.stack([ls[j].masked_fill(~self.s_mask[pp[j]],-1e9) for j in range(len(batch))])
            lw_m = torch.stack([lw[j].masked_fill(~self.w_mask[pp[j]],-1e9) for j in range(len(batch))])
            sp   = ls_m.argmax(1)
            wp   = lw_m.argmax(1)
            ward_probs = torch.softmax(lw_m, dim=1).gather(1, wp.unsqueeze(1)).squeeze(1)

            for j in range(len(batch)):
                sv   = self.street_cls[sp[j].item()] if sp[j].item()>0 else ""
                wv   = self.ward_cls[wp[j].item()]   if wp[j].item()>0 else ""
                pv   = self.prov_cls[pp[j].item()]
                conf = round(ward_probs[j].item(), 4)
                results.append({
                    "street": sv, "ward": wv, "province": pv,
                    "confidence": conf,
                    "valid": bool(wv and conf >= self.CONF_THRESHOLD and pv in wv),
                })
        return results

# Singleton
_instance = None
def get_ml_normalizer():
    global _instance
    if _instance is None:
        _instance = MLNormalizer()
    return _instance
