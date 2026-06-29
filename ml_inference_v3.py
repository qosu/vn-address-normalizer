"""
VN Address ML Inference v3 — Province + Ward anchored beam search
Fixes vs v2:
  1. Province-constrained ward lookup
  2. Ward anchoring narrows search to ~10-50 strings
  3. Numbered ward detection → immediate reject
  4. 187K clean canonicals (thien0291 district expansion)
"""
import json, re, time, sys
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from unidecode import unidecode

MODEL_DIR = Path('/root/vn_address/model_v3_final')
sys.path.insert(0, '/root/vn_address')

def slug(s): return unidecode(s).lower().strip()

# ── Load artifacts ────────────────────────────────────────────────────────────
cfg       = json.load(open(MODEL_DIR/'config.json'))
src_vocab = json.load(open(MODEL_DIR/'src_vocab.json', encoding='utf-8'))
tgt_vocab = json.load(open(MODEL_DIR/'tgt_vocab.json', encoding='utf-8'))
all_canonicals = json.load(open(MODEL_DIR/'valid_canonicals.json', encoding='utf-8'))
src_ch2id = {c:i for i,c in enumerate(src_vocab)}
tgt_ch2id = {c:i for i,c in enumerate(tgt_vocab)}
tgt_id2ch = {i:c for c,i in tgt_ch2id.items()}
SRC_PAD,SRC_UNK,SRC_BOS,SRC_EOS = 0,1,2,3
TGT_PAD,TGT_UNK,TGT_BOS,TGT_EOS = 0,1,2,3

from fst import load as fst_load
fst = fst_load()
from normalizer import get_indexes
idx = get_indexes()

valid_prov_set = {meta['name'] for meta in fst.prov_meta.values()}
valid_ward_set = {meta['canonical'] for meta in fst.ward_meta.values()}

# ── Clean canonicals ──────────────────────────────────────────────────────────
def is_clean(s):
    parts = [p.strip() for p in s.split(',')]
    if len(parts) < 2 or len(parts) > 3: return False
    if len(s) > 90: return False
    if parts[-1].strip() not in valid_prov_set: return False
    if f"{parts[-2].strip()}, {parts[-1].strip()}" not in valid_ward_set: return False
    if len(parts) == 3:
        st = parts[0].strip()
        if len(st) > 60: return False
        if re.search(r'\b(block|căn hộ|chung cư|ecogreen|toà nhà)\b', st, re.I): return False
    return True

clean = [c for c in all_canonicals if is_clean(c)]
print(f"Canonicals: {len(all_canonicals):,} → {len(clean):,} clean")

# ── Indexes ───────────────────────────────────────────────────────────────────
prov_to_c = defaultdict(list)
pw_to_c   = defaultdict(list)   # (prov, ward_slug) → canonicals

for c in clean:
    parts = [p.strip() for p in c.split(',')]
    prov = parts[-1]
    prov_to_c[prov].append(c)
    ward_part = parts[-2]
    for ws in [slug(ward_part),
               re.sub(r'^(phuong|xa|thi tran|dac khu)\s+','',slug(ward_part)).strip()]:
        pw_to_c[(prov, ws)].append(c)

# ── Trie ──────────────────────────────────────────────────────────────────────
class TrieNode:
    __slots__=('children','is_terminal')
    def __init__(self): self.children={}; self.is_terminal=False

class Trie:
    def __init__(self,strings=None):
        self.root=TrieNode()
        if strings:
            for s in strings: self.insert(s)
    def insert(self,s):
        n=self.root
        for c in s:
            if c not in n.children: n.children[c]=TrieNode()
            n=n.children[c]
        n.is_terminal=True
    def valid_next(self,p):
        n=self.root
        for c in p:
            if c not in n.children: return frozenset(),False
            n=n.children[c]
        return frozenset(n.children.keys()),n.is_terminal
    def accepts(self,s):
        n=self.root
        for c in s:
            if c not in n.children: return False
            n=n.children[c]
        return n.is_terminal

full_trie = Trie(clean)
_pt = {}
def get_pt(prov): 
    if prov not in _pt: _pt[prov] = Trie(prov_to_c.get(prov,[]))
    return _pt[prov]

print("Tries built.", flush=True)

# ── Province resolver ─────────────────────────────────────────────────────────
_OLD = {"hcm":"ho chi minh","tphcm":"ho chi minh","saigon":"ho chi minh",
        "sai gon":"ho chi minh","hanoi":"ha noi","ha giang":"tuyen quang",
        "yen bai":"lao cai","bac kan":"thai nguyen","vinh phuc":"phu tho"}
ps = {}
for pc,meta in fst.prov_meta.items():
    ps[slug(meta['name'])] = meta['name']
    ps[re.sub(r'^(tinh|thanh pho|tp\.?)\s*','',slug(meta['name'])).strip()] = meta['name']

def _resolve_prov(ts):
    ts2=re.sub(r'^(tinh|tp\.?\s*|thanh pho)\s+','',ts).strip()
    ts3=re.sub(r'[.\s]','',ts)
    for key in [ts,ts2,ts3]:
        if key in ps: return ps[key]
        alias=_OLD.get(key)
        if alias:
            for k,v in ps.items():
                if alias in k: return v
    for k,v in ps.items():
        if ts2 and len(ts2)>2 and (ts2 in k or k in ts2): return v
    return None

def detect_prov(raw):
    # Parse components first to extract province token
    from normalizer import _parse_no_comma, _extract, _slug
    comps = _extract(raw) if ',' in raw else _parse_no_comma(raw)
    for field in ['province', 'district_hint']:
        v = comps.get(field)
        if v:
            r = _resolve_prov(slug(v))
            if r: return r
    return _resolve_prov(slug(raw))

# ── Ward hint extractor ───────────────────────────────────────────────────────
_WS = re.compile(r'\b(?:phuong|p\.|p\s|xa|x\.)\s*([a-z0-9][a-z0-9\s]{1,40})',re.I)
_NUM = re.compile(r'^\d{1,3}$')

def detect_ward(raw, prov):
    m = _WS.search(slug(raw))
    if not m: return None, None
    words = m.group(1).strip().split()
    for n in range(min(4,len(words)),0,-1):
        cand = ' '.join(words[:n])
        if _NUM.match(cand.split()[0] if cand.split() else cand): return None, 'numbered'
        for ws in [cand, re.sub(r'^(phuong|xa|thi tran)\s+','',cand).strip()]:
            if prov:
                canons = pw_to_c.get((prov,ws),[])
                if canons: return ws, canons
            rb = idx.slug_to_canon.get(ws,[]) + idx.legacy_idx.get(ws,[])
            if rb:
                pf = [c for c in rb if prov and prov in c] if prov else rb
                if pf: return ws, pf
    return None, None

# ── Model ─────────────────────────────────────────────────────────────────────
class S2S(nn.Module):
    def __init__(self):
        super().__init__()
        D=cfg['D_MODEL']
        self.src_emb=nn.Embedding(cfg['SRC_VOCAB'],D,padding_idx=0)
        self.src_pos=nn.Embedding(cfg['MAX_SRC'],D)
        el=nn.TransformerEncoderLayer(D,cfg['N_HEADS'],cfg['D_FF'],.1,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(el,cfg['ENC_LAYERS'])
        self.enc_norm=nn.LayerNorm(D)
        self.tgt_emb=nn.Embedding(cfg['TGT_VOCAB'],D,padding_idx=0)
        self.tgt_pos=nn.Embedding(cfg['MAX_TGT'],D)
        dl=nn.TransformerDecoderLayer(D,cfg['N_HEADS'],cfg['D_FF'],.1,batch_first=True,norm_first=True,activation='gelu')
        self.decoder=nn.TransformerDecoder(dl,cfg['DEC_LAYERS'])
        self.dec_norm=nn.LayerNorm(D)
        self.out_proj=nn.Linear(D,cfg['TGT_VOCAB'])

    def encode(self,src):
        B,L=src.shape
        h=self.src_emb(src)+self.src_pos(torch.arange(L,device=src.device))
        h=self.encoder(h,src_key_padding_mask=(src==0))
        return self.enc_norm(h),(src==0)

    def step(self,tgt,mem,sp):
        L=tgt.shape[1]
        cm=nn.Transformer.generate_square_subsequent_mask(L,device=tgt.device)
        h=self.tgt_emb(tgt)+self.tgt_pos(torch.arange(L,device=tgt.device))
        h=self.decoder(h,mem,tgt_mask=cm,memory_key_padding_mask=sp)
        return self.out_proj(self.dec_norm(h))[:,-1,:]

model=S2S()
model.load_state_dict(torch.load(MODEL_DIR/'model_best.pt',map_location='cpu',weights_only=True))
model.eval()
print("Model loaded.", flush=True)

def enc_src(text):
    ids=[SRC_BOS]+[src_ch2id.get(c,SRC_UNK) for c in text[:cfg['MAX_SRC']-2]]+[SRC_EOS]
    return ids+[SRC_PAD]*(cfg['MAX_SRC']-len(ids))

def beam(mem,sp,trie,B=5,maxs=96):
    dev=mem.device
    beams=[(0.,"",[ TGT_BOS])]; done=[]
    for _ in range(maxs-1):
        if not beams: break
        nb=[]
        for sc,cs,ids in beams:
            vc,it=trie.valid_next(cs)
            if it and not vc: done.append((sc,cs)); continue
            tgt=torch.tensor([ids],dtype=torch.long,device=dev)
            with torch.no_grad(): lp=F.log_softmax(model.step(tgt,mem,sp)[0],dim=-1)
            cands=[]
            if it: cands.append((sc+lp[TGT_EOS].item(),cs,ids+[TGT_EOS],True))
            for c in vc:
                if c in tgt_ch2id:
                    cid=tgt_ch2id[c]
                    cands.append((sc+lp[cid].item(),cs+c,ids+[cid],False))
            if not cands:
                if it: done.append((sc,cs))
                continue
            cands.sort(key=lambda x:x[0],reverse=True)
            for ns,nss,ni,d in cands[:B]:
                if d: done.append((ns,nss))
                else: nb.append((ns,nss,ni))
        nb.sort(key=lambda x:x[0],reverse=True); beams=nb[:B]
    for sc,s,_ in beams:
        _,it=trie.valid_next(s)
        if it: done.append((sc,s))
    if not done: return "",0.
    done.sort(key=lambda x:x[0],reverse=True)
    return done[0][1],done[0][0]

def normalize(raw, beam_size=5):
    t0=time.perf_counter()
    src=torch.tensor([enc_src(raw)],dtype=torch.long)
    with torch.no_grad(): mem,sp=model.encode(src)

    prov=detect_prov(raw)
    ward_hint,ward_c=None,None
    if prov:
        ward_hint,ward_c=detect_ward(raw,prov)
        if ward_c=='numbered':
            return {"canonical":"","valid":False,"confidence":0.,"latency_ms":round((time.perf_counter()-t0)*1e3,1)}

    if ward_hint and isinstance(ward_c,list) and ward_c:
        trie=Trie(ward_c); n=len(ward_c)
    elif prov and prov_to_c.get(prov):
        trie=get_pt(prov); n=len(prov_to_c[prov])
    else:
        trie=full_trie; n=len(clean)

    res,sc=beam(mem,sp,trie,B=beam_size)
    ms=round((time.perf_counter()-t0)*1e3,1)
    return {"canonical":res,"province":prov,"ward_hint":ward_hint,
            "search_space":n,"valid":bool(res and full_trie.accepts(res)),"latency_ms":ms}

# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__=='__main__':
    TESTS=[
        ("45 tran binh trong p tan dinh q1 hcm",
         "Trần Bình Trọng, Phường Tân Định, Thành phố Hồ Chí Minh"),
        ("phuong ba dinh ha noi",
         "Phường Ba Đình, Thành phố Hà Nội"),
        ("XA CU CHI TP HO CHI MINH",
         "Xã Củ Chi, Thành phố Hồ Chí Minh"),
        ("P. Ben Nghe Q1 HCM",
         "Phường Sài Gòn, Thành phố Hồ Chí Minh"),
        ("Xa thuong lam tinh Ha Giang",
         "Xã Thượng Lâm, Tỉnh Tuyên Quang"),
        ("Phuong 14 Quan 5 HCM",""),
        ("nguyen trai phuong thuong dinh thanh xuan ha noi",
         "Nguyễn Trãi, Phường Thượng Đình, Thành phố Hà Nội"),
        ("duong le loi phuong ben nghe quan 1 tphcm",
         "Đường Lê Lợi, Phường Sài Gòn, Thành phố Hồ Chí Minh"),
    ]
    ok=0
    for inp,exp in TESTS:
        r=normalize(inp)
        correct=(r['canonical']==exp) if exp else (not r['valid'])
        ok+=correct
        sym="✓" if correct else "✗"
        print(f"{sym} {inp[:40]:<40} → {r['canonical'][:40]:<40} [{r.get('ward_hint','?')}] {r.get('search_space',0)}c {r['latency_ms']}ms")
    print(f"\n{ok}/{len(TESTS)}")
