"""
VN Address Normalizer — CharBERT Training
Kaggle T4×2 | ~6h | 1.4M pairs | 3-head classification

Architecture:
  Char embedding (256d) → 4-layer Transformer → 3 heads
  Head 1: Province (34)
  Head 2: Street  (~25K, masked by province)
  Head 3: Ward    (3321, masked by province)

Run on Kaggle: accelerator=GPU T4x2, persistent=off
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import json, re, time, os, pickle, random
from pathlib import Path
from collections import defaultdict
from unidecode import unidecode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    # Paths (Kaggle)
    TRAIN_FILE  = "/kaggle/input/vn-address-training-2026/train.jsonl.gz"
    VAL_FILE    = "/kaggle/input/vn-address-training-2026/val.jsonl.gz"
    OUTPUT_DIR  = "/kaggle/working"

    # Model
    MAX_LEN     = 128       # max input characters
    D_MODEL     = 256       # transformer hidden size
    N_HEADS     = 4         # attention heads
    N_LAYERS    = 4         # transformer layers
    D_FF        = 1024      # feedforward dim
    DROPOUT     = 0.1

    # Training
    BATCH_SIZE  = 256
    GRAD_ACCUM  = 4         # effective batch = 1024
    LR          = 3e-4
    EPOCHS      = 5
    WARMUP_STEPS= 1000
    MAX_GRAD_NORM = 1.0
    FP16        = True

    # Loss weights
    W_PROVINCE  = 0.2       # rule-based handles this well already
    W_STREET    = 0.4
    W_WARD      = 0.4

CFG = Config()

# ── Vocabulary ────────────────────────────────────────────────────────────────
class CharVocab:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    def __init__(self):
        self.ch2id = {c: i for i, c in enumerate(self.SPECIAL)}
        self.id2ch = list(self.SPECIAL)

    def build(self, texts, min_freq=2):
        from collections import Counter
        freq = Counter(c for t in texts for c in t)
        for c, n in sorted(freq.items()):
            if n >= min_freq and c not in self.ch2id:
                self.ch2id[c] = len(self.id2ch)
                self.id2ch.append(c)
        print(f"Vocab size: {len(self.id2ch)}")
        return self

    def encode(self, text, max_len=128):
        ids = [self.BOS] + [self.ch2id.get(c, self.UNK) for c in text[:max_len-2]] + [self.EOS]
        pad = max_len - len(ids)
        return ids + [self.PAD] * pad

    def __len__(self): return len(self.id2ch)
    def save(self, path): json.dump({"ch2id": self.ch2id, "id2ch": self.id2ch}, open(path,'w'))
    @classmethod
    def load(cls, path):
        v = cls(); d = json.load(open(path))
        v.ch2id = d['ch2id']; v.id2ch = d['id2ch']; return v

# ── Label encoders ────────────────────────────────────────────────────────────
class LabelEncoder:
    def __init__(self): self.classes_ = []; self._map = {}
    def fit(self, labels):
        unique = sorted(set(labels))
        self.classes_ = ['<UNK>'] + unique
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self
    def transform(self, label): return self._map.get(label, 0)
    def inverse(self, idx): return self.classes_[idx] if idx < len(self.classes_) else ''
    def __len__(self): return len(self.classes_)
    def save(self, path): json.dump(self.classes_, open(path,'w',encoding='utf-8'), ensure_ascii=False)
    @classmethod
    def load(cls, path):
        e = cls(); e.classes_ = json.load(open(path,encoding='utf-8'))
        e._map = {c: i for i, c in enumerate(e.classes_)}; return e

# ── Province-Street and Province-Ward masks ───────────────────────────────────
def build_masks(pairs, street_enc, ward_enc, prov_enc):
    """
    Returns:
      prov_to_street_mask[prov_id] = bool tensor (n_streets,)
      prov_to_ward_mask[prov_id]   = bool tensor (n_wards,)
    """
    prov_streets = defaultdict(set)
    prov_wards   = defaultdict(set)
    for p in pairs:
        pid = prov_enc.transform(p.get('province', ''))
        sid = street_enc.transform(p.get('street', ''))
        wid = ward_enc.transform(p.get('ward_canonical', ''))
        if pid and sid: prov_streets[pid].add(sid)
        if pid and wid: prov_wards[pid].add(wid)

    n_prov   = len(prov_enc)
    n_street = len(street_enc)
    n_ward   = len(ward_enc)

    s_masks = torch.zeros(n_prov, n_street, dtype=torch.bool)
    w_masks = torch.zeros(n_prov, n_ward,   dtype=torch.bool)
    s_masks[:, 0] = True   # UNK always valid
    w_masks[:, 0] = True

    for pid, sids in prov_streets.items():
        for sid in sids: s_masks[pid, sid] = True
    for pid, wids in prov_wards.items():
        for wid in wids: w_masks[pid, wid] = True

    # Fallback: if province has no streets/wards, allow all
    for pid in range(n_prov):
        if s_masks[pid].sum() <= 1: s_masks[pid] = True
        if w_masks[pid].sum() <= 1: w_masks[pid] = True

    print(f"Street mask avg coverage: {s_masks.float().mean():.3f}")
    print(f"Ward mask avg coverage  : {w_masks.float().mean():.3f}")
    return s_masks, w_masks

# ── Dataset ───────────────────────────────────────────────────────────────────
def extract_province(ward_canonical):
    """'Phường X, Thành phố Y' → 'Thành phố Y'"""
    parts = ward_canonical.split(', ')
    return parts[-1] if len(parts) >= 2 else ''

class AddressDataset(Dataset):
    def __init__(self, pairs, vocab, street_enc, ward_enc, prov_enc):
        self.pairs      = pairs
        self.vocab      = vocab
        self.street_enc = street_enc
        self.ward_enc   = ward_enc
        self.prov_enc   = prov_enc

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        inp  = torch.tensor(self.vocab.encode(p['input'], CFG.MAX_LEN), dtype=torch.long)
        ward = p.get('ward_canonical', '')
        prov = extract_province(ward)
        return {
            'input':    inp,
            'street_id':torch.tensor(self.street_enc.transform(p.get('street','')),    dtype=torch.long),
            'ward_id':  torch.tensor(self.ward_enc.transform(ward),                     dtype=torch.long),
            'prov_id':  torch.tensor(self.prov_enc.transform(prov),                     dtype=torch.long),
        }

# ── Model ─────────────────────────────────────────────────────────────────────
class AddressNormalizer(nn.Module):
    def __init__(self, vocab_size, n_street, n_ward, n_prov):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, CFG.D_MODEL, padding_idx=0)
        self.pos   = nn.Embedding(CFG.MAX_LEN, CFG.D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=CFG.D_MODEL, nhead=CFG.N_HEADS,
            dim_feedforward=CFG.D_FF, dropout=CFG.DROPOUT,
            batch_first=True, norm_first=True,   # pre-norm more stable
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=CFG.N_LAYERS)
        self.norm    = nn.LayerNorm(CFG.D_MODEL)
        self.drop    = nn.Dropout(CFG.DROPOUT)

        # 3 classification heads
        self.head_prov   = nn.Linear(CFG.D_MODEL, n_prov)
        self.head_street = nn.Linear(CFG.D_MODEL, n_street)
        self.head_ward   = nn.Linear(CFG.D_MODEL, n_ward)

    def encode(self, x):
        """x: (B, L) token ids → (B, D) CLS representation"""
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)
        mask = (x == 0)    # padding mask
        h    = self.drop(self.embed(x) + self.pos(pos))
        h    = self.encoder(h, src_key_padding_mask=mask)
        h    = self.norm(h)
        return h[:, 0]     # CLS token (position 0 = BOS)

    def forward(self, x):
        cls = self.encode(x)
        return (self.head_prov(cls),
                self.head_street(cls),
                self.head_ward(cls))

# ── Training ──────────────────────────────────────────────────────────────────
def load_pairs(path):
    import gzip
    pairs = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs):,} pairs from {path}")
    return pairs

def get_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return base_lr * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2

def train():
    print("="*60)
    print("VN Address Normalizer — Training")
    print("="*60)

    # Load data
    train_pairs = load_pairs(CFG.TRAIN_FILE)
    val_pairs   = load_pairs(CFG.VAL_FILE)

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab = CharVocab().build([p['input'] for p in train_pairs], min_freq=2)

    # Build label encoders
    print("Building label encoders...")
    all_streets = [p.get('street','') for p in train_pairs]
    all_wards   = [p.get('ward_canonical','') for p in train_pairs]
    all_provs   = [extract_province(w) for w in all_wards]

    street_enc = LabelEncoder().fit(all_streets)
    ward_enc   = LabelEncoder().fit(all_wards)
    prov_enc   = LabelEncoder().fit(all_provs)

    print(f"  Streets: {len(street_enc):,}")
    print(f"  Wards  : {len(ward_enc):,}")
    print(f"  Provs  : {len(prov_enc):,}")

    # Build province masks
    print("Building province masks...")
    s_masks, w_masks = build_masks(train_pairs, street_enc, ward_enc, prov_enc)

    # Datasets
    train_ds = AddressDataset(train_pairs, vocab, street_enc, ward_enc, prov_enc)
    val_ds   = AddressDataset(val_pairs,   vocab, street_enc, ward_enc, prov_enc)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE*2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    model = AddressNormalizer(
        vocab_size=len(vocab),
        n_street=len(street_enc),
        n_ward=len(ward_enc),
        n_prov=len(prov_enc),
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    s_masks = s_masks.to(device)
    w_masks = w_masks.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LR,
        weight_decay=0.01, betas=(0.9, 0.98)
    )
    total_steps = len(train_loader) * CFG.EPOCHS // CFG.GRAD_ACCUM
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    scaler = GradScaler(enabled=CFG.FP16)

    ce = nn.CrossEntropyLoss(ignore_index=0)   # ignore UNK label

    # ── Training loop ─────────────────────────────────────────────────────────
    best_ward_acc = 0.0
    global_step   = 0

    for epoch in range(CFG.EPOCHS):
        model.train()
        t0 = time.time()
        total_loss = total_street = total_ward = total_prov = 0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            inp      = batch['input'].to(device)
            sid_true = batch['street_id'].to(device)
            wid_true = batch['ward_id'].to(device)
            pid_true = batch['prov_id'].to(device)

            with autocast(enabled=CFG.FP16):
                logits_prov, logits_street, logits_ward = model(inp)

                loss_prov   = ce(logits_prov,   pid_true)
                loss_street = ce(logits_street, sid_true)
                loss_ward   = ce(logits_ward,   wid_true)
                loss = (CFG.W_PROVINCE * loss_prov +
                        CFG.W_STREET   * loss_street +
                        CFG.W_WARD     * loss_ward) / CFG.GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % CFG.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss   += loss.item() * CFG.GRAD_ACCUM
            total_street += loss_street.item()
            total_ward   += loss_ward.item()
            total_prov   += loss_prov.item()
            n_batches    += 1

            if step % 500 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  E{epoch+1} step {step:>5}/{len(train_loader)} | "
                      f"loss={total_loss/n_batches:.3f} "
                      f"st={total_street/n_batches:.3f} "
                      f"wd={total_ward/n_batches:.3f} "
                      f"lr={lr:.2e}", flush=True)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        n_correct_street = n_correct_ward = n_correct_full = n_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inp      = batch['input'].to(device)
                sid_true = batch['street_id'].to(device)
                wid_true = batch['ward_id'].to(device)
                pid_true = batch['prov_id'].to(device)

                logits_prov, logits_street, logits_ward = model(inp)

                # Apply province masks
                pid_pred = logits_prov.argmax(dim=1)
                # Mask streets: for each sample, zero out invalid street logits
                street_mask = s_masks[pid_pred]   # (B, n_street)
                ward_mask   = w_masks[pid_pred]   # (B, n_ward)
                logits_street = logits_street.masked_fill(~street_mask, -1e9)
                logits_ward   = logits_ward.masked_fill(~ward_mask,   -1e9)

                sid_pred = logits_street.argmax(dim=1)
                wid_pred = logits_ward.argmax(dim=1)

                n_correct_street += (sid_pred == sid_true).sum().item()
                n_correct_ward   += (wid_pred == wid_true).sum().item()
                n_correct_full   += ((sid_pred == sid_true) & (wid_pred == wid_true)).sum().item()
                n_total          += inp.shape[0]

        acc_street = n_correct_street / n_total
        acc_ward   = n_correct_ward   / n_total
        acc_full   = n_correct_full   / n_total
        elapsed    = time.time() - t0

        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS} [{elapsed:.0f}s]")
        print(f"  Street acc : {acc_street:.4f}")
        print(f"  Ward acc   : {acc_ward:.4f}")
        print(f"  Full acc   : {acc_full:.4f}  ← main metric")
        print()

        if acc_ward > best_ward_acc:
            best_ward_acc = acc_ward
            # Save checkpoint
            m = model.module if hasattr(model, 'module') else model
            torch.save(m.state_dict(), f"{CFG.OUTPUT_DIR}/model_best.pt")
            print(f"  ✓ Saved best model (ward_acc={acc_ward:.4f})")

    # ── Save everything ───────────────────────────────────────────────────────
    m = model.module if hasattr(model, 'module') else model
    torch.save(m.state_dict(), f"{CFG.OUTPUT_DIR}/model_final.pt")
    vocab.save(f"{CFG.OUTPUT_DIR}/vocab.json")
    street_enc.save(f"{CFG.OUTPUT_DIR}/street_classes.json")
    ward_enc.save(f"{CFG.OUTPUT_DIR}/ward_classes.json")
    prov_enc.save(f"{CFG.OUTPUT_DIR}/prov_classes.json")
    torch.save(s_masks.cpu(), f"{CFG.OUTPUT_DIR}/street_masks.pt")
    torch.save(w_masks.cpu(), f"{CFG.OUTPUT_DIR}/ward_masks.pt")

    # Save model config
    cfg_dict = {k: v for k, v in vars(CFG).items() if not k.startswith('_')}
    json.dump(cfg_dict, open(f"{CFG.OUTPUT_DIR}/config.json",'w'))

    print(f"\n✓ Training complete. Best ward acc: {best_ward_acc:.4f}")
    print(f"  Files saved to {CFG.OUTPUT_DIR}/")

    # ── Export ONNX ───────────────────────────────────────────────────────────
    print("\nExporting ONNX...")
    m.eval()
    dummy = torch.zeros(1, CFG.MAX_LEN, dtype=torch.long).to(device)
    torch.onnx.export(
        m, dummy,
        f"{CFG.OUTPUT_DIR}/model.onnx",
        input_names=['input'],
        output_names=['prov_logits','street_logits','ward_logits'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=14,
    )
    import os
    onnx_mb = os.path.getsize(f"{CFG.OUTPUT_DIR}/model.onnx") / 1024 / 1024
    print(f"  ONNX saved: {onnx_mb:.1f} MB")

if __name__ == '__main__':
    train()
