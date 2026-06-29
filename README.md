---
language:
- vi
tags:
- address-normalization
- vietnamese
- seq2seq
- named-entity-recognition
license: mit
pipeline_tag: text2text-generation
---

# Vietnamese Address Normalizer

Normalizes Vietnamese address strings to canonical post-2025 administrative form.
Uses a Transformer seq2seq model with province-constrained beam search over
187K clean canonical addresses.

## Quick Start

```bash
git clone https://huggingface.co/<your-username>/vn-address-normalizer
cd vn-address-normalizer
pip install -r requirements.txt
python inference.py "p tan dinh q1 tphcm"
```

Expected output:

```
Input:       p tan dinh q1 tphcm
Canonical:   Phường Tân Định, Thành phố Hồ Chí Minh
Valid:       True
Province:    Thành phố Hồ Chí Minh
Ward hint:   tan dinh
Space:       170 candidates
Latency:     ~150 ms (warm), ~1.5 s (cold — model load + trie build)
```

## Python API

```python
from inference import normalize

# With or without Vietnamese diacritics
result = normalize("p tan dinh q1 tphcm")
result = normalize("Phường Tân Định, Quận 1, TP.HCM")
result = normalize("Xa Cu Chi TP HCM")

print(result["canonical"])  # "Phường Tân Định, Thành phố Hồ Chí Minh"
print(result["valid"])      # True
print(result["latency_ms"]) # ~150 ms warm
```

### Return fields

| Field | Type | Description |
|---|---|---|
| `canonical` | str | Normalized address; empty if not found |
| `valid` | bool | True if canonical exists in address database |
| `confidence` | float | Log-prob score (higher = more confident) |
| `province` | str | Resolved province name, or None |
| `ward_hint` | str | Detected ward slug, or None |
| `search_space` | int | Number of trie candidates searched |
| `latency_ms` | float | Wall-clock inference time in milliseconds |

## Input / Output Examples

| Input | Output |
|---|---|
| `p tan dinh q1 tphcm` | `Phường Tân Định, Thành phố Hồ Chí Minh` |
| `Phuong Ba Dinh Ha Noi` | `Phường Ba Đình, Thành phố Hà Nội` |
| `Xa Cu Chi TP HCM` | `Xã Củ Chi, Thành phố Hồ Chí Minh` |
| `P. Bến Nghé Q.1 HCM` | `Phường Sài Gòn, Thành phố Hồ Chí Minh` *(pre-2025 rename)* |
| `phuong 14 quan 10 tphcm` | *(empty — numbered wards removed post-2025)* |
| `duong le loi phuong ben nghe q1 tphcm` | `Đường Lê Lợi, Phường Sài Gòn, Thành phố Hồ Chí Minh` |

## Architecture

**Seq2Seq Transformer** (26M parameters):
- Encoder: 4-layer, 256-dim, 4-head, GELU, Pre-LN
- Decoder: 3-layer, same config
- Input: character-level tokenization (287-token vocab)
- Output: character-level decoding (269-token vocab)

**Inference pipeline:**
1. Detect province from raw text (regex + alias table)
2. Detect ward hint (regex + ward slug index + legacy ward map for pre-2025 names)
3. Build constrained trie: ward candidates (~10–500) → province candidates (~3K–52K) → full (~187K)
4. Province-constrained beam search (beam=5, max 96 steps)
5. Result is guaranteed in the canonical address database (trie acceptance)

**Training data:**
- 187K clean canonical addresses (34 provinces, 3,321 wards, post-2025 boundaries)
- Coverage: 79.3% of Vietnamese address variants

## Files

| File | Description |
|---|---|
| `inference.py` | Standalone inference — the only file you need to run |
| `model_v3_final/model.safetensors` | Model weights (26M params) |
| `model_v3_final/config.json` | Model hyperparameters |
| `model_v3_final/src_vocab.json` | Source character vocabulary |
| `model_v3_final/tgt_vocab.json` | Target character vocabulary |
| `model_v3_final/clean_canonicals.json` | 187K pre-filtered canonical addresses |
| `model_v3_final/legacy_ward_idx.json` | Pre-2025 → 2025 ward name mapping (13K entries) |

## Limitations

- **ML-only mode.** `inference.py` uses the neural model without the rule-based FST
  engine (which requires the `vietnam_provinces` package and a heavier index build).
  For production use with higher accuracy, integrate `normalizer.py` + `fst.py` from
  the full server-side stack.

- **Post-2025 boundaries only.** Canonical addresses reflect the 2025 administrative
  reorganization. Pre-2025 ward names (e.g. "Bến Nghé" → "Sài Gòn") are handled via
  the legacy map, but coverage is not exhaustive.

- **Numbered wards rejected.** Wards identified only by number (e.g. "Phường 14",
  "Quận 10") are rejected — these no longer exist post-2025.

- **2-variable input.** Model handles up to 3 address components (street, ward,
  province). Highly complex or apartment-style addresses may not simplify cleanly.

- **Cold start latency.** First call takes ~2–4 s (model load + trie construction).
  Subsequent calls: ~10–150 ms depending on search space.

- **CPU inference only.** GPU not required or used.

## Requirements

```
torch>=1.13.0
unidecode>=1.3.0
safetensors>=0.3.0
```

Python 3.10+ required (uses structural pattern matching in pipeline internals).
