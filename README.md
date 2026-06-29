# Vietnamese Address Normalizer

Normalizes Vietnamese address strings to canonical post-2025 administrative form.

Vietnam's 2025 administrative reorganization merged thousands of wards and districts. This tool maps both old and new address variants to the current canonical form using a seq2seq Transformer with province-constrained beam search over 187K clean canonical addresses.

```
Input:  "p tan dinh q1 tphcm"
Output: "Phường Tân Định, Thành phố Hồ Chí Minh"

Input:  "P. Ben Nghe Q1 HCM"
Output: "Phường Sài Gòn, Thành phố Hồ Chí Minh"  ← pre-2025 rename handled

Input:  "Phuong 14 Quan 5 HCM"
Output: ""  ← numbered wards no longer exist post-2025
```

---

## Quickstart

**Step 1 — Get the model weights from HuggingFace:**

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('qox/vn-address-normalizer', local_dir='.')
"
```

Or manually download from: https://huggingface.co/qox/vn-address-normalizer

**Step 2 — Install dependencies:**

```bash
pip install -r requirements.txt
```

**Step 3 — Run inference:**

```bash
python3 inference.py
```

Or as a Python module:

```python
from inference import normalize

result = normalize("p tan dinh q1 tphcm")
print(result["canonical"])   # "Phường Tân Định, Thành phố Hồ Chí Minh"
print(result["valid"])       # True
print(result["latency_ms"])  # ~150 ms warm
```

---

## Model

**Architecture:** Seq2Seq Transformer, 26M parameters
- Encoder: 4-layer, 256-dim, 4-head attention, GELU, Pre-LN
- Decoder: 3-layer, same config
- Tokenization: character-level (287 source / 269 target vocab)

**Inference pipeline:**
1. Detect province from raw text (regex + alias table)
2. Detect ward hint (regex + ward slug index + 13K legacy pre-2025 mappings)
3. Build constrained trie: ward → province → full 187K candidates
4. Province-constrained beam search (beam=5, max 96 steps)
5. Output is guaranteed to exist in the canonical address database

**Coverage:** 79.3% of Vietnamese address variants across 34 provinces, 3,321 wards

**Latency:** ~10–150 ms warm, ~2–4 s cold start (model load + trie build)

---

## Return fields

```python
result = normalize("xa cu chi tp hcm")
```

| Field | Type | Description |
|---|---|---|
| `canonical` | str | Normalized address; empty string if not found |
| `valid` | bool | True if canonical exists in address database |
| `confidence` | float | Log-prob score (higher = more confident) |
| `province` | str | Resolved province name, or None |
| `ward_hint` | str | Detected ward slug, or None |
| `search_space` | int | Trie candidates searched |
| `latency_ms` | float | Wall-clock inference time |

---

## REST API

```bash
pip install fastapi uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"address": "p tan dinh q1 tphcm"}'
```

---

## Training

Training data: 187K canonical addresses, character-level pairs generated from canonical → variant augmentation.

```bash
python3 train_vn_address.py
```

Trained on Kaggle (single T4 GPU, ~12 epochs). Model weights are published at [qox/vn-address-normalizer](https://huggingface.co/qox/vn-address-normalizer).

---

## Tests

```bash
python3 test_regression.py
# 8/8 cases: ward detection, legacy name mapping, numbered ward rejection, street+ward+province
```

---

## Repository Structure

```
.
├── inference.py          # Standalone inference — only file needed to run
├── normalizer.py         # Full normalizer: FST + rule-based + ML fallback
├── normalize_ml.py       # ML inference wrapper (used by api.py)
├── fst.py                # FST-based exact matcher
├── api.py                # FastAPI REST server
├── train_vn_address.py   # Training pipeline
├── benchmark.py          # Latency and accuracy benchmarks
├── test_regression.py    # Regression test suite (8 cases)
├── requirements.txt
├── model_v3_final/
│   ├── config.json           # Model hyperparameters
│   ├── src_vocab.json        # Source character vocabulary
│   ├── tgt_vocab.json        # Target character vocabulary
│   ├── clean_canonicals.json # 187K canonical addresses
│   └── legacy_ward_idx.json  # Pre-2025 → post-2025 ward mappings (13K entries)
├── street_ward_map.json      # Street → ward index
└── scripts/
    ├── phase1_generate.py    # Generate training pairs from canonical addresses
    ├── phase0_spatial.py     # OSM spatial join for street data
    ├── phase0_spatial_join.py
    ├── nominatim_enrich.py   # Nominatim geocoding enrichment
    └── google_geocode.py     # Google Maps geocoding enrichment
```

**Model weights** (`model.safetensors`, 26MB) are hosted on HuggingFace — not included here due to size.

---

## Limitations

- **Post-2025 boundaries only.** Pre-2025 ward names (e.g. "Bến Nghé" → "Sài Gòn") are handled via the legacy map, but coverage is not exhaustive.
- **Numbered wards rejected.** "Phường 14", "Quận 10" no longer exist post-2025.
- **CPU inference only.** GPU not required or used.
- **Cold start:** ~2–4 s (model load + trie build). Subsequent calls: ~10–150 ms.

---

## License

MIT
