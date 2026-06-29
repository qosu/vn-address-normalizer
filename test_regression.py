"""
Phase 0 — Regression Test Suite
Run: python3 test_regression.py
Must pass before every deploy.
"""
import sys, json, time
sys.path.insert(0, "/root/vn_address")
from normalize_ml import normalize, warm_up

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_canonical(c):
    """Extract (street, ward, province) from canonical string."""
    if not c: return None, None, None
    parts = [p.strip() for p in c.split(",")]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return None, parts[0], parts[1]
    return None, None, None

# ── Test cases ────────────────────────────────────────────────────────────────
TESTS = [
    {
        "id": 1,
        "input": "45 tran binh trong p tan dinh q1 hcm",
        "expect_province": "Thành phố Hồ Chí Minh",
        "expect_ward":     "Phường Tân Định",
        "expect_street":   "Trần Bình Trọng",   # exact match preferred
        "allow_partial":   True,                 # street may be missing/wrong
        "expect_valid":    True,
        "note": "street+ward+province, legacy district Q1"
    },
    {
        "id": 2,
        "input": "phuong ba dinh ha noi",
        "expect_province": "Thành phố Hà Nội",
        "expect_ward":     "Phường Ba Đình",
        "expect_street":   None,
        "allow_partial":   False,
        "expect_valid":    True,
        "note": "ward-only, no accent"
    },
    {
        "id": 3,
        "input": "XA CU CHI TP HO CHI MINH",
        "expect_province": "Thành phố Hồ Chí Minh",
        "expect_ward":     "Xã Củ Chi",
        "expect_street":   None,
        "allow_partial":   False,
        "expect_valid":    True,
        "note": "ward-only uppercase"
    },
    {
        "id": 4,
        "input": "P. Ben Nghe Q1 HCM",
        "expect_province": "Thành phố Hồ Chí Minh",
        "expect_ward":     "Phường Sài Gòn",    # legacy: Bến Nghé → Sài Gòn
        "expect_street":   None,
        "allow_partial":   False,
        "expect_valid":    True,
        "note": "legacy ward mapping"
    },
    {
        "id": 5,
        "input": "Xa thuong lam tinh Ha Giang",
        "expect_province": "Tỉnh Tuyên Quang",  # Ha Giang → Tuyen Quang post-2025
        "expect_ward":     "Xã Thượng Lâm",
        "expect_street":   None,
        "allow_partial":   False,
        "expect_valid":    True,
        "note": "province merge: Hà Giang → Tuyên Quang"
    },
    {
        "id": 6,
        "input": "Phuong 14 Quan 5 HCM",
        "expect_province": None,
        "expect_ward":     None,
        "expect_street":   None,
        "allow_partial":   False,
        "expect_valid":    False,                # numbered ward = invalid
        "note": "numbered ward must be rejected"
    },
    {
        "id": 7,
        "input": "nguyen trai phuong thuong dinh thanh xuan ha noi",
        "expect_province": "Thành phố Hà Nội",
        "expect_ward":     "Phường Thanh Xuân",  # Thượng Đình merged → Thanh Xuân post-2025
        "expect_street":   "Nguyễn Trãi",
        "allow_partial":   True,
        "expect_valid":    True,
        "note": "street+ward, ward merge post-2025"
    },
    {
        "id": 8,
        "input": "duong le loi phuong ben nghe quan 1 tphcm",
        "expect_province": "Thành phố Hồ Chí Minh",
        "expect_ward":     "Phường Sài Gòn",    # legacy: Bến Nghé → Sài Gòn
        "expect_street":   None,                 # "Lê Lợi" preferred but not required yet
        "allow_partial":   True,
        "expect_valid":    True,
        "note": "street+legacy ward — ward must be correct, street is best-effort"
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────
def run_tests(verbose=True):
    passed = failed = 0
    failures = []

    for tc in TESTS:
        t0 = time.perf_counter()
        r  = normalize(tc["input"])
        ms = (time.perf_counter() - t0) * 1000

        street, ward, province = parse_canonical(r.canonical)

        ok = True
        reasons = []

        # valid check
        if tc["expect_valid"] and not r.valid:
            ok = False; reasons.append(f"expect valid=True got valid=False")
        if not tc["expect_valid"] and r.valid and r.canonical:
            ok = False; reasons.append(f"expect invalid but got {r.canonical!r}")

        # province check (strict)
        if tc["expect_province"] and province != tc["expect_province"]:
            ok = False; reasons.append(
                f"province: expected {tc['expect_province']!r} got {province!r}")

        # ward check (strict)
        if tc["expect_ward"] and ward != tc["expect_ward"]:
            ok = False; reasons.append(
                f"ward: expected {tc['expect_ward']!r} got {ward!r}")

        # street check (only if not allow_partial)
        if tc["expect_street"] and not tc["allow_partial"]:
            if street != tc["expect_street"]:
                ok = False; reasons.append(
                    f"street: expected {tc['expect_street']!r} got {street!r}")

        sym = "✓" if ok else "✗"
        if ok: passed += 1
        else:
            failed += 1
            failures.append({"id": tc["id"], "input": tc["input"],
                             "reasons": reasons, "got": r.canonical})

        if verbose:
            print(f"  {sym} [{tc['id']}] {tc['input'][:45]:<45} "
                  f"→ {(r.canonical or '(empty)')[:45]:<45} "
                  f"[{r.stage}] {ms:.1f}ms")
            for reason in reasons:
                print(f"       ↳ FAIL: {reason}")

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{len(TESTS)} passed")
    if failures:
        print(f"\nFailed cases:")
        for f in failures:
            print(f"  [{f['id']}] {f['input']!r}")
            print(f"       got: {f['got']!r}")
            for r in f['reasons']: print(f"       → {r}")
    return passed, failed

if __name__ == "__main__":
    warm_up()
    print("VN Address Normalizer — Regression Tests")
    print("="*60)
    passed, failed = run_tests(verbose=True)
    sys.exit(0 if failed == 0 else 1)
