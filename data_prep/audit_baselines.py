"""Audit the cached baselines before spending money on training.

Prints the CER distribution for the low-res and high-res first passes, flags
documents where the high-res pass disagrees badly with the provided ground
truth (a data-quality signal, not a training signal), and reports the fraction
of documents sitting in the CER band where RL has usable signal.

Normalization and the CER function are imported from the environment module on
purpose. INIT.md section 8 lists "normalization mismatch between reward
computation and eval metric" as a reward-function bug that masquerades as
learning; sharing one implementation makes that class of bug impossible.

Usage:
    uv run python data_prep/audit_baselines.py
    uv run python data_prep/audit_baselines.py --split test
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments" / "ocr_corrector"))
from ocr_corrector import CER_CAP, cer, normalize  # noqa: E402

# Signal band from INIT.md 6.4: below this the baseline is already good enough
# that corrections cannot help; above it the image is too degraded to recover.
BAND_LOW, BAND_HIGH = 0.05, 0.30
# High-res first pass this far from the ground truth suggests the label, not the OCR.
GT_SUSPECT_CER = 0.30


def summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name:<26} n=   0")
        return
    a = sorted(values)
    n = len(a)

    def pct(p: float) -> float:
        return a[min(n - 1, int(n * p))]

    print(
        f"{name:<26} n={n:>4} mean={sum(a)/n:.3f} p25={pct(0.25):.3f} "
        f"median={pct(0.50):.3f} p75={pct(0.75):.3f} p90={pct(0.90):.3f}"
    )


def repetition_score(text: str, window: int = 40) -> float:
    """Fraction of the text covered by a repeated window. Detects the runaway
    loops small VLMs fall into on illegible input."""
    t = normalize(text)
    if len(t) < 2 * window:
        return 0.0
    chunks = [t[i : i + window] for i in range(0, len(t) - window, window)]
    if not chunks:
        return 0.0
    most_common = Counter(chunks).most_common(1)[0][1]
    return most_common / len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--baselines", type=Path, default=Path("baselines.jsonl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--show-worst", type=int, default=0, help="print N worst-CER doc_ids")
    args = ap.parse_args()

    splits = load_from_disk(str(args.dataset))
    meta = {ex["doc_id"]: ex for ex in splits[args.split]}

    low_cers: list[float] = []
    high_cers: list[float] = []
    by_lang: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"low": [], "high": []})
    finish = Counter()
    repetitive_low = repetitive_high = 0
    len_ratios: list[float] = []
    per_doc: list[tuple[float, str]] = []
    missing = 0

    with args.baselines.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ex = meta.get(r["doc_id"])
            if ex is None or r.get("split") != args.split:
                continue
            gt = normalize(ex["ground_truth"])
            if not gt:
                missing += 1
                continue
            lc = cer(gt, normalize(r["baseline_low"]))
            hc = cer(gt, normalize(r["baseline_high"]))
            low_cers.append(lc)
            high_cers.append(hc)
            lang = ex["language"]
            by_lang[lang]["low"].append(lc)
            by_lang[lang]["high"].append(hc)
            finish[r.get("finish_low", "?")] += 1
            finish[r.get("finish_high", "?")] += 1
            if repetition_score(r["baseline_low"]) > 0.3:
                repetitive_low += 1
            if repetition_score(r["baseline_high"]) > 0.3:
                repetitive_high += 1
            if gt:
                len_ratios.append(len(normalize(r["baseline_low"])) / len(gt))
            per_doc.append((lc, r["doc_id"]))

    n = len(low_cers)
    if not n:
        raise SystemExit(f"no cached baselines matched split={args.split}")

    print(f"=== baseline CER vs ground truth (split={args.split}, capped at {CER_CAP}) ===")
    summarize("low-res  overall", low_cers)
    summarize("high-res overall", high_cers)
    for lang, vals in sorted(by_lang.items()):
        summarize(f"low-res  {lang}", vals["low"])
        summarize(f"high-res {lang}", vals["high"])

    print("\n=== first-pass health ===")
    print(f"finish reasons: {dict(finish)}")
    print(f"repetition-looped low-res:  {repetitive_low}/{n} = {repetitive_low/n:.1%}")
    print(f"repetition-looped high-res: {repetitive_high}/{n} = {repetitive_high/n:.1%}")
    summarize("low/GT length ratio", len_ratios)
    if missing:
        print(f"skipped {missing} docs with empty ground truth")

    print("\n=== go / no-go ===")
    in_band = sum(1 for c in low_cers if BAND_LOW <= c <= BAND_HIGH)
    print(f"low-res baselines in [{BAND_LOW}, {BAND_HIGH}] band: {in_band}/{n} = {in_band/n:.1%}")
    print("  (INIT.md 6.4: under 30% means weak RL signal -- restratify or rethink)")

    flagged = sum(1 for c in high_cers if c > GT_SUSPECT_CER)
    print(f"high-res baseline CER > {GT_SUSPECT_CER} (GT may be off): {flagged}/{n} = {flagged/n:.1%}")
    print("  (INIT.md 6.4: over 5% means audit the ground truth before spending compute)")

    headroom = [lc - hc for lc, hc in zip(low_cers, high_cers)]
    summarize("headroom (low - high)", headroom)
    no_headroom = sum(1 for h in headroom if h <= 0.01)
    print(f"docs with no headroom (<=0.01): {no_headroom}/{n} = {no_headroom/n:.1%}")

    if args.show_worst:
        print(f"\nworst {args.show_worst} low-res docs: "
              f"{[d for _, d in sorted(per_doc, reverse=True)[: args.show_worst]]}")


if __name__ == "__main__":
    main()
