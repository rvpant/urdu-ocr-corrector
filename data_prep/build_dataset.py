"""Convert the raw data/ folders into a HuggingFace Dataset on disk.

Actual layout in this repo (differs from the sketch in INIT.md section 4):

    data/test/lr-images/{i}.png      low-res capture
    data/test/hr-images/{i}.png      high-res capture of the same document
    data/test/groundtruth/{i}.txt    verified transcription, UTF-8

`{i}` is a bare integer (1, 2, ... 829), not zero-padded. The folder is named
"test" because it is the test split of the upstream corpus; for our purposes it
is the whole corpus, and we cut our own train/test split out of it.

Image bytes are not embedded here — we store paths and encode at prompt-build
time. See package_dataset.py for the portable (image-embedded) variant needed
for hosted training.

Usage:
    uv run python data_prep/build_dataset.py
    uv run python data_prep/build_dataset.py --source data/test --out hf_dataset
"""

import argparse
from pathlib import Path

from datasets import Dataset, Features, Value

# Urdu's letter inventory is a superset of Persian's: Urdu uses پ چ ژ گ too, so
# those are not discriminating (INIT.md's heuristic counted them as Persian
# evidence and mislabelled ~6% of this corpus). The only one-directional
# evidence is the retroflex/aspirate set plus ں and ے, which Persian never uses.
# Their *absence* is weak evidence at best, so we report "unknown" rather than
# claiming Persian.
URDU_ONLY = "ٹڈڑںےھ"


def infer_language(text: str) -> str:
    return "ur" if any(c in URDU_ONLY for c in text) else "unknown"


FEATURES = Features(
    {
        "doc_id": Value("string"),
        "low_res_path": Value("string"),
        "high_res_path": Value("string"),
        "ground_truth": Value("string"),
        "language": Value("string"),
    }
)


def _sort_key(stem: str):
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def build_rows(source: Path) -> list[dict]:
    gt_dir = source / "groundtruth"
    low_dir = source / "lr-images"
    high_dir = source / "hr-images"
    for d in (gt_dir, low_dir, high_dir):
        if not d.is_dir():
            raise SystemExit(f"missing expected directory: {d}")

    rows, skipped = [], []
    for gt_path in sorted(gt_dir.glob("*.txt"), key=lambda p: _sort_key(p.stem)):
        idx = gt_path.stem
        low = low_dir / f"{idx}.png"
        high = high_dir / f"{idx}.png"
        if not (low.exists() and high.exists()):
            skipped.append(idx)
            continue
        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            skipped.append(idx)
            continue
        rows.append(
            {
                "doc_id": idx,
                "low_res_path": str(low),
                "high_res_path": str(high),
                "ground_truth": text,
                "language": infer_language(text),
            }
        )
    if skipped:
        print(f"skipped {len(skipped)} docs (missing image or empty GT): {skipped[:10]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/test"))
    ap.add_argument("--out", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build_rows(args.source)
    if not rows:
        raise SystemExit(f"no usable documents found under {args.source}")

    ds = Dataset.from_list(rows, features=FEATURES)
    splits = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    splits.save_to_disk(str(args.out))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["language"]] = counts.get(r["language"], 0) + 1
    print(f"wrote {args.out}: train={len(splits['train'])} test={len(splits['test'])}")
    print(f"language mix: {counts}")


if __name__ == "__main__":
    main()
