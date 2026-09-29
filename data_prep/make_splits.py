"""Build the three-way disjoint split for the SFT -> RL experiment.

Training SFT and then RL on the same documents is not a valid experiment: the
SFT stage memorises its examples, so RL has no headroom left on them and any
measured gain is recall rather than generalisation. This partitions the corpus
so each stage sees documents the previous stage never did.

    sft_train   479   stage 1 supervision
    rl_train    250   stage 2 -- never seen during SFT
    test        100   never trained on by anything

The split is deterministic given --seed, and is pushed as three named splits of
one HF dataset so the Colab notebook can load it by name with no file juggling.

Usage:
    uv run python data_prep/make_splits.py --push-to-hub rpant/ocr-urdu-splits
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, concatenate_datasets, load_from_disk

FEATURES = Features(
    {
        "doc_id": Value("string"),
        "image": Image(),
        "image_source": Value("string"),
        "ground_truth": Value("string"),
        "language": Value("string"),
        "baseline_low": Value("string"),
        "baseline_high": Value("string"),
        "baseline_model": Value("string"),
        "low_upscale": Value("int32"),
        "high_upscale": Value("int32"),
        "split_role": Value("string"),
    }
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("packaged_dataset"),
                    help="output of data_prep/package_dataset.py (images embedded)")
    ap.add_argument("--out", type=Path, default=Path("splits_dataset"))
    ap.add_argument("--push-to-hub", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--n-rl", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dd = load_from_disk(str(args.dataset))
    # The packaged dataset carries the old 746/83 train/test cut, which is
    # unrelated to the three-way partition we need here. Flatten and re-split.
    full = concatenate_datasets([dd[k] for k in sorted(dd)])
    full = full.shuffle(seed=args.seed)
    n = len(full)
    n_test, n_rl = args.n_test, args.n_rl
    n_sft = n - n_test - n_rl
    if n_sft <= 0:
        raise SystemExit(f"corpus of {n} too small for rl={n_rl} + test={n_test}")

    pieces = {
        "test": full.select(range(n_test)),
        "rl_train": full.select(range(n_test, n_test + n_rl)),
        "sft_train": full.select(range(n_test + n_rl, n)),
    }

    out = {}
    for name, ds in pieces.items():
        ds = ds.add_column("split_role", [name] * len(ds))
        out[name] = ds.cast(FEATURES)
    dsd = DatasetDict(out)

    # Disjointness is the entire point of this script, so assert it rather than
    # trusting the arithmetic above.
    ids = {k: set(v["doc_id"]) for k, v in dsd.items()}
    for a in ids:
        for b in ids:
            if a < b and ids[a] & ids[b]:
                raise SystemExit(f"OVERLAP between {a} and {b}: {sorted(ids[a] & ids[b])[:5]}")
    assert sum(len(v) for v in ids.values()) == n, "documents lost during split"

    dsd.save_to_disk(str(args.out))
    print(f"wrote {args.out}")
    for k, v in dsd.items():
        print(f"  {k:<10} {len(v):>4}")
    print(f"  {'TOTAL':<10} {n:>4}  (disjointness verified)")

    manifest = {
        "seed": args.seed,
        "counts": {k: len(v) for k, v in dsd.items()},
        "doc_ids": {k: sorted(ids[k], key=lambda x: int(x) if x.isdigit() else 0) for k in ids},
    }
    # Inside --out, not beside it: a sibling path is shared by every invocation,
    # so packaging a second variant silently overwrites the first one's manifest.
    mpath = args.out / "splits_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {mpath}")

    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub, private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
