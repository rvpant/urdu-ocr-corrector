"""Package a self-contained dataset for hosted training.

The local dataset stores filesystem paths, which is fine while everything runs
on this machine. Hosted training runs the environment in a remote container
that has no access to this workspace, so the images and the cached baselines
have to travel with the dataset.

This merges `hf_dataset/` + `baselines.jsonl` into one dataset with low-res
image bytes embedded, saves it locally, and optionally pushes it to the HF Hub.
`load_environment(dataset_id=...)` reads exactly this schema.

`--image-source` picks which capture is embedded. It matters more than it looks.

For the *correction* task (INIT.md 3.3) low-res is the only defensible choice:
the job is to repair a low-res first pass, and shipping the high-res twin would
leak the answer. For *direct* OCR that reasoning does not apply, and low-res is
simply a worse input. Measured over all 829 documents with the same
Qwen3.5-4B and the same prompt:

    low-res,  4x upscale   CER 0.332   sub 0.155   ins 0.240
    high-res, 2x upscale   CER 0.168   sub 0.068   ins 0.117

High-res wins on 96.1% of documents and halves the substitution rate, so the
model is genuinely reading better rather than just writing less. Every run in
`RL_report_v1.md` used low-res, which was correct for the task it was written
for and wrong for the task we ended up training. Default stays `low` so the
older artifacts remain reproducible; pass `--image-source high` for new work.

Usage:
    uv run python data_prep/package_dataset.py --out packaged_dataset
    uv run python data_prep/package_dataset.py --image-source high \
        --out packaged_dataset_hr --push-to-hub <user>/ocr-corrector-ur-hr --private
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from datasets import load_from_disk

FEATURES = Features(
    {
        "doc_id": Value("string"),
        "image": Image(),
        # Which capture `image` holds. Consumers that care about resolution must
        # read this rather than inferring it from pixel dimensions, since the
        # two sources overlap in size (low-res runs 121-363 px wide, high-res
        # 484-1452, but upscaling either one destroys the distinction).
        "image_source": Value("string"),
        "ground_truth": Value("string"),
        "language": Value("string"),
        "baseline_low": Value("string"),
        "baseline_high": Value("string"),
        "baseline_model": Value("string"),
        # Carried so the environment can refuse to run when its image_upscale
        # disagrees with how the cached baseline was transcribed.
        "low_upscale": Value("int32"),
        "high_upscale": Value("int32"),
    }
)


def load_baselines(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["doc_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--baselines", type=Path, default=Path("baselines.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("packaged_dataset"))
    ap.add_argument("--push-to-hub", default=None, help="HF Hub dataset id")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--image-source", choices=("low", "high"), default="low",
                    help="which capture to embed; see module docstring")
    args = ap.parse_args()
    path_key = f"{args.image_source}_res_path"

    splits = load_from_disk(str(args.dataset))
    baselines = load_baselines(args.baselines)

    packaged = {}
    for split_name in splits:
        rows, dropped = [], 0
        for ex in splits[split_name]:
            b = baselines.get(ex["doc_id"])
            if b is None:
                dropped += 1
                continue
            rows.append(
                {
                    "doc_id": ex["doc_id"],
                    "image": ex[path_key],
                    "image_source": args.image_source,
                    "ground_truth": ex["ground_truth"],
                    "language": ex["language"],
                    "baseline_low": b.get("baseline_low", ""),
                    "baseline_high": b.get("baseline_high", ""),
                    "baseline_model": b.get("model", "unknown"),
                    "low_upscale": int(b.get("low_upscale", 1)),
                    "high_upscale": int(b.get("high_upscale", 1)),
                }
            )
        if dropped:
            print(f"{split_name}: dropped {dropped} docs with no cached baseline")
        packaged[split_name] = Dataset.from_list(rows, features=FEATURES)

    dd = DatasetDict(packaged)
    dd.save_to_disk(str(args.out))
    print(f"wrote {args.out}: " + " ".join(f"{k}={len(v)}" for k, v in dd.items()))

    if args.push_to_hub:
        dd.push_to_hub(args.push_to_hub, private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.push_to_hub}")
        print(f"use it with:  -a '{{\"dataset_id\": \"{args.push_to_hub}\"}}'")


if __name__ == "__main__":
    main()
