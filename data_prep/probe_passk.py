"""Does the corrector EVER beat its baseline? The decisive pre-training check.

GRPO can only amplify behaviour the policy already samples. If no rollout in a
group beats the frozen baseline, every advantage within that group is negative
or flat, and the gradient points at whichever rollout edited least -- i.e. at
inaction, not at capability. No amount of training installs a skill that is
absent from the sample space.

The smoke eval saw 0/8 rollouts beat baseline, but at only 2 rollouts per
document it cannot see the tail. This samples at the real training width.

Reads:
    best-of-k > baseline on a decent share of documents  -> signal in the tail,
        GRPO has something to amplify; proceed with RL.
    best-of-k worse than baseline nearly everywhere      -> the capability is
        not in the sample space; distillation, not RL.

Usage:
    uv run python data_prep/probe_passk.py -k 16 -n 10
"""

import argparse
import asyncio
import base64
import io
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

from datasets import load_from_disk
from openai import AsyncOpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments" / "ocr_corrector"))
from ocr_corrector import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    cer,
    normalize,
)

DEFAULT_BASE_URL = "https://api.pinference.ai/api/v1"
_CORRECTED = re.compile(r"<corrected>(.*?)</corrected>", re.DOTALL)


def resolve_api_key() -> str:
    key = os.environ.get("PRIME_API_KEY")
    if key:
        return key
    cfg = Path.home() / ".prime" / "config.json"
    if cfg.exists():
        key = json.loads(cfg.read_text()).get("api_key")
        if key:
            return key
    raise SystemExit("no API key: set PRIME_API_KEY or run `prime login`")


def encode(path: str, upscale: int) -> str:
    with Image.open(path) as im:
        if upscale > 1:
            im = im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# Direct-OCR prompt: no draft, no reasoning block, no tags. Deliberately the
# same shape that generated baselines.jsonl without a single failure in 1,658
# calls -- the <reasoning> block is what sends this model into `Wait, ...` loops,
# and 34% of correction rollouts died that way.
DIRECT_PROMPT = (
    "Transcribe all text in this document image. "
    "Preserve line breaks. Output only the transcription."
)


def _build_messages(data_url: str, baseline: str, direct: bool) -> list[dict]:
    if direct:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DIRECT_PROMPT},
                ],
            }
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": USER_TEMPLATE.format(baseline=baseline)},
            ],
        },
    ]


async def one_rollout(client, model, data_url, baseline, args) -> str | None:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=_build_messages(data_url, baseline, args.direct),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return None
    if args.direct:
        return text.strip() or None
    m = _CORRECTED.search(text)
    return m.group(1).strip() if m else None


async def main_async(args) -> None:
    splits = load_from_disk(str(args.dataset))
    baselines = {}
    with open(args.baselines, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                baselines[r["doc_id"]] = r

    docs = [ex for ex in splits[args.split] if ex["doc_id"] in baselines][: args.num_docs]
    client = AsyncOpenAI(base_url=args.base_url, api_key=resolve_api_key())
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(*a):
        async with sem:
            return await one_rollout(*a)

    print(f"model={args.model} docs={len(docs)} k={args.k} temp={args.temperature}\n")
    rows = []
    for ex in docs:
        b = baselines[ex["doc_id"]]["baseline_low"]
        gt = normalize(ex["ground_truth"])
        base_cer = cer(gt, normalize(b))
        data_url = await asyncio.to_thread(encode, ex["low_res_path"], args.upscale)
        outs = await asyncio.gather(*(guarded(client, args.model, data_url, b, args) for _ in range(args.k)))
        cers = [cer(gt, normalize(o)) if o is not None else None for o in outs]
        valid = [c for c in cers if c is not None]
        n_better = sum(1 for c in valid if c < base_cer)
        best = min(valid) if valid else float("nan")
        rows.append(
            dict(doc=ex["doc_id"], base=base_cer, best=best,
                 mean=st.mean(valid) if valid else float("nan"),
                 n_better=n_better, n_valid=len(valid))
        )
        print(f"doc {ex['doc_id']:>4}  base={base_cer:.3f}  best_of_{args.k}={best:.3f}  "
              f"mean={rows[-1]['mean']:.3f}  better={n_better}/{len(valid)}  "
              f"{'IMPROVED' if best < base_cer else ''}")

    await client.close()

    n = len(rows)
    docs_improved = sum(1 for r in rows if r["best"] < r["base"])
    total_roll = sum(r["n_valid"] for r in rows)
    total_better = sum(r["n_better"] for r in rows)
    print("\n" + "=" * 64)
    print(f"documents where best-of-{args.k} beat baseline : {docs_improved}/{n} = {docs_improved/n:.1%}")
    print(f"individual rollouts that beat baseline        : {total_better}/{total_roll} = "
          f"{total_better/max(1,total_roll):.1%}")
    print(f"mean baseline CER                             : {st.mean(r['base'] for r in rows):.3f}")
    print(f"mean best-of-{args.k} CER                          : "
          f"{st.mean(r['best'] for r in rows if r['best'] == r['best']):.3f}")
    print(f"mean-of-means CER                             : "
          f"{st.mean(r['mean'] for r in rows if r['mean'] == r['mean']):.3f}")
    print("=" * 64)
    print("\nREAD: a high best-of-k improvement rate with a poor mean means the")
    print("capability is present but unreliable -- exactly what GRPO amplifies.")
    print("Near-zero best-of-k improvement means it is absent; prefer distillation.")

    if args.dump:
        Path(args.dump).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nrows -> {args.dump}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--baselines", type=Path, default=Path("baselines.jsonl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("-k", type=int, default=16, help="rollouts per document")
    ap.add_argument("-n", "--num-docs", type=int, default=10)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--base-url", default=os.environ.get("PRIME_INFERENCE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--direct", action="store_true",
                    help="direct OCR (no draft in the prompt) instead of correction")
    ap.add_argument("--dump", type=Path, default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
