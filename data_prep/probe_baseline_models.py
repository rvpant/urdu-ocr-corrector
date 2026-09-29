"""Probe candidate first-pass OCR models on a handful of real documents.

INIT.md section 3.2 assumes Qwen3.5-9B for both roles. That assumption needs to
survive contact with Prime Inference: model availability drifts, and a model
that is "Available" for hosted training is not necessarily responsive on the
inference endpoint. This script measures CER and latency per (model, variant)
so the baseline-model choice is made on evidence.

Variants:
    low        low-res image as-is (the production input distribution)
    low_up{N}  low-res upscaled NxN before encoding -- adds no information, but
               gives the vision encoder more patches to work with
    high       high-res image (the achievability-ceiling reference)

Usage:
    uv run python data_prep/probe_baseline_models.py --docs 100 1 414 --models Qwen/Qwen3.5-4B
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import jiwer
from datasets import load_from_disk
from openai import AsyncOpenAI
from PIL import Image

DEFAULT_BASE_URL = "https://api.pinference.ai/api/v1"
PROMPT = (
    "Transcribe all text in this document image. "
    "Preserve line breaks. Output only the transcription."
)


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


def encode(path: str, upscale: int = 1) -> str:
    if upscale == 1:
        raw = Path(path).read_bytes()
    else:
        im = Image.open(path)
        im = im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        raw = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


async def transcribe(client, model, data_url, max_tokens, timeout):
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = resp.choices[0]
        return {
            "text": choice.message.content or "",
            "finish": choice.finish_reason,
            "secs": time.time() - t0,
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "finish": f"error:{type(exc).__name__}", "secs": time.time() - t0, "prompt_tokens": None}


async def run(args: argparse.Namespace) -> None:
    splits = load_from_disk(str(args.dataset))
    by_id = {ex["doc_id"]: ex for split in splits for ex in splits[split]}
    docs = args.docs or list(by_id)[: args.num_docs]

    variants: list[tuple[str, str, int]] = [("low", "low_res_path", 1)]
    for n in args.upscale:
        variants.append((f"low_up{n}", "low_res_path", n))
    if args.include_high:
        variants.append(("high", "high_res_path", 1))
        for n in args.high_upscale:
            variants.append((f"high_up{n}", "high_res_path", n))

    client = AsyncOpenAI(base_url=args.base_url, api_key=resolve_api_key())
    sem = asyncio.Semaphore(args.concurrency)

    async def one(model, doc_id, variant, field, up):
        ex = by_id[doc_id]
        data_url = await asyncio.to_thread(encode, ex[field], up)
        async with sem:
            res = await transcribe(client, model, data_url, args.max_tokens, args.timeout)
        res["cer"] = jiwer.cer(ex["ground_truth"], res["text"]) if res["text"] else None
        return {"model": model, "doc_id": doc_id, "variant": variant, **res}

    jobs = [
        one(m, d, v, f, u)
        for m in args.models
        for d in docs
        for (v, f, u) in variants
    ]
    results = await asyncio.gather(*jobs)
    await client.close()

    print(f"\n{'model':<34} {'variant':<10} {'n':>3} {'medCER':>7} {'medSec':>7} {'ptok':>6}  errors")
    agg: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        agg.setdefault((r["model"], r["variant"]), []).append(r)
    for (model, variant), rs in agg.items():
        cers = sorted(r["cer"] for r in rs if r["cer"] is not None)
        med = cers[len(cers) // 2] if cers else float("nan")
        secs = sorted(r["secs"] for r in rs)
        ptok = [r["prompt_tokens"] for r in rs if r["prompt_tokens"]]
        errs = sum(1 for r in rs if str(r["finish"]).startswith("error"))
        print(
            f"{model:<34} {variant:<10} {len(rs):>3} {med:>7.3f} "
            f"{secs[len(secs)//2]:>7.1f} {(sum(ptok)//len(ptok)) if ptok else 0:>6}  {errs}"
        )

    if args.dump:
        args.dump.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nfull results -> {args.dump}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--docs", nargs="*", default=None, help="explicit doc_ids")
    ap.add_argument("--num-docs", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3.5-4B"])
    ap.add_argument("--upscale", nargs="*", type=int, default=[2])
    ap.add_argument("--high-upscale", nargs="*", type=int, default=[])
    ap.add_argument("--include-high", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("PRIME_INFERENCE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--dump", type=Path, default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
