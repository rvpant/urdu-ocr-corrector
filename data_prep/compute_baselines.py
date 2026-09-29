"""Compute frozen first-pass OCR baselines for both low-res and high-res images.

Low-res baselines are the actual corrector input at rollout time; high-res
baselines are the achievability-ceiling reference (INIT.md section 3.3).

Runs once and caches to baselines.jsonl. Do NOT re-run this per training step --
the whole point of the frozen first pass is that the reward signal is stable.

The script is resumable: it appends one JSON line per document as soon as that
document's low-res and high-res calls both land, and on restart it skips any
doc_id already present in the cache. Interrupting it is safe.

Usage:
    export PRIME_API_KEY=...            # or rely on ~/.prime/config.json
    uv run python data_prep/compute_baselines.py --limit 4     # cheap smoke
    uv run python data_prep/compute_baselines.py               # full corpus
"""

import argparse
import asyncio
import base64
import io
import json
import os
import random
from pathlib import Path

from datasets import load_from_disk
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm

# Prime Inference is OpenAI-compatible. Verified against
# ~/.prime/config.json ("inference_url") and prime_cli.core.config defaults.
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


def encode_image(path: str, upscale: int = 1) -> str:
    """Encode an image, optionally enlarged first.

    These captures are tiny -- a full newspaper clipping can be 129x130 -- and
    at native size the vision encoder allocates only ~100 tokens to the whole
    page, which sends small models into repetition loops. Upscaling adds no
    information but buys the encoder more patches, and on this corpus it moves
    Qwen3.5-4B's low-res CER from 4.18 (degenerate) to 0.34. See
    probe_baseline_models.py for the measurements.
    """
    if upscale <= 1:
        raw = Path(path).read_bytes()
    else:
        with Image.open(path) as im:
            im = im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            raw = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


class Transcriber:
    def __init__(self, client: AsyncOpenAI, model: str, max_tokens: int, retries: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.retries = retries

    async def __call__(self, image_path: str, upscale: int = 1) -> tuple[str, str]:
        """Returns (text, finish_reason). finish_reason=='error:...' on failure."""
        data_url = await asyncio.to_thread(encode_image, image_path, upscale)
        last_err = ""
        for attempt in range(self.retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
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
                    max_tokens=self.max_tokens,
                    # The first pass should transcribe, not deliberate. Qwen3.5
                    # otherwise burns the whole budget in reasoning_content and
                    # returns content=None.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = resp.choices[0]
                return choice.message.content or "", choice.finish_reason or "unknown"
            except Exception as exc:  # noqa: BLE001 - surface as a per-doc failure
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    await asyncio.sleep(2**attempt + random.random())
        return "", f"error:{last_err[:200]}"


async def run(args: argparse.Namespace) -> None:
    splits = load_from_disk(str(args.dataset))

    done: set[str] = set()
    if args.out.exists() and not args.force:
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["doc_id"])

    pending = []
    for split_name in args.splits:
        for ex in splits[split_name]:
            if ex["doc_id"] in done:
                continue
            pending.append((split_name, ex))
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"cached={len(done)} pending={len(pending)} model={args.model}")
    if not pending:
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key=resolve_api_key())
    transcribe = Transcriber(client, args.model, args.max_tokens, args.retries)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    mode = "w" if args.force else "a"
    failures = 0

    with args.out.open(mode, encoding="utf-8") as out_f:
        pbar = tqdm(total=len(pending), desc="baselines")

        async def one(split_name: str, ex: dict) -> None:
            nonlocal failures
            async with sem:
                low, low_fr = await transcribe(ex["low_res_path"], args.low_upscale)
                high, high_fr = await transcribe(ex["high_res_path"], args.high_upscale)
            record = {
                "doc_id": ex["doc_id"],
                "split": split_name,
                "baseline_low": low,
                "baseline_high": high,
                "finish_low": low_fr,
                "finish_high": high_fr,
                "model": args.model,
                # The corrector must see the SAME image the baseline was read
                # from, or the CER delta measures preprocessing rather than
                # correction. load_environment() checks image_upscale against
                # this field and refuses to run on a mismatch.
                "low_upscale": args.low_upscale,
                "high_upscale": args.high_upscale,
            }
            if low_fr.startswith("error") or high_fr.startswith("error"):
                failures += 1
            async with write_lock:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)

        await asyncio.gather(*(one(s, ex) for s, ex in pending))
        pbar.close()

    await client.close()
    print(f"wrote {args.out} (+{len(pending)} docs, {failures} with errors)")
    if failures:
        print("re-run the same command to retry failed docs after removing their lines")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--out", type=Path, default=Path("baselines.jsonl"))
    # Qwen3.5-9B and -122B-A10B are listed as available but time out on the
    # inference endpoint (100-230s, no response). 4B is responsive and is on the
    # hosted-training list too, which keeps the one-model-two-roles setup intact.
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--base-url", default=os.environ.get("PRIME_INFERENCE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    # Upscale factors measured on this corpus (see probe_baseline_models.py):
    # low 4x lands the baseline at CER ~0.25, inside the RL signal band, while
    # high 2x gives an honest ceiling at ~0.13. low-upscale MUST match the
    # environment's image_upscale.
    ap.add_argument("--low-upscale", type=int, default=4)
    ap.add_argument("--high-upscale", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=-1, help="cap pending docs (smoke tests)")
    ap.add_argument("--force", action="store_true", help="ignore and overwrite the cache")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
