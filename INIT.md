# INIT: Urdu/Persian OCR Corrector via RL on Prime Lab

This document seeds the repo. It captures the project scope, research
framing, environment code sketch, training plan, and known risks. Written
against Prime Lab as of August 2026 (Lab has been GA since May 7, 2026).

Read this end-to-end before writing any code or launching any run. Section 3
in particular ("Research framing") argues *against* several plausible-sounding
project shapes and settles on one; if you skip it you will make the wrong
call somewhere.

---

## 1. Goal

Train a LoRA adapter on a small multimodal model that acts as an **OCR
corrector**: given a document image and a first-pass OCR transcription, it
produces an improved transcription. Focused on Urdu and Persian documents,
where the dominant baseline failure modes are (a) dot-based letter confusion
(the Arabic-script family distinguishes many letters purely by dot count and
placement) and (b) missed or garbled regions in low-quality captures.

The corrector is trained on Prime Lab (hosted RL, LoRA) with a
document-level CER-improvement reward.

## 2. Current state of Prime Lab (Aug 2026)

Verified from docs and status page before writing:

- **GA since May 7, 2026.** Public beta since Feb 2026. Both hosted training
  and hosted evaluations are operational.
- **Hosted training is RL-only.** SFT, GEPA, DPO, GKD are on the roadmap;
  not shipped. Trainer is LoRA on top of `prime-rl` with GRPO.
- **Hosted model list (as of Aug 2026)** includes: Qwen3.5-0.8B / -2B / -4B
  / -9B / -35B-A3B, Qwen3.6-35B-A3B, openai/gpt-oss-20b, Meta Llama-3.2-1B /
  -3B, NVIDIA Nemotron-3-Nano-30B-A3B, poolside/Laguna-XS-2.1. Confirm with
  `prime rl models` before writing configs; the list drifts.
- **Qwen3.5 is natively multimodal** (early-fusion vision-language). No
  separate `-VL` variant needed. Image inputs use OpenAI-style
  `image_url` content blocks. This is the family we target.
- **Verifiers library** supports multimodal-input tasks; the reference is
  `environments/mmmu` in the `PrimeIntellect-ai/verifiers` repo. Read
  that file before starting on the environment code; it is the canonical
  pattern for how images are attached to prompt messages.
- **Prime Sprints (May 20, 2026)** is Prime Intellect's community
  research program on reward hacking. Their finding: reward hacking is
  systematic and appears at 1B scale within 30 minutes of training. Their
  released environments demonstrate this. Section 8 of this document
  builds a monitoring checklist inspired by their work; treat it as
  non-optional.
- **Config is TOML** (`configs/lab/*.toml`), launch is
  `prime rl run <config>`, monitor via `prime rl logs -f` or the dashboard.
- **Pricing is per-token** for hosted training, not cluster-hour. Small
  validation runs cost cents; medium runs cost a few dollars. Do not
  optimize wall-time; optimize token spend on runs that give clean signal.

## 3. Research framing

This is the section that gets skipped and shouldn't be.

### 3.1 Why not just SFT?

Our dataset is `(image, ground_truth_text)` pairs. This is the textbook
supervised setup: cross-entropy on the target string per image. SFT would
almost certainly outperform RL per dollar on the exact same data, because
every token gets a dense gradient signal instead of a sparse trajectory-level
reward.

**We are doing RL anyway for two deliberate reasons:**

1. **Lab is RL-only today.** SFT support is on the roadmap but not shipped.
   Doing SFT means going off-platform (rent a GPU pod, use Unsloth or TRL,
   ship the LoRA back). We accept the trade-off because a stated goal is to
   learn Prime Lab specifically.
2. **The task shape we chose is genuinely RL-shaped.** We are not doing
   end-to-end image-to-text OCR via RL (which is a bad shape — every token
   shares one document-level reward, credit assignment is diffuse, and the
   model has to bootstrap the whole capability). We are doing
   **correction** on top of a frozen first-pass, where the action is
   improve-a-noisy-string and the reward is CER-delta versus the baseline
   input. This is closer to a retrieval/verifier task. RL handles it well;
   SFT would need synthetic correction targets we don't have.

If your goal shifts from "learn Lab and ship a useful corrector" to
"maximize OCR quality per dollar," reopen this decision. SFT-first,
RL-as-polish is the higher-EV recipe for pure quality.

### 3.2 One model, two roles

At training time we use **the same base model in two roles**:

- **First-pass OCR:** frozen Qwen3.5-9B, called via Prime Inference
  once per document, cached to disk as `baselines.jsonl`. This runs
  outside the training loop.
- **Corrector:** Qwen3.5-9B + LoRA adapter, updated by Lab. Input at
  rollout time is (low-res image, cached baseline text). Output is the
  corrected transcription.

Same base weights on disk, differing only by whether the LoRA is loaded.
Operationally clean and matches Lab's adapter deployment model. The
frozen baseline is *not* re-generated per rollout, both because it would
be wasteful and because we want a stable reward signal.

If we later want a different baseline distribution (a cheaper OCR
engine, a different VLM, sampled baselines for diversity), we regenerate
`baselines.jsonl` once and rerun. The training code doesn't change.

### 3.3 What we do with the high-res / low-res pairing

The dataset has both a high-res and a low-res image per ground truth. This
is genuinely useful and we should exploit it, but *not* by putting both
images in the model's context.

- **Training input to the corrector: low-res image only.** This matches
  the production distribution (real docs are captured under real
  conditions; you don't get a matching high-res twin in the field).
- **High-res image use #1: ground-truth sanity check.** Run high-res
  through the baseline OCR too, cache separately. If high-res-baseline
  disagrees strongly with the provided ground truth on a given document,
  the ground truth is suspect; flag and review. This is a data-quality
  audit, not a training signal.
- **High-res image use #2: achievability ceiling.** The CER a frozen
  baseline achieves on high-res is an upper bound on what our
  low-res-input corrector can realistically achieve — the corrector
  can't invent information the low-res image doesn't contain. Reporting
  trained-corrector CER against both baselines gives us a headroom
  metric: "how much of the theoretically-recoverable improvement did
  we capture?"
- **Optional stretch:** use high-res-baseline as a stronger supervision
  target than the provided ground truth if the human labels turn out to
  be inconsistent (line-break conventions, diacritic policy, etc.).
  Decide after auditing.

We do **not** feed both images to the model at rollout time. Doing so
would let the model cheat by reading the high-res version, and the
resulting corrector would be useless in production.

### 3.4 Why the reward is not 1/0

A single "matches ground truth or not" reward has known problems for
this task:

- **Sparsity:** the corrector is rarely producing an exact-match
  transcription, so most rollouts get 0. GRPO needs reward variance
  within a group of ~8-16 rollouts per example; if 14/16 get zero, the
  gradient is dominated by noise.
- **No-op incentive:** copy-the-baseline gets reward 0, which is often
  a group-median outcome and therefore risk-free. The model learns to
  do nothing.
- **No reasoning shaping:** we want the corrector to look at dots,
  scan for missed regions, and use context. None of that is expressible
  as a binary output-match.

The design in Section 6.5 uses a composite reward with format bonus,
length guardrail, CER-improvement delta, and an abstention-friendly
structure. Each component is logged separately so we can diagnose
which one is moving.

## 4. Dataset assumptions

Assumed layout (adjust `data_prep/build_dataset.py` if wrong):

```
data/
├── high_res/{i}.png       # high-res capture, i = 0000, 0001, ...
├── low_res/{i}.png        # low-res capture of the same document
└── ground_truth/{i}.txt   # verified transcription, UTF-8
```

If your actual layout differs (different folder names, JSONL manifest,
language subfolders), only `build_dataset.py` needs to change; the rest
of the pipeline reads from the built HF dataset.

Assumptions to verify before Phase 1:
- Ground truth text is UTF-8 and uses consistent Unicode forms for the
  Arabic-script letter variants (Arabic vs Farsi yeh, Arabic vs Keheh
  kaf). Normalization in `normalize()` handles the common cases but
  won't recover from ground truth that mixes conventions inconsistently.
- Dataset size is ~1k-10k documents. If it's smaller, GRPO variance
  will suffer; consider chunk-per-page to expand effective example
  count. If it's much larger, first pass at 1-2k for iteration speed.
- Language balance: if you have both Urdu and Persian, tag the language
  per-example so we can stratify metrics. If you don't know the language
  per document, a fast Persian-vs-Urdu classifier is a 30-line script
  or you can heuristically infer from character frequency
  (Persian-specific letters پ چ ژ گ, Urdu-specific ٹ ڈ ڑ etc).

## 5. Repo layout

```
ocr-corrector/
├── INIT.md                      # this file
├── AGENTS.md                    # (pre-existing)
├── CLAUDE.md                    # (pre-existing)
├── configs/
│   └── lab/
│       ├── smoke.toml           # 50-step validation
│       └── medium.toml          # 300-step iteration
├── environments/
│   └── ocr_corrector/
│       ├── ocr_corrector.py     # verifiers env
│       ├── pyproject.toml
│       └── README.md
├── data_prep/
│   ├── build_dataset.py         # raw folders -> HF Dataset on disk
│   ├── compute_baselines.py     # frozen first-pass OCR, cached
│   └── audit_baselines.py       # CER distribution + sanity checks
├── data/                        # raw (see Section 4)
├── hf_dataset/                  # built by build_dataset.py
├── baselines.jsonl              # written by compute_baselines.py
└── pyproject.toml
```

## 6. Implementation walkthrough

### 6.1 One-time environment setup

```bash
# uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Prime CLI
uv tool install prime
prime login

# In the repo root
prime lab setup

# Verify multimodal model is on hosted training
prime rl models   # look for Qwen/Qwen3.5-9B or nearest equivalent
```

Sanity-check the platform end-to-end with a stock env before writing your
own:

```bash
prime env install primeintellect/alphabet-sort
prime eval run primeintellect/alphabet-sort \
  -m Qwen/Qwen3.5-0.8B -n 5 -r 1
prime eval tui
```

If that works, your account, auth, and endpoint are wired up.

### 6.2 Build the HuggingFace dataset

`data_prep/build_dataset.py`:

```python
"""Convert raw data/ folders into a HuggingFace Dataset on disk."""
from pathlib import Path
from datasets import Dataset, Features, Value

RAW = Path("./data")
OUT = Path("./hf_dataset")

def gen():
    gt_dir = RAW / "ground_truth"
    for gt_path in sorted(gt_dir.glob("*.txt")):
        idx = gt_path.stem
        low = RAW / "low_res" / f"{idx}.png"
        high = RAW / "high_res" / f"{idx}.png"
        if not (low.exists() and high.exists()):
            continue
        yield {
            "doc_id": idx,
            "low_res_path": str(low),
            "high_res_path": str(high),
            "ground_truth": gt_path.read_text(encoding="utf-8").strip(),
            "language": infer_language(gt_path.read_text(encoding="utf-8")),
        }

def infer_language(text: str) -> str:
    # Persian-only letters: پ چ ژ گ; Urdu-only: ٹ ڈ ڑ ں ے
    urdu_hits = sum(c in "ٹڈڑںے" for c in text)
    persian_hits = sum(c in "پچژگ" for c in text)
    if urdu_hits > persian_hits:
        return "ur"
    if persian_hits > urdu_hits:
        return "fa"
    return "unknown"

features = Features({
    "doc_id": Value("string"),
    "low_res_path": Value("string"),
    "high_res_path": Value("string"),
    "ground_truth": Value("string"),
    "language": Value("string"),
})

ds = Dataset.from_generator(gen, features=features)
splits = ds.train_test_split(test_size=0.10, seed=42)
splits.save_to_disk(OUT)
print(f"train={len(splits['train'])} test={len(splits['test'])}")
```

Notes:
- Storing paths as strings rather than embedding image bytes in the schema.
  Simpler, and the rollout code will base64-encode at prompt-build time.
- `infer_language` is a heuristic; replace with actual language tags if
  you have them.
- If you later push this to the HF Hub, switch to `Image()` features for
  portability.

### 6.3 Compute frozen baselines

`data_prep/compute_baselines.py`. Runs once, cached, do not re-run per
training step:

```python
"""Compute frozen first-pass OCR baselines for both low-res and high-res.
Low-res baselines are the actual corrector input; high-res baselines are the
achievability-ceiling reference.
"""
import asyncio
import base64
import json
import os
from pathlib import Path

from datasets import load_from_disk
from openai import AsyncOpenAI

DATASET_DIR = Path("./hf_dataset")
CACHE_PATH = Path("./baselines.jsonl")

# Prime Inference exposes an OpenAI-compatible endpoint.
# Verify the exact base_url and env-var name via `prime inference --help`
# or the dashboard's API access page before running.
BASE_URL = os.environ.get("PRIME_INFERENCE_BASE_URL",
                          "https://api.pinference.ai/api/v1")
API_KEY = os.environ["PRIME_API_KEY"]
MODEL = "Qwen/Qwen3.5-9B"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

PROMPT = (
    "Transcribe all text in this document image. "
    "Preserve line breaks. Output only the transcription."
)

async def transcribe(image_path: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            temperature=0.0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

async def main():
    splits = load_from_disk(DATASET_DIR)
    sem = asyncio.Semaphore(8)
    out = []
    for split_name in ["train", "test"]:
        low_tasks = [transcribe(ex["low_res_path"], sem)
                     for ex in splits[split_name]]
        high_tasks = [transcribe(ex["high_res_path"], sem)
                      for ex in splits[split_name]]
        low_results = await asyncio.gather(*low_tasks)
        high_results = await asyncio.gather(*high_tasks)
        for ex, low, high in zip(splits[split_name],
                                 low_results, high_results):
            out.append({
                "doc_id": ex["doc_id"],
                "split": split_name,
                "baseline_low": low,
                "baseline_high": high,
            })
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
```

Cost check: this is 2 API calls per document. For ~2000 docs at, say,
1200 tokens per call, it's ~5M tokens total; a few dollars on Prime
Inference. Cheap.

### 6.4 Audit the baselines before you train

`data_prep/audit_baselines.py`:

```python
"""Print CER distribution on low-res and high-res baselines, plus disagreement
between high-res baseline and provided ground truth (data-quality signal).
"""
import json
from pathlib import Path
from collections import defaultdict

import jiwer
import numpy as np
from datasets import load_from_disk

splits = load_from_disk("./hf_dataset")
gt = {ex["doc_id"]: ex["ground_truth"] for ex in splits["train"]}
lang = {ex["doc_id"]: ex["language"] for ex in splits["train"]}

low_cers, high_cers = [], []
by_lang = defaultdict(lambda: {"low": [], "high": []})

for line in open("./baselines.jsonl"):
    r = json.loads(line)
    if r["split"] != "train":
        continue
    truth = gt.get(r["doc_id"])
    if truth is None:
        continue
    lc = jiwer.cer(truth, r["baseline_low"])
    hc = jiwer.cer(truth, r["baseline_high"])
    low_cers.append(lc)
    high_cers.append(hc)
    by_lang[lang[r["doc_id"]]]["low"].append(lc)
    by_lang[lang[r["doc_id"]]]["high"].append(hc)

def summarize(name, values):
    a = np.array(values)
    print(f"{name:20s} n={len(a):4d} median={np.median(a):.3f} "
          f"p25={np.percentile(a,25):.3f} p75={np.percentile(a,75):.3f}")

summarize("low-res  overall", low_cers)
summarize("high-res overall", high_cers)
for lg, vals in by_lang.items():
    summarize(f"low-res {lg}", vals["low"])
    summarize(f"high-res {lg}", vals["high"])

# Data-quality flag: high-res baseline strongly disagrees with GT
gap = [(l, h) for l, h in zip(low_cers, high_cers)]
flagged = sum(1 for _, h in gap if h > 0.30)
print(f"\nDocs where high-res baseline CER > 0.30 (GT may be off): {flagged}")

# Sweet-spot fraction for training
in_band = sum(1 for c in low_cers if 0.05 <= c <= 0.30)
print(f"Low-res baselines in [0.05, 0.30] band (best for RL): "
      f"{in_band}/{len(low_cers)} = {in_band/len(low_cers):.2%}")
```

**Go/no-go criterion:** if fewer than 30% of low-res baselines are in the
[0.05, 0.30] CER band, the reward signal is going to be weak (too easy or
too hard) and you need to either restratify (train only on the middle
band) or reconsider the project. If flagged docs are > 5%, audit the
ground truth before spending compute.

### 6.5 The verifiers environment

`environments/ocr_corrector/pyproject.toml`:

```toml
[project]
name = "ocr-corrector"
version = "0.1.0"
description = "RL-trained OCR corrector for Urdu/Persian documents"
requires-python = ">=3.10"
dependencies = ["verifiers", "datasets", "jiwer", "Pillow"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`environments/ocr_corrector/ocr_corrector.py`:

```python
"""OCR corrector environment.

Input to the model: low-res document image + baseline OCR from that image.
Output: corrected transcription.
Reward: improvement in CER versus baseline, plus format and length guards.
"""
import base64
import json
import re
import unicodedata
from pathlib import Path

import jiwer
import verifiers as vf
from datasets import load_from_disk

DATA_DIR = Path("./hf_dataset")
BASELINES_PATH = Path("./baselines.jsonl")

# -------- Text normalization --------

# Common Arabic to Persian/Urdu canonicalizations. Without this, we punish
# the model for producing the right character in the wrong codepoint.
_NORM = str.maketrans({
    "\u064A": "\u06CC",  # Arabic yeh -> Farsi yeh
    "\u0643": "\u06A9",  # Arabic kaf -> Keheh
    "\u0640": "",         # tatweel -> drop
})

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(_NORM)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------- Load cached baselines --------

def _load_baselines() -> dict[str, dict]:
    out = {}
    with open(BASELINES_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["doc_id"]] = r
    return out


# -------- Prompt construction --------

_INSTRUCTIONS = (
    "You are an OCR correction assistant for Urdu and Persian documents. "
    "You will see (1) an image of a document, and (2) a first-pass OCR "
    "transcription that likely contains errors. Common errors include "
    "wrong dot placement on letters that share the same body shape "
    "(for example ب / ت / ث / پ / ن), missing regions, and hallucinated "
    "text.\n\n"
    "Examine the image carefully. Produce a corrected transcription that "
    "preserves the original line structure. If the baseline is already "
    "correct, return it unchanged. Do not invent content not visible in "
    "the image.\n\n"
    "Format exactly as:\n"
    "<reasoning>brief notes on what you changed and why</reasoning>\n"
    "<corrected>the full corrected transcription</corrected>\n\n"
    "First-pass OCR:\n"
)

def _build_prompt(image_path: str, baseline_text: str) -> list[dict]:
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    return [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",
             "text": _INSTRUCTIONS + baseline_text},
        ],
    }]


def _build_dataset(split: str):
    raw = load_from_disk(DATA_DIR)[split]
    baselines = _load_baselines()

    def to_example(ex):
        b = baselines.get(ex["doc_id"], {})
        return {
            "prompt": _build_prompt(ex["low_res_path"],
                                    b.get("baseline_low", "")),
            "answer": ex["ground_truth"],
            "info": {
                "baseline_low": b.get("baseline_low", ""),
                "baseline_high": b.get("baseline_high", ""),
                "language": ex["language"],
                "doc_id": ex["doc_id"],
            },
        }

    return raw.map(to_example, remove_columns=raw.column_names)


# -------- Output parsing --------

_CORRECTED_RE = re.compile(r"<corrected>(.*?)</corrected>", re.DOTALL)

def _extract(completion) -> str | None:
    text = (completion[-1]["content"] if isinstance(completion, list)
            else completion)
    m = _CORRECTED_RE.search(text or "")
    return m.group(1).strip() if m else None


# -------- Reward components (each logged separately) --------

async def format_reward(completion, **kwargs) -> float:
    return 0.1 if _extract(completion) is not None else 0.0


async def length_guardrail(completion, state, **kwargs) -> float:
    """Zero out rollouts with pathological length. Guardrail, not a teacher."""
    corr = _extract(completion)
    if corr is None:
        return 0.0
    baseline = state["info"]["baseline_low"]
    if not baseline:
        return 0.0
    ratio = len(corr) / max(1, len(baseline))
    return 0.0 if 0.5 <= ratio <= 2.0 else -0.5


async def cer_improvement(completion, answer, state, **kwargs) -> float:
    """Main signal: CER(baseline, GT) - CER(corrected, GT), scaled."""
    corr = _extract(completion)
    if corr is None:
        return 0.0
    gt = normalize(answer)
    base = normalize(state["info"]["baseline_low"])
    out = normalize(corr)
    cer_base = jiwer.cer(gt, base) if base else 1.0
    cer_out = jiwer.cer(gt, out) if out else 1.0
    delta = cer_base - cer_out
    bonus = 0.05 if cer_out < cer_base else 0.0
    return 10.0 * delta + bonus


# -------- Non-reward metrics (logged, do not affect gradient) --------

async def m_baseline_cer(answer, state, **kwargs) -> float:
    return jiwer.cer(normalize(answer), normalize(state["info"]["baseline_low"]))

async def m_corrected_cer(completion, answer, state, **kwargs) -> float:
    corr = _extract(completion)
    if corr is None:
        return 1.0
    return jiwer.cer(normalize(answer), normalize(corr))

async def m_ceiling_cer(answer, state, **kwargs) -> float:
    """CER of the high-res baseline vs GT: the achievability ceiling."""
    return jiwer.cer(normalize(answer), normalize(state["info"]["baseline_high"]))

async def m_copied_baseline(completion, state, **kwargs) -> float:
    """1.0 if the corrector output equals the baseline verbatim.
    Watch this. High values = trivial-copy convergence (reward hacking)."""
    corr = _extract(completion)
    if corr is None:
        return 0.0
    return 1.0 if normalize(corr) == normalize(state["info"]["baseline_low"]) else 0.0


# -------- Entry point --------

def load_environment(split: str = "train") -> vf.Environment:
    dataset = _build_dataset(split)
    rubric = vf.Rubric(
        funcs=[format_reward, length_guardrail, cer_improvement],
        weights=[1.0, 1.0, 1.0],
        # Confirm exact kwarg name for non-reward metrics against current
        # verifiers version; the pattern is present but the API may be
        # `metric_funcs=` or nested inside the Rubric class.
        metrics=[m_baseline_cer, m_corrected_cer, m_ceiling_cer,
                 m_copied_baseline],
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

Install locally:

```bash
cd environments/ocr_corrector
uv pip install -e .
cd ../..
```

Points to verify before first run (see Section 9):
- Multimodal prompt format. The `image_url` content-block pattern above
  matches OpenAI convention and works with Qwen3.5's chat template via
  vLLM. Cross-check against `environments/mmmu/mmmu.py` in the verifiers
  repo, which is the canonical multimodal example.
- Exact API for non-reward metrics on `vf.Rubric`. The concept exists in
  the docs; the kwarg name should be checked against the installed
  verifiers version.
- `state["info"]` access from reward functions. Docs describe this
  pattern; a smoke eval will confirm.

### 6.6 Local smoke test

Run five examples with two rollouts each to check plumbing before
spending compute on training:

```bash
prime eval run ocr-corrector \
  -m Qwen/Qwen3.5-9B \
  -n 5 -r 2
prime eval tui
```

Checklist in the TUI:
- **Format reward ~0.1** on well-formed rollouts. If it's 0, the prompt
  isn't producing the `<corrected>` block.
- **`m_baseline_cer` populated** and reasonable (matches audit results).
- **`m_corrected_cer` roughly comparable to baseline** for now — the
  model is untrained.
- **`m_copied_baseline` is not always 1.0** — if the model is verbatim
  regurgitating, the prompt needs sharpening.
- **Eyeball 3 rollouts**: does the reasoning block reference specific
  characters or regions? If it's generic ("I checked the text"), the
  model isn't attending to the image.

Debug at this stage. Everything is 100x faster locally than after
launching hosted training.

### 6.7 Training configs

`configs/lab/smoke.toml`:

```toml
model = "Qwen/Qwen3.5-9B"
max_steps = 50
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 2048
temperature = 0.7

[[env]]
id = "ocr-corrector"
```

`configs/lab/medium.toml`:

```toml
model = "Qwen/Qwen3.5-9B"
max_steps = 300
batch_size = 256
rollouts_per_example = 16

[sampling]
max_tokens = 2048
temperature = 0.7

[wandb]
project = "ocr-corrector"
name = "qwen35-9b-medium"

[eval]
interval = 50

[val]
num_examples = 32
rollouts_per_example = 1
interval = 10

[buffer]
online_difficulty_filtering = true
```

`online_difficulty_filtering` drops examples whose rollout group has zero
reward variance (all-succeed or all-fail). This directly targets the
weak-signal risk from Section 6.4 — if a document's baseline is already
perfect, corrections can't improve it and the whole group gets ~0. Skip
those.

### 6.8 Launching

```bash
prime rl run configs/lab/smoke.toml
prime rl logs <run-id> -f     # or watch the dashboard URL from the launch output
```

If the smoke run shows reward trending up over 50 steps (even by a
small amount) and per-component metrics move sensibly, move to
`medium.toml`. If reward is flat or `m_copied_baseline` climbs to ~1.0,
stop and revisit rewards or prompts before spending more.

### 6.9 Evaluation

Held-out eval against the test split, comparing trained adapter vs
baseline:

```bash
# Baseline (base model, no adapter)
prime eval run ocr-corrector \
  -m Qwen/Qwen3.5-9B \
  --split test -n 200 -r 1

# Trained adapter
prime eval run ocr-corrector \
  -m <your-adapter-id-from-dashboard> \
  --split test -n 200 -r 1
```

Report:
- Mean `m_corrected_cer` vs `m_baseline_cer` (headline).
- Mean `m_ceiling_cer` (how much room was left to close).
- Stratified by language and by baseline-CER bucket.
- `m_copied_baseline` rate (should be low; high = trivial-copy hack).

The interesting number is not raw corrected CER; it's `(baseline_cer -
corrected_cer) / (baseline_cer - ceiling_cer)` — fraction of
recoverable error the corrector actually recovered.

## 7. Iteration loop

Expect the first medium run to expose specific failure modes. Diagnose
by opening rollouts on the dashboard, then adjust one thing at a time:

- If reward hacking (copy-baseline, length gaming): tighten
  `length_guardrail`, add a copy penalty term, or increase the
  `improvement_bonus`.
- If reward plateau but rollouts show good reasoning: prompt is fine,
  reward is saturated; look at reward variance per group. If variance
  is near zero, restratify data toward harder documents.
- If reward plateau and rollouts show shallow reasoning ("Looks
  correct."): the prompt isn't demanding real inspection. Add
  requirement to enumerate specific corrections in the reasoning
  block.
- If Persian or Urdu underperforms specifically: check normalization
  (may be dropping language-relevant codepoints) or add
  language-stratified sampling.

## 8. Reward hacking watchlist (mandatory before every run)

Prime Intellect's Prime Sprints research (May 2026) documents that
reward hacking is systematic, appears at 1B scale, and often only takes
30 minutes of training to emerge. Do not treat this as theoretical.

Specific failure modes to actively monitor:

- **Trivial copy convergence.** Watch `m_copied_baseline`. Rising
  toward 1.0 = the model has learned that copying is safest.
  Mitigations: increase `improvement_bonus`, add a small penalty for
  exact-match output.
- **Length gaming.** Corrector emits very short output to skip hard
  regions, gaining modest CER improvement on the easy parts.
  `length_guardrail` catches the gross version; watch output length
  distribution over training for subtler versions.
- **Format optimization over content.** `format_reward` climbs while
  `cer_improvement` stays flat. The model learned to wrap output in
  the right tags without actually correcting. Cap the format-reward
  weight; audit rollouts.
- **Ground-truth mimicry via prior knowledge.** If Qwen3.5 has
  memorized parts of Urdu/Persian literature, it may reproduce known
  passages verbatim regardless of what's in the image. Check by
  running the corrector with an image of a *different* document plus
  a specific baseline — does it correct toward the baseline's content
  or the image's content?
- **Reward function bugs that look like learning.** If reward suddenly
  spikes, check the reward function code before celebrating. A common
  bug: normalization mismatch between reward computation and eval
  metric, producing artificial delta.

Log all four metrics from Section 6.5. Do not turn off logging even on
production runs.

## 9. Things to verify before writing more code

I flagged these in-line above; consolidated here so nothing gets missed:

1. **Prime Inference base URL and auth env var name.** Check
   `prime inference --help` or the dashboard API access page. `PRIME_API_KEY`
   is likely; the endpoint host may differ from `api.pinference.ai`.
2. **Verifiers multimodal prompt convention.** Confirm against
   `environments/mmmu/mmmu.py` in the current
   `PrimeIntellect-ai/verifiers` main branch. If they use a different
   image content-block shape, match it.
3. **`vf.Rubric` API for non-reward metrics.** The `metrics=` kwarg name
   in the code above is my best guess; verify against installed
   verifiers version. If the API differs, adjust; the concept is
   supported.
4. **Qwen3.5-9B on hosted training and its context/token limits.** Run
   `prime rl models` to see current list, per-token pricing, and
   context length. Adjust `max_tokens` in configs if the model's context
   is tighter than assumed.
5. **Actual data layout.** `build_dataset.py` assumes
   `data/{high_res,low_res,ground_truth}/{i}.{png,txt}`. Adjust if your
   layout is different.

## 10. Open research questions (not blockers)

Deferred deliberately, not forgotten:

- **Should we chunk by page region?** If documents are long
  (multi-paragraph), full-document rewards get diffuse. Chunking by
  double-newline in the baseline gives per-chunk rewards without
  data-engineering the chunks manually. Try if reward signal is weak
  after Section 7.
- **Hallucination penalty via separate verifier.** Not in v1. Would
  require a second model or a heuristic (e.g., n-gram overlap with the
  low-res baseline) to detect content added out of thin air.
- **Sampled baselines for diversity.** Currently one baseline per
  document at temperature 0. Sampling multiple baselines per document
  at temperature > 0 would give the corrector a wider input distribution
  and may improve generalization. Cost: more inference calls up front.
- **Curriculum by baseline difficulty.** Start training on middle-band
  documents (CER 0.10-0.25), gradually expand. May improve early
  learning signal. Requires a custom sampler.

## 11. Time and cost budget

Rough estimates for planning:

- Data prep (build + baselines + audit): 3-5 hrs incl waiting on API calls
  and inspection. Baseline API cost: a few dollars.
- Environment + local smoke test: 3-4 hrs.
- Smoke training run: ~1 hr wall time, order-of-a-dollar per-token cost.
- Medium training run: 4-8 hrs wall time, likely tens of dollars.
- Iteration: open-ended; budget a couple of weeks of part-time to see
  real gains.

If the smoke run at $1 doesn't show any signal, do not scale up. Fix
the environment first.

---

**Written August 2026. Prime Lab moves fast; re-verify the model list,
API endpoints, and verifiers API against current docs before each
non-trivial change.**
