"""Direct OCR environment: low-res document image in, transcription out.

This is the sibling of `ocr-corrector`, with the first-pass draft removed from
the prompt. It exists because the correction framing turned out to be a poor RL
target on this corpus:

* pass@16 on the correction task captured 2.1% of the available headroom
  (best-of-16 CER 0.259 vs a 0.262 baseline). GRPO can only amplify behaviour
  the policy already samples, and a draft in the context anchors every rollout
  to the draft's errors, collapsing within-group variance.
* 34% of correction rollouts produced no parseable output at all, because the
  required <reasoning> block sends this model into `Wait, ...` loops on
  ambiguous glyphs.

So this environment strips both: no draft, no reasoning block, no XML tags. The
prompt is deliberately the same shape that generated `baselines.jsonl` without a
single failure across 1,658 calls.

It also supports `loss = "sft"` on hosted training with an external teacher (a
frontier VLM reads these pages at ~0.02 CER, versus 0.25 for Qwen3.5-4B), and
`loss = "rl"` warm-started from an SFT checkpoint. The reward is computed
against the *human* transcription in both cases, so the verifiable signal never
comes from the teacher.
"""

from __future__ import annotations

import base64
import io
import math
import re
import unicodedata
from pathlib import Path

import jiwer
import verifiers as vf
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image

# -------- Text normalization (identical to ocr_corrector; see note below) -----

# Duplicated rather than imported: each environment must be independently
# installable. Any change here must be mirrored in ocr_corrector.py, or the two
# environments' CER numbers stop being comparable.
_NORM = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh      -> Farsi yeh
        "ى": "ی",  # Alef maksura    -> Farsi yeh
        "ك": "ک",  # Arabic kaf      -> Keheh
        "ـ": "",  # tatweel (kashida)     -> drop
        "‌": " ",  # ZWNJ                 -> space
        "‍": "",  # ZWJ                   -> drop
        "‎": "",  # LRM                   -> drop
        "‏": "",  # RLM                   -> drop
    }
)

# Cap on CER before it enters the reward. Two competing pressures:
#
#   Too low (1.0, the original): an output twice as long as the label already
#   scores ~1.0 from insertions alone, and one three times as long scores 1.0
#   too -- so past ~2x length, additional garbage is FREE. The first 100-step run
#   walked straight into that plateau: length ratio drifted 0.93 -> 2.17 while
#   CER rose 0.337 -> 0.585 and reward fell 0.66 -> 0.26.
#
#   Too high (uncapped): a repetition-looped rollout scores 4.0+, inflating its
#   group's reward std and shrinking every other rollout's advantage to noise.
#
# 1.5 keeps a live gradient against over-generation out to ~2.5x length while
# bounding any single rollout's influence on its group.
CER_CAP = 1.5


def normalize(s: str | None) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(_NORM)
    return re.sub(r"\s+", " ", s).strip()


def cer_breakdown(reference: str, hypothesis: str) -> dict:
    """Capped CER plus its substitution / deletion / insertion decomposition.

    All three are normalised by reference length, the same denominator CER uses,
    so sub + del + ins == uncapped CER. This is the diagnostic that separates
    the two ways of being wrong: misreading glyphs shows up as substitutions,
    while rambling past the end of the page shows up purely as insertions.
    """
    if not reference:
        return {"cer": 0.0 if not hypothesis else CER_CAP, "sub": 0.0, "del": 0.0, "ins": 0.0}
    if not hypothesis:
        return {"cer": CER_CAP, "sub": 0.0, "del": 1.0, "ins": 0.0}
    out = jiwer.process_characters(reference, hypothesis)
    n = max(1, len(reference))
    return {
        "cer": min(float(out.cer), CER_CAP),
        "sub": out.substitutions / n,
        "del": out.deletions / n,
        "ins": out.insertions / n,
    }


def cer(reference: str, hypothesis: str) -> float:
    return cer_breakdown(reference, hypothesis)["cer"]


# -------- Output cleaning --------

# Models sometimes wrap the transcription in a markdown fence or open with a
# preamble. Stripping those is not leniency -- charging the model CER for
# "Here is the transcription:" would measure instruction-following, not OCR.
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_PREAMBLE = re.compile(
    r"^\s*(here (is|are)[^:\n]*:|the transcription[^:\n]*:|transcription:)\s*",
    re.IGNORECASE,
)


def extract_transcription(completion) -> str | None:
    if isinstance(completion, list):
        if not completion:
            return None
        text = completion[-1].get("content")
    else:
        text = completion
    if not text:
        return None
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    text = _PREAMBLE.sub("", text)
    return text.strip() or None


# -------- Prompt --------

INSTRUCTION = (
    "Transcribe all text in this document image. "
    "Preserve line breaks. Output only the transcription."
)


def _encode_image_bytes(raw: bytes, upscale: int, fmt: str = "jpeg", quality: int = 90) -> str:
    """Upscale and encode as a data URL.

    Encoding format matters operationally, not just for storage. A 4x-upscaled
    PNG averages ~770KB of base64 on this corpus, and at 128 in-flight rollouts
    that is ~98MB of payload moving concurrently -- which is what produced a 47%
    ModelError rate on the first 100-step run (the 16-rollout probe, ~12MB, saw
    zero errors). Measured over 8 documents at 4x:

        PNG      770 KB base64   median CER 0.250
        JPEG q85 200 KB base64   median CER 0.258
        WEBP q90 138 KB base64   median CER 0.253

    Accuracy is flat across all three, so the PNG payload buys nothing. JPEG is
    the default rather than the smaller WEBP purely for decoder ubiquity in the
    serving stack; WEBP is available via image_format if payload matters more.
    """
    fmt = fmt.lower()
    if upscale <= 1 and fmt == "png":
        return f"data:image/png;base64,{base64.b64encode(raw).decode()}"

    with Image.open(io.BytesIO(raw)) as im:
        if upscale > 1:
            im = im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)
        buf = io.BytesIO()
        if fmt in ("jpeg", "jpg"):
            im.convert("RGB").save(buf, format="JPEG", quality=quality)
            mime = "jpeg"
        elif fmt == "webp":
            im.convert("RGB").save(buf, format="WEBP", quality=quality)
            mime = "webp"
        elif fmt == "png":
            im.save(buf, format="PNG", optimize=True)
            mime = "png"
        else:
            raise ValueError(f"unsupported image_format {fmt!r}; use jpeg, webp or png")
        raw = buf.getvalue()
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}"


def _build_prompt(data_url: str) -> list[dict]:
    # No system_prompt= on the env: that path prepends a string-content system
    # message to this list-content user message, and the two cannot share one
    # Arrow column ("cannot mix list and non-list, non-null values").
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]


# -------- Data loading --------


def _rows_from_local(dataset_path: Path, split: str) -> list[dict]:
    raw = load_from_disk(str(dataset_path))[split]
    return [
        {
            "doc_id": ex["doc_id"],
            "image_bytes": Path(ex["low_res_path"]).read_bytes(),
            "ground_truth": ex["ground_truth"],
            "language": ex["language"],
        }
        for ex in raw
    ]


def _rows_from_hub(dataset_id: str, split: str, revision: str | None) -> list[dict]:
    raw = load_dataset(dataset_id, split=split, revision=revision)
    rows = []
    for ex in raw:
        image = ex["image"]
        if isinstance(image, dict):
            raw_bytes = image["bytes"]
        elif isinstance(image, (bytes, bytearray)):
            raw_bytes = bytes(image)
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        rows.append(
            {
                "doc_id": ex["doc_id"],
                "image_bytes": raw_bytes,
                "ground_truth": ex["ground_truth"],
                "language": ex.get("language", "unknown"),
            }
        )
    return rows


def _to_dataset(
    rows: list[dict], *, upscale: int, max_examples: int, fmt: str, quality: int
) -> Dataset:
    records = []
    for r in rows:
        if not r["ground_truth"].strip():
            continue
        records.append(
            {
                "prompt": _build_prompt(
                    _encode_image_bytes(r["image_bytes"], upscale, fmt, quality)
                ),
                "answer": r["ground_truth"],
                "info": {"doc_id": r["doc_id"], "language": r["language"]},
            }
        )
        if 0 < max_examples <= len(records):
            break
    if not records:
        raise ValueError("no usable examples found")
    return Dataset.from_list(records)


# -------- Scoring --------


def _scores(completion, answer: str, state: dict) -> dict:
    cached = state.get("_ocr_scores")
    if cached is not None:
        return cached
    out = extract_transcription(completion)
    gt = normalize(answer)
    norm_out = normalize(out) if out else ""
    b = cer_breakdown(gt, norm_out)
    scores = {
        "parsed": bool(norm_out),
        "gt_len": len(gt),
        "out_len": len(norm_out),
        **b,
    }
    state["_ocr_scores"] = scores
    return scores


def make_rubric(*, length_coef: float) -> vf.Rubric:
    async def transcription_accuracy(completion, answer, state) -> float:
        """Primary signal: 1 - CER against the human transcription.

        Range [1 - CER_CAP, 1] = [-0.5, 1]. Absolute accuracy, not a delta
        against a draft. Computed against human ground truth in every training
        mode, including distillation -- the teacher supplies the imitation
        target, this supplies the verification, and they stay separate.
        """
        return 1.0 - _scores(completion, answer, state)["cer"]

    async def length_regularizer(completion, answer, state) -> float:
        """Smooth, weak pressure toward the label's length: -coef * |ln(ratio)|.

        Replaces a hard -0.5 cliff outside [0.4, 2.0]. That cliff was the
        proximate cause of the first run's collapse: it is flat everywhere
        inside the band, so nothing pulled the policy back toward ratio 1.0 as
        it drifted from 0.93 to 2.0, and then it is a sheer wall, which
        whipsawed the policy from over-long straight into empty outputs (51.5%
        empty at step 25).

        This version is zero at ratio 1.0 and rises smoothly and symmetrically
        in both directions -- there is always a restoring force, and it never
        has an edge to fall off. Deliberately weak (0.1 * ln) against an
        accuracy term spanning 1.5: at 2x length it costs 0.07, far less than
        the ~0.5 of accuracy that over-generation destroys. Per Ramp's rule,
        shaping must never be worth more than solving the actual task.
        """
        s = _scores(completion, answer, state)
        if not s["parsed"] or not s["gt_len"] or not s["out_len"]:
            return 0.0
        return -length_coef * abs(math.log(s["out_len"] / s["gt_len"]))

    async def m_cer(completion, answer, state) -> float:
        return _scores(completion, answer, state)["cer"]

    async def m_sub_rate(completion, answer, state) -> float:
        """Substitutions / label length: reading the wrong glyph. This is the
        error the project exists to reduce."""
        return _scores(completion, answer, state)["sub"]

    async def m_del_rate(completion, answer, state) -> float:
        """Deletions / label length: text on the page the model never emitted."""
        return _scores(completion, answer, state)["del"]

    async def m_ins_rate(completion, answer, state) -> float:
        """Insertions / label length: text the model invented. Over-generation
        shows up here and nowhere else, so watching this against m_sub_rate
        separates 'reading better' from 'writing more'."""
        return _scores(completion, answer, state)["ins"]

    async def m_length_ratio(completion, answer, state) -> float:
        s = _scores(completion, answer, state)
        return s["out_len"] / s["gt_len"] if s["gt_len"] else 0.0

    async def m_empty(completion, answer, state) -> float:
        """Share of rollouts producing nothing usable. Went 0.000 -> 0.515
        between steps 24 and 25 of the first run as the policy collapsed."""
        return 0.0 if _scores(completion, answer, state)["parsed"] else 1.0

    rubric = vf.Rubric(
        funcs=[transcription_accuracy, length_regularizer],
        weights=[1.0, 1.0],
    )
    for metric in (m_cer, m_sub_rate, m_del_rate, m_ins_rate, m_length_ratio, m_empty):
        rubric.add_metric(metric)
    return rubric


# -------- Entry point --------


def load_environment(
    dataset_path: str = "hf_dataset",
    dataset_id: str | None = None,
    revision: str | None = None,
    train_split: str = "train",
    eval_split: str = "test",
    max_examples: int = -1,
    max_eval_examples: int = -1,
    image_upscale: int = 4,
    image_format: str = "jpeg",
    image_quality: int = 90,
    length_coef: float = 0.1,
    max_tokens: int = 3072,
    enable_thinking: bool = False,
) -> vf.Environment:
    """Load the direct OCR environment.

    Args:
        dataset_id: HF Hub dataset with image bytes embedded. Required for
            hosted training, which cannot see this workspace's data/ folder.
        image_upscale: enlarge the low-res capture before encoding. These pages
            are ~129x130px; at native size Qwen3.5's vision encoder spends ~100
            tokens on a whole newspaper page and loops. 4x is the measured knee
            (CER 3.006 -> 0.251). Frontier VLMs are insensitive to this.
        image_format: "jpeg" (default), "webp" or "png". PNG averages 468KB of
            base64 per example here and peaks at 2.4MB; JPEG q90 averages 133KB
            for the same CER. The payload is carried per rollout, so this is a
            throughput and memory knob, not a storage one.
        enable_thinking: off by default -- left on, this model burns 40k+ tokens
            re-litigating single glyphs and never emits an answer.
    """

    def rows(split: str) -> list[dict]:
        if dataset_id:
            return _rows_from_hub(dataset_id, split, revision)
        return _rows_from_local(Path(dataset_path), split)

    def builder(split: str, cap: int):
        """Defer materialisation until the split is actually used.

        A run starts one env-server per environment -- here a training server
        and a separate eval server. Building both splits eagerly meant the eval
        server also held the full 746-document training set, encoded, that it
        never reads, and vice versa. Lazy builders keep each server to the split
        it serves.
        """

        def build() -> Dataset:
            return _to_dataset(
                rows(split),
                upscale=image_upscale,
                max_examples=cap,
                fmt=image_format,
                quality=image_quality,
            )

        return build

    dataset = builder(train_split, max_examples)
    eval_dataset = builder(eval_split, max_eval_examples)

    # NOTE: max_tokens here is overridden by `prime eval run`, which always
    # passes an explicit max_tokens (None when -t is omitted) and shallow-updates
    # over these. Pass `-t 3072` to evals. Training reads [sampling] from TOML.
    sampling_args: dict = {"max_tokens": max_tokens}
    if not enable_thinking:
        sampling_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        sampling_args=sampling_args,
        rubric=make_rubric(length_coef=length_coef),
    )
