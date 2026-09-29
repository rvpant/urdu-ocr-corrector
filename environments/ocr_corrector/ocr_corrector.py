"""OCR corrector environment for Urdu/Persian documents.

Input to the model: a low-res document image plus the frozen first-pass OCR of
that same image. Output: a corrected transcription.
Reward: CER improvement over the baseline, plus format and length guards.

The first pass is computed once offline (`data_prep/compute_baselines.py`) and
cached, so the reward signal is stable across training steps.

Two data sources are supported:

* **local** (default) -- reads the on-disk dataset built by
  `data_prep/build_dataset.py` plus `baselines.jsonl`, encoding images from
  their filesystem paths. Use for local eval and iteration.
* **hub** (`dataset_id=...`) -- reads a self-contained dataset with image bytes
  and baselines embedded, built by `data_prep/package_dataset.py`. Required for
  hosted training, where the environment container has no access to this
  workspace's `data/` folder.
"""

from __future__ import annotations

import base64
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import jiwer
import verifiers as vf
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image

# -------- Text normalization --------

# Arabic-script canonicalizations shared by Urdu and Persian. Without these we
# would punish the model for emitting the right character in the wrong
# codepoint. Deliberately conservative: it does NOT fold the Urdu-specific
# retroflex/aspirate letters or ں/ے, which carry real orthographic content.
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

# Cap CER before differencing. jiwer's CER is unbounded above (insertions), and
# an untrained model looping on a degenerate repetition can score 4.0+. Without
# a cap a single blown-up rollout swamps the group advantage in GRPO.
CER_CAP = 1.0


def normalize(s: str | None) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(_NORM)
    return re.sub(r"\s+", " ", s).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Capped character error rate. 1.0 means 'worthless', as does 4.0."""
    if not reference:
        return 0.0 if not hypothesis else CER_CAP
    if not hypothesis:
        return CER_CAP
    return min(float(jiwer.cer(reference, hypothesis)), CER_CAP)


# -------- Prompt construction --------

SYSTEM_PROMPT = (
    "You are an OCR correction assistant for Urdu and Persian documents.\n"
    "You will see (1) an image of a document, and (2) a first-pass OCR "
    "transcription that likely contains errors. Common errors include wrong "
    "dot placement on letters that share the same body shape (for example "
    "ب / ت / ث / پ / ن), dropped or duplicated words, missing regions, and "
    "runaway repetition where the first-pass model got stuck in a loop.\n\n"
    "Examine the image carefully, region by region. Produce a corrected "
    "transcription that preserves the original line structure. If the baseline "
    "is already correct, return it unchanged. Do not invent content that is "
    "not visible in the image, and do not continue a repetition loop present "
    "in the baseline.\n\n"
    "In the reasoning block, name the specific words or characters you changed "
    "and what in the image justifies each change. Do not write generic notes "
    "such as 'the text looks correct'.\n\n"
    "Format your answer exactly as:\n"
    "<reasoning>\nbrief notes on what you changed and why\n</reasoning>\n"
    "<corrected>\nthe full corrected transcription\n</corrected>"
)

USER_TEMPLATE = "First-pass OCR:\n{baseline}"


def _encode_image_bytes(raw: bytes, upscale: int) -> str:
    if upscale > 1:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            raw = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


def _build_prompt(data_url: str, baseline_text: str) -> list[dict]:
    """Build the full prompt, system message included.

    The system message is deliberately built here with list-typed content rather
    than passed to SingleTurnEnv as `system_prompt=`. That path prepends
    `{"role": "system", "content": <str>}` to our user message, whose content is
    a *list* of content parts; the two then land in one Arrow column and
    serialization fails with "cannot mix list and non-list, non-null values".
    Wrapping the system text as a single text part keeps the column uniformly
    list-typed. This applies to any multimodal environment, not just this one.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": USER_TEMPLATE.format(baseline=baseline_text)},
            ],
        },
    ]


# -------- Data loading --------


def _load_baselines(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"baselines cache not found at {path}. Run "
            "`uv run python data_prep/compute_baselines.py` first."
        )
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                out[record["doc_id"]] = record
    return out


def _check_upscale(records: list[dict], image_upscale: int) -> None:
    """The corrector must see exactly the image the baseline was read from.

    If the cached baseline was transcribed from a 4x-upscaled image but the
    environment feeds a 1x image (or vice versa), the CER delta measures image
    preprocessing rather than correction -- reward moves, nothing is learned.
    That is the "reward function bug that looks like learning" from INIT.md
    section 8, so it fails loudly rather than silently.
    """
    cached = {r.get("low_upscale", 1) for r in records}
    if cached and cached != {image_upscale}:
        raise ValueError(
            f"image_upscale={image_upscale} does not match the baselines cache "
            f"(low_upscale={sorted(cached)}). The corrector would see a different "
            "image than the baseline was read from, making the CER delta meaningless. "
            "Recompute baselines with --low-upscale, or pass a matching image_upscale."
        )


def _rows_from_local(
    dataset_path: Path, baselines_path: Path, split: str, image_upscale: int
) -> list[dict]:
    raw = load_from_disk(str(dataset_path))[split]
    baselines = _load_baselines(baselines_path)
    _check_upscale(list(baselines.values()), image_upscale)
    rows = []
    for ex in raw:
        b = baselines.get(ex["doc_id"])
        if b is None:
            continue
        rows.append(
            {
                "doc_id": ex["doc_id"],
                "image_bytes": Path(ex["low_res_path"]).read_bytes(),
                "ground_truth": ex["ground_truth"],
                "language": ex["language"],
                "baseline_low": b.get("baseline_low", ""),
                "baseline_high": b.get("baseline_high", ""),
            }
        )
    return rows


def _rows_from_hub(
    dataset_id: str, split: str, revision: str | None, image_upscale: int
) -> list[dict]:
    raw = load_dataset(dataset_id, split=split, revision=revision)
    _check_upscale([{"low_upscale": r} for r in set(raw["low_upscale"])], image_upscale)
    rows = []
    for ex in raw:
        image = ex["image"]
        if isinstance(image, dict):  # datasets Image feature, decode disabled
            raw_bytes = image["bytes"]
        elif isinstance(image, (bytes, bytearray)):
            raw_bytes = bytes(image)
        else:  # PIL image
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        rows.append(
            {
                "doc_id": ex["doc_id"],
                "image_bytes": raw_bytes,
                "ground_truth": ex["ground_truth"],
                "language": ex.get("language", "unknown"),
                "baseline_low": ex.get("baseline_low", ""),
                "baseline_high": ex.get("baseline_high", ""),
            }
        )
    return rows


def _to_dataset(
    rows: list[dict],
    *,
    upscale: int,
    min_baseline_cer: float,
    max_baseline_cer: float,
    max_examples: int,
) -> Dataset:
    records = []
    for r in rows:
        baseline_cer = cer(normalize(r["ground_truth"]), normalize(r["baseline_low"]))
        if not (min_baseline_cer <= baseline_cer <= max_baseline_cer):
            continue
        records.append(
            {
                "prompt": _build_prompt(
                    _encode_image_bytes(r["image_bytes"], upscale), r["baseline_low"]
                ),
                "answer": r["ground_truth"],
                "info": {
                    "doc_id": r["doc_id"],
                    "language": r["language"],
                    "baseline_low": r["baseline_low"],
                    "baseline_high": r["baseline_high"],
                },
            }
        )
        if 0 < max_examples <= len(records):
            break
    if not records:
        raise ValueError(
            "no examples survived the baseline-CER band "
            f"[{min_baseline_cer}, {max_baseline_cer}] -- widen it or check the baselines cache"
        )
    return Dataset.from_list(records)


# -------- Scoring --------


def _scores(completion, answer: str, info: dict[str, Any], state: dict, parser) -> dict:
    """Compute every CER once per rollout and cache it on the shared state."""
    cached = state.get("_ocr_scores")
    if cached is not None:
        return cached

    corrected = parser.parse_answer(completion)
    gt = normalize(answer)
    baseline = normalize(info.get("baseline_low", ""))
    ceiling_hyp = normalize(info.get("baseline_high", ""))
    out = normalize(corrected) if corrected is not None else ""

    scores = {
        "parsed": corrected is not None,
        "corrected": corrected or "",
        "gt_len": len(gt),
        "baseline_len": len(baseline),
        "out_len": len(out),
        "baseline_cer": cer(gt, baseline),
        "corrected_cer": cer(gt, out) if corrected is not None else CER_CAP,
        "ceiling_cer": cer(gt, ceiling_hyp),
        "copied": bool(corrected is not None and out == baseline and out != ""),
    }
    state["_ocr_scores"] = scores
    return scores


def make_rubric(
    *,
    format_bonus: float,
    length_penalty: float,
    length_min_ratio: float,
    length_max_ratio: float,
    cer_scale: float,
    improvement_bonus: float,
    parser: vf.Parser,
) -> vf.Rubric:
    async def format_reward(completion, answer, info, state, parser) -> float:
        return format_bonus if _scores(completion, answer, info, state, parser)["parsed"] else 0.0

    async def length_guardrail(completion, answer, info, state, parser) -> float:
        """Zero out rollouts with pathological length. Guardrail, not a teacher."""
        s = _scores(completion, answer, info, state, parser)
        if not s["parsed"] or not s["baseline_len"]:
            return 0.0
        ratio = s["out_len"] / s["baseline_len"]
        return 0.0 if length_min_ratio <= ratio <= length_max_ratio else length_penalty

    async def cer_improvement(completion, answer, info, state, parser) -> float:
        """Main signal: CER(baseline, GT) - CER(corrected, GT), scaled."""
        s = _scores(completion, answer, info, state, parser)
        if not s["parsed"]:
            return 0.0
        delta = s["baseline_cer"] - s["corrected_cer"]
        bonus = improvement_bonus if delta > 0 else 0.0
        return cer_scale * delta + bonus

    # -------- Metrics: logged, zero weight, never affect the gradient --------

    async def m_baseline_cer(completion, answer, info, state, parser) -> float:
        return _scores(completion, answer, info, state, parser)["baseline_cer"]

    async def m_corrected_cer(completion, answer, info, state, parser) -> float:
        return _scores(completion, answer, info, state, parser)["corrected_cer"]

    async def m_ceiling_cer(completion, answer, info, state, parser) -> float:
        """CER of the high-res first pass vs GT: the achievability ceiling."""
        return _scores(completion, answer, info, state, parser)["ceiling_cer"]

    async def m_copied_baseline(completion, answer, info, state, parser) -> float:
        """1.0 when the output is the baseline verbatim. Watch this: a climb
        toward 1.0 is the trivial-copy convergence from INIT.md section 8."""
        return 1.0 if _scores(completion, answer, info, state, parser)["copied"] else 0.0

    async def m_length_ratio(completion, answer, info, state, parser) -> float:
        """Output length over baseline length. Subtler length gaming shows up
        here as a distribution drifting below 1.0 well before the guardrail fires."""
        s = _scores(completion, answer, info, state, parser)
        return s["out_len"] / s["baseline_len"] if s["baseline_len"] else 0.0

    async def m_recovered_fraction(completion, answer, info, state, parser) -> float:
        """Share of the theoretically recoverable error that was recovered:
        (baseline - corrected) / (baseline - ceiling). The headline number from
        INIT.md section 6.9. Zero when there was no headroom to begin with."""
        s = _scores(completion, answer, info, state, parser)
        headroom = s["baseline_cer"] - s["ceiling_cer"]
        if headroom <= 1e-6:
            return 0.0
        return max(-1.0, min(1.0, (s["baseline_cer"] - s["corrected_cer"]) / headroom))

    rubric = vf.Rubric(
        funcs=[format_reward, length_guardrail, cer_improvement],
        weights=[1.0, 1.0, 1.0],
        parser=parser,
    )
    for metric in (
        m_baseline_cer,
        m_corrected_cer,
        m_ceiling_cer,
        m_copied_baseline,
        m_length_ratio,
        m_recovered_fraction,
    ):
        rubric.add_metric(metric)
    return rubric


# -------- Entry point --------


def load_environment(
    dataset_path: str = "hf_dataset",
    baselines_path: str = "baselines.jsonl",
    dataset_id: str | None = None,
    revision: str | None = None,
    train_split: str = "train",
    eval_split: str = "test",
    max_examples: int = -1,
    max_eval_examples: int = -1,
    image_upscale: int = 4,
    min_baseline_cer: float = 0.0,
    max_baseline_cer: float = 1.0,
    format_bonus: float = 0.1,
    length_penalty: float = -0.5,
    length_min_ratio: float = 0.5,
    length_max_ratio: float = 2.0,
    cer_scale: float = 10.0,
    improvement_bonus: float = 0.05,
    max_tokens: int = 3072,
    enable_thinking: bool = False,
) -> vf.Environment:
    """Load the OCR corrector environment.

    Args:
        dataset_path: on-disk dataset from `data_prep/build_dataset.py`.
        baselines_path: JSONL cache from `data_prep/compute_baselines.py`.
        dataset_id: HF Hub dataset with images/baselines embedded. When set, it
            takes precedence over `dataset_path`/`baselines_path`.
        image_upscale: integer factor to enlarge the low-res image before
            encoding. Adds no information, but gives the vision encoder more
            patches on these very small captures.
        min_baseline_cer / max_baseline_cer: keep only documents whose baseline
            CER falls in this band. Restratification lever from INIT.md 6.4 --
            groups with no headroom produce zero reward variance under GRPO.
    """
    parser = vf.XMLParser(fields=["reasoning", "corrected"], answer_field="corrected")

    def rows(split: str) -> list[dict]:
        if dataset_id:
            return _rows_from_hub(dataset_id, split, revision, image_upscale)
        return _rows_from_local(
            Path(dataset_path), Path(baselines_path), split, image_upscale
        )

    band = {
        "upscale": image_upscale,
        "min_baseline_cer": min_baseline_cer,
        "max_baseline_cer": max_baseline_cer,
    }
    dataset = _to_dataset(rows(train_split), max_examples=max_examples, **band)
    eval_dataset = _to_dataset(rows(eval_split), max_examples=max_eval_examples, **band)

    # Bound generation at the environment level rather than relying on the
    # caller. Qwen3.5's native thinking mode is a hazard on this task: left on,
    # the model burns tens of thousands of tokens in a "Wait, `پاک مقصد`."
    # repetition loop and never emits a <corrected> block at all -- 5 of 10
    # rollouts in the first smoke eval produced no parseable output for exactly
    # this reason, at ~40k output tokens each. The prompt already asks for an
    # explicit <reasoning> block, which is the visible, bounded substitute, and
    # it is how the cached baselines were generated. Set enable_thinking=True to
    # revisit once the model is stronger at the task.
    #
    # Caveat on max_tokens: Environment.evaluate shallow-updates these with the
    # caller's sampling args, and `prime eval run` always passes an explicit
    # `max_tokens` (None when -t is omitted), which clobbers the value below.
    # `extra_body` survives because the CLI does not set that key. So the
    # thinking switch is enforceable from here but the token cap is not --
    # **pass `-t 3072` to `prime eval run`**. Training is unaffected: the TOML
    # configs set `[sampling] max_tokens` explicitly.
    sampling_args: dict = {"max_tokens": max_tokens}
    if not enable_thinking:
        sampling_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    # No system_prompt= here on purpose -- see _build_prompt.
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        sampling_args=sampling_args,
        parser=parser,
        rubric=make_rubric(
            format_bonus=format_bonus,
            length_penalty=length_penalty,
            length_min_ratio=length_min_ratio,
            length_max_ratio=length_max_ratio,
            cer_scale=cer_scale,
            improvement_bonus=improvement_bonus,
            parser=parser,
        ),
    )
