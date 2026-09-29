# ocr-corrector

### Overview
- **Environment ID**: `ocr-corrector`
- **Short description**: Given a low-resolution Urdu/Persian document image and a frozen first-pass OCR transcription, produce a corrected transcription. Rewarded on character-error-rate improvement over that baseline.
- **Tags**: single-turn, multimodal, ocr, urdu, persian, train, eval

### Datasets
- **Primary dataset**: 829 Urdu newspaper clippings, each with a low-res capture, a high-res capture of the same page, and a verified transcription. Split 90/10 (`train=746`, `test=83`, seed 42).
- **Built by**: `data_prep/build_dataset.py` → `hf_dataset/`, plus `data_prep/compute_baselines.py` → `baselines.jsonl`.
- **Portable variant**: `data_prep/package_dataset.py` embeds the low-res image bytes and both baselines into one dataset for the HF Hub. Hosted training needs this — the training container cannot see this workspace's `data/` folder.

Only the **low-res** image is ever shown to the model. That matches the production distribution: real captures do not come with a high-res twin. The high-res image is used offline only, to produce the achievability-ceiling reference.

### Task
- **Type**: single-turn, multimodal (one `image_url` content part + text)
- **Output format**: `<reasoning>...</reasoning>` followed by `<corrected>...</corrected>`, parsed with `vf.XMLParser`
- **Rubric**: three weighted reward functions plus six zero-weight metrics

#### Reward functions (weight 1.0 each)

| Function | Range | Purpose |
| --- | --- | --- |
| `format_reward` | `0` or `+0.1` | Output carries a parseable `<corrected>` block |
| `length_guardrail` | `0` or `-0.5` | Fires when output/baseline length ratio leaves `[0.5, 2.0]` |
| `cer_improvement` | `cer_scale × Δ + bonus` | Main signal: `CER(baseline) − CER(corrected)`, plus `+0.05` when the correction actually helped |

CER is capped at 1.0 before differencing. jiwer's CER is unbounded above via insertions, and an untrained model stuck in a repetition loop scores 4.0+ on this corpus — uncapped, one blown-up rollout swamps the whole GRPO group advantage.

### Quickstart

```bash
# 1. Build the dataset and the frozen first pass (once)
uv run python data_prep/build_dataset.py
uv run python data_prep/compute_baselines.py
uv run python data_prep/audit_baselines.py     # go/no-go before spending compute

# 2. Install and smoke-test
prime env install ocr-corrector
prime eval run ocr-corrector -m Qwen/Qwen3.5-4B -n 5 -r 2 -t 3072
prime eval tui
```

`prime eval run` evaluates the `test` split by default (it is wired as `eval_dataset`).

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | `"hf_dataset"` | On-disk dataset from `build_dataset.py` |
| `baselines_path` | str | `"baselines.jsonl"` | Frozen first-pass cache |
| `dataset_id` | str | `None` | HF Hub dataset with images/baselines embedded. Takes precedence over the two paths above. **Required for hosted training.** |
| `revision` | str | `None` | Hub revision pin |
| `train_split` / `eval_split` | str | `"train"` / `"test"` | Split names |
| `max_examples` / `max_eval_examples` | int | `-1` | Cap dataset size |
| `image_upscale` | int | `4` | Enlarge the low-res image before encoding |
| `min_baseline_cer` / `max_baseline_cer` | float | `0.0` / `1.0` | Keep only documents whose baseline CER falls in this band |
| `format_bonus` | float | `0.1` | Reward for a parseable output |
| `length_penalty` | float | `-0.5` | Penalty when the length guardrail fires |
| `length_min_ratio` / `length_max_ratio` | float | `0.5` / `2.0` | Acceptable output/baseline length band |
| `cer_scale` | float | `10.0` | Multiplier on the CER delta |
| `improvement_bonus` | float | `0.05` | Flat bonus when the correction improved CER |
| `max_tokens` | int | `3072` | Generation cap. **Not enforceable from here for evals** — see below |
| `enable_thinking` | bool | `False` | Qwen3.5 native thinking. Off by default; left on it loops to 40k+ tokens |

#### On `max_tokens` — pass `-t 3072` to `prime eval run`

`Environment.evaluate` shallow-updates the environment's `sampling_args` with the
caller's, and `prime eval run` always passes an explicit `max_tokens` (`None`
when `-t` is omitted), which overwrites the value set here. `extra_body` survives
because the CLI does not set that key, which is why `enable_thinking` *is*
enforceable from the environment and the token cap is not.

Without `-t`, a single rollout on this corpus reached 61,949 output tokens.
Hosted training is unaffected — the TOML configs set `[sampling] max_tokens`
explicitly.

#### On `image_upscale`

These captures are tiny — a full newspaper clipping can be 129×130 px — and at native size the vision encoder spends only ~100 tokens on the entire page, which sends small models into repetition loops. Upscaling adds no information; it just buys more patches. Measured on this corpus with Qwen3.5-4B (`data_prep/probe_baseline_models.py`):

| Input | Prompt tokens | Median CER |
| --- | --- | --- |
| low-res, native | 104 | 3.006 |
| low-res, 4× | 386 | 0.251 |
| low-res, 6× | 834 | 0.242 |
| high-res, native | 386 | 0.208 |
| high-res, 2× | 1459 | 0.133 |

4× lands the baseline inside the `[0.05, 0.30]` band where RL has usable signal, against a ~0.13 ceiling — roughly 0.12 CER of recoverable headroom.

`image_upscale` **must** match the `--low-upscale` the baselines were computed with. The corrector has to see exactly the image the baseline was read from, or the CER delta measures preprocessing instead of correction. `load_environment` compares the two and raises rather than training on a meaningless signal.

### Metrics

All zero-weight; they are logged, never differentiated.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted sum of the three reward functions |
| `m_baseline_cer` | CER of the frozen first pass vs ground truth |
| `m_corrected_cer` | CER of the model's output vs ground truth — the headline number |
| `m_ceiling_cer` | CER of the high-res first pass vs ground truth: what is achievable at all |
| `m_copied_baseline` | 1.0 when the output is the baseline verbatim. **Watch this.** A climb toward 1.0 is trivial-copy convergence |
| `m_length_ratio` | Output length ÷ baseline length. Subtle length gaming shows up as drift below 1.0 before the guardrail fires |
| `m_recovered_fraction` | `(baseline − corrected) / (baseline − ceiling)`: share of recoverable error actually recovered |

`m_recovered_fraction` is the number worth reporting. Raw corrected CER conflates "the corrector is good" with "this document was legible in the first place".

### Required Environment Variables

None. The environment reads cached baselines and does no external API calls of its own.
