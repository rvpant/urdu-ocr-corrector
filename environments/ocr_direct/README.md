# ocr-direct

### Overview
- **Environment ID**: `ocr-direct`
- **Short description**: Transcribe a low-resolution Urdu newspaper image. Scored by character error rate against a human transcription.
- **Tags**: single-turn, multimodal, ocr, urdu, train, eval

### Why this exists alongside `ocr-corrector`

`ocr-corrector` frames the task as *improve a first-pass draft*. That framing came from INIT.md §3.1, which chose it because hosted training was RL-only at the time and a correction task is more RL-shaped than end-to-end OCR. Two measurements undermined it:

| Evidence | Result |
| --- | --- |
| pass@16 on the correction task | best-of-16 CER **0.259** vs baseline **0.262** — captures **2.1%** of available headroom |
| Rollouts producing parseable output | **66%** — a third died in `Wait, ...` reasoning loops |
| Frontier VLM on the same low-res images | **0.022** CER — the information is in the pixels |

GRPO amplifies behaviour the policy already samples. Putting a draft in the context anchors every rollout to the draft's errors, collapsing within-group variance to the third decimal place. This environment removes the draft, the reasoning block, and the XML tags — the prompt is the same shape that produced `baselines.jsonl` with zero failures across 1,658 calls.

### Task
- **Type**: single-turn, multimodal (one `image_url` content part + text)
- **Output format**: raw transcription, no tags. Markdown fences and `Here is the transcription:` preambles are stripped before scoring — charging CER for those would measure instruction-following, not OCR.

#### Reward

| Function | Range | Purpose |
| --- | --- | --- |
| `transcription_accuracy` | `[0, 1]` | `1 − CER` against the human transcription |
| `length_guardrail` | `0` or `−0.5` | Fires outside a `[0.4, 2.0]` output/ground-truth length ratio |

Absolute accuracy, not a delta against a draft. CER is capped at 1.0 before use: `jiwer.cer` is unbounded above via insertions, and a repetition-looped rollout scores 4.0+ on this corpus, which would dominate its group's advantage.

**The reward is always computed against human ground truth**, including when the model is being distilled from a teacher via `loss = "sft"`. The teacher supplies the imitation target; the reward supplies the verification. Those stay separate on purpose.

### Metrics

| Metric | Meaning |
| --- | --- |
| `m_cer` | Character error rate — the headline number |
| `m_length_ratio` | Output length ÷ ground-truth length; catches subtle truncation |
| `m_empty` | Share of rollouts producing nothing usable. Was 34% on the correction task |

### Reference points (Qwen3.5-4B, low-res at 4× upscale, 10 documents)

| | CER |
| --- | --- |
| Qwen3.5-4B, greedy | 0.251 |
| Qwen3.5-9B, greedy | 0.212 |
| gemini-3-flash | **0.022** |

The gap between 0.251 and 0.022 is the target. It is a capability gap, not a data-quality limit.

### Quickstart

```bash
prime env install ocr-direct
prime eval run ocr-direct -m Qwen/Qwen3.5-4B -n 5 -r 2 -t 3072
```

The `-t 3072` is required. `Environment.evaluate` shallow-updates the environment's sampling args with the caller's, and `prime eval run` always passes an explicit `max_tokens` — `None` when `-t` is omitted — which overwrites the value set in `load_environment`. Without it a single rollout can reach 60k output tokens. Hosted training is unaffected; the TOML configs set `[sampling] max_tokens`.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | `"hf_dataset"` | On-disk dataset from `data_prep/build_dataset.py` |
| `dataset_id` | str | `None` | HF Hub dataset with images embedded. **Required for hosted training** |
| `revision` | str | `None` | Hub revision pin |
| `train_split` / `eval_split` | str | `"train"` / `"test"` | Split names |
| `max_examples` / `max_eval_examples` | int | `-1` | Cap dataset size |
| `image_upscale` | int | `4` | Enlarge before encoding. Measured knee: CER 3.006 → 0.251 |
| `length_penalty` | float | `-0.5` | Penalty when the guardrail fires |
| `length_min_ratio` / `length_max_ratio` | float | `0.4` / `2.0` | Acceptable length band |
| `max_tokens` | int | `3072` | Generation cap; see the note above for evals |
| `enable_thinking` | bool | `False` | Qwen3.5 native thinking. Left on, this model burns 40k+ tokens on single glyphs |

Unlike `ocr-corrector`, this environment needs no `baselines.jsonl` and imposes no upscale-consistency constraint — there is no cached draft whose preprocessing must match.

### Required Environment Variables

None.
