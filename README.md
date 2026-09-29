# ocr-corrector

Can post-training get decent Urdu OCR out of a **small open-weights VLM**? The
corpus is the Urdu Newspaper Benchmark (829 paragraph images, 9,982
ground-truth-aligned lines, [arXiv:2505.13943](https://arxiv.org/abs/2505.13943)),
which gives published numbers to measure against:

| system | CER |
|---|---:|
| TrOCR — best specialist baseline | **0.159 ← the bar** |
| Qwen3.5-4B zero-shot, high-res | 0.168 |
| Gemini-2.5-Pro zero-shot | 0.032 |

## Current state

**RL is a closed negative result.** Three GRPO runs on Prime hosted, two model
sizes, four reward configurations — all degraded the policy, including one
starting from a competent 35B. The cause is an advantage signal below the
reward's own noise floor: advantage sd at step 1 of the 35B run was 0.032, which
`data_prep/probe_passk.py` predicted as 0.028 beforehand for ~$2. **Run that
probe before committing to any RL run.** Full analysis in `RL_report_v1.md`
(what happened) and `RL_review_v1.md` (why, versus the literature) — both local,
not committed.

**Every one of those runs used the wrong images.** The corpus ships a 4×-larger
capture per document. The table under *Decisions that departed from INIT.md*
below measured this on day one, but `package_dataset.py` embedded low-res only —
correct for the original correction task, and never revisited when the task
became direct OCR. Paired over all 829 documents, same model and prompt:

| input | CER | sub | ins |
|---|---:|---:|---:|
| low-res, 4× upscale | 0.332 | 0.155 | 0.240 |
| **high-res, 2× upscale** | **0.168** | **0.068** | **0.117** |

Better on 96.1% of documents, substitutions halved. A 4B model on high-res
matches a 35B model on low-res.

**Next step is SFT, and it has never been run.** `notebooks/sft_then_rl.ipynb`
(Colab A100): high-res input, vision tower unfrozen, gated on a pass@k
re-measurement before any RL stage. Model-loading, collator, and trainer cells
have not executed on a GPU — §7a has a 2-step smoke test for exactly that.

## Two task shapes were built

| environment | prompt | status |
|---|---|---|
| `ocr_corrector` | image + frozen first-pass draft → correction | abandoned pre-training; the draft anchors rollouts, within-group spread 0.019 |
| `ocr_direct` | image → transcription | what was actually trained |

`INIT.md` is the original brief and describes the correction framing. Most of
this file below documents that phase; it is accurate as history.

## Pipeline

```bash
# 1. Raw folders -> HF dataset on disk (train=746, test=83)
uv run python data_prep/build_dataset.py

# 2. Frozen first pass, cached. Resumable; safe to interrupt. ~2h, ~$1.
uv run python data_prep/compute_baselines.py

# 3. Go/no-go before spending training compute
uv run python data_prep/audit_baselines.py --split train

# 4. Install the environment and smoke-test it
prime env install ocr-corrector
prime eval run ocr-corrector -m Qwen/Qwen3.5-4B -n 5 -r 2 -t 3072
prime eval tui

# 5. Package a self-contained dataset for hosted training, and push both
uv run python data_prep/package_dataset.py --push-to-hub <hf-user>/ocr-corrector-ur
prime env push --path ./environments/ocr_corrector

# 6. Train
prime train configs/lab/smoke.toml      # 50 steps, plumbing + first signal
prime train configs/lab/medium.toml     # 300 steps, only after smoke looks right
```

### Current path (direct OCR, high-res, SFT)

Steps 1–3 above still apply. From there:

```bash
# High-res dataset + the three-way disjoint split, same seed as the low-res
# splits so document assignment stays identical and results stay comparable.
uv run python data_prep/package_dataset.py --image-source high --out packaged_dataset_hr
uv run python data_prep/make_splits.py --dataset packaged_dataset_hr \
    --out splits_dataset_hr --push-to-hub <hf-user>/ocr-urdu-splits-hr --private

# Before any RL run, always. ~$2, 30 min, and it predicted all three failures.
uv run python data_prep/probe_passk.py --direct

# SFT: open notebooks/sft_then_rl.ipynb in Colab (A100), run sections 1-8.
uv run python notebooks/build_notebook.py   # regenerate after editing the builder
```

Edit `notebooks/build_notebook.py`, not the `.ipynb` — the notebook is generated.

The `-t 3072` on the eval is **not optional** — see the note under Verification
status. Configs are wired to the `rpant/` namespace on both Prime and HF.

## Layout

```
data_prep/
  build_dataset.py           raw folders -> hf_dataset/
  compute_baselines.py       frozen first-pass OCR -> baselines.jsonl
  audit_baselines.py         CER distribution, data-quality flags, go/no-go
  package_dataset.py         self-contained dataset for hosted training
  probe_baseline_models.py   measure candidate models / preprocessing before committing
  make_splits.py             three-way disjoint split (sft_train/rl_train/test)
  probe_passk.py             THE pre-flight diagnostic -- run before any RL
environments/ocr_corrector/  image + draft -> correction (abandoned)
environments/ocr_direct/     image -> transcription (what was trained)
configs/lab/                 hosted-training configs; direct_rl50_35b.toml was the decisive run
notebooks/build_notebook.py  generator -- edit this
notebooks/sft_then_rl.ipynb  generated; the SFT path
hf_dataset/                  built artifact (gitignored)
baselines.jsonl              built artifact (gitignored)
data/                        source corpus, not vendored (gitignored)
```

Hub artifacts: `rpant/ocr-direct`, `rpant/ocr-corrector` (environments);
`rpant/ocr-corrector-ur`, `rpant/ocr-urdu-splits`, `rpant/ocr-urdu-splits-hr`
(datasets, private).

## What the data turned out to be

829 Urdu newspaper clippings, each with a low-res capture, a high-res capture at
4× the linear resolution, and a verified transcription. Ground truth runs 175 to
2465 characters, median 454.

The layout is `data/test/{lr-images,hr-images,groundtruth}/{i}.{png,txt}` with
bare integer ids — not the `data/{high_res,low_res,ground_truth}/{i}` with
zero-padded ids assumed in INIT.md §4. Only `build_dataset.py` cares.

The corpus is **entirely Urdu**; there is no Persian to stratify against. Every
one of the 829 documents contains Urdu-specific letters. INIT.md §6.2's language
heuristic labelled ~6% of them Persian because it counted پ چ ژ گ as Persian
evidence — but Urdu's letter inventory is a superset of Persian's and uses all
four. Only ٹ ڈ ڑ ں ے ھ are one-directional evidence, and their absence is weak,
so `infer_language` now returns `ur` or `unknown` and never claims Persian. The
`language` field stays in the schema for future data; language-stratified
metrics are currently a no-op.

## Decisions that departed from INIT.md

**Baseline and corrector model: Qwen3.5-4B, not 9B.** `prime rl models` lists
Qwen3.5-9B as available for hosted training, but it does not respond on Prime
Inference — five attempts, read timeouts at 45s/95s/100s/150s and a 232s median
across a 12-call probe. Qwen3.5-122B-A10B behaves the same way. The frozen first
pass has to run on inference, so 9B is unusable for this pipeline today. 4B is
responsive there *and* on the hosted-training list, which preserves INIT.md
§3.2's one-model-two-roles property. Re-probe with
`data_prep/probe_baseline_models.py` if 9B comes back.

**Images are upscaled 4× before the model sees them.** This was not in the
brief and it is the single largest quality lever found. The captures are tiny —
a full newspaper clipping can be 129×130 px — and at native size the vision
encoder spends ~100 tokens on the entire page, which drives small models into
repetition loops. Measured with Qwen3.5-4B:

| Input | Prompt tokens | Median CER |
| --- | --- | --- |
| low-res, native | 104 | 3.006 |
| low-res, 4× | 386 | 0.251 |
| low-res, 6× | 834 | 0.242 |
| high-res, native | 386 | 0.208 |
| high-res, 2× | 1459 | 0.133 |

Upscaling adds no information — it only buys the encoder more patches — so this
is not the model cheating its way to the high-res twin. 4× is the knee: it lands
the baseline inside the `[0.05, 0.30]` band where RL has signal, at 6× the token
cost of native, while 6× doubles tokens again for 0.009 CER. The ceiling
reference uses high-res at 2×.

Because of this, `image_upscale` in the environment **must** match
`--low-upscale` in `compute_baselines.py`. The corrector has to see exactly the
image its baseline was read from, or the CER delta measures preprocessing rather
than correction — INIT.md §8's "reward function bug that looks like learning".
`load_environment` compares the two and raises on a mismatch.

**`[buffer] online_difficulty_filtering` does not exist any more.** INIT.md §6.7
specifies it, but the CLI's `RLConfig` sets `extra="forbid"` and has no `buffer`
field — the run would fail validation before starting. Its documented
replacement is a `zero_advantage` pre-batch filter, which `medium.toml` uses.
(The stock `configs/rl/qwen-3-5-moe-advanced.toml` template still carries the
old key and would fail the same way.)

**`vf.Rubric` has no `metrics=` kwarg.** INIT.md §6.5 flagged this as a guess.
In verifiers 0.1.14 the API is `rubric.add_metric(fn)`, shorthand for
`add_reward_func(fn, weight=0)`.

**`prime rl run` is deprecated** in favour of `prime train <config>`.

**CER is capped at 1.0 before differencing.** jiwer's CER is unbounded above via
insertions; a repetition-looped rollout scores 4.0+ on this corpus. Uncapped, a
single blown-up rollout swamps the group advantage under GRPO.

## Verification status

| Check | Result |
| --- | --- |
| Multimodal `image_url` content blocks | Confirmed end-to-end against Prime Inference and `verifiers/types.py` |
| Prime Inference base URL / auth | `https://api.pinference.ai/api/v1`, `PRIME_API_KEY` (falls back to `~/.prime/config.json`) |
| Reward functions on synthetic rollouts | Copy-baseline scores 0.1, perfect 0.576, repetition −9.97; unparseable and `content: None` score 0.0 without crashing |
| Training configs | Both validate against the live `RLConfig` schema |
| Audit go/no-go | Passes — see below |
| Live smoke eval | Passes after three fixes — see below |

### Audit result (train split, 746 docs)

```
low-res  baseline CER   median 0.276   p25 0.237   p75 0.330
high-res baseline CER   median 0.145   p25 0.115   p75 0.179
headroom (low - high)   median 0.134
in [0.05, 0.30] band    475/746 = 63.7%     (INIT.md 6.4 wants >30%)
high-res CER > 0.30     31/746  =  4.2%     (INIT.md 6.4 wants <5%)
no headroom (<=0.01)    32/746  =  4.3%
repetition-looped       20/746  =  2.7%
```

Both gates pass, with real recoverable error between the baseline and the
ceiling.

### Smoke eval result (test split, 5 docs × 2 rollouts)

`m_baseline_cer` 0.252 and `m_ceiling_cer` 0.134 both reproduce the audit, so
the environment reads what the audit measured. 8/10 rollouts parse,
`m_copied_baseline` is 0.000, output is bounded at ~1,443 tokens.

`m_corrected_cer` is **0.440 against a 0.252 baseline**: the untrained model
actively corrupts text rather than merely failing to improve it. Reward is
−0.328 with std 0.442 — negative mean, but healthy within-group variance, which
is what GRPO needs.

Three fixes were required to get here, all invisible to offline testing:

1. `system_prompt=` on a multimodal env produces an Arrow schema crash (string
   content mixed with list content in one column). The system message is now
   built with list-typed content inside `_build_prompt`.
2. Qwen3.5's thinking mode ran away to 40k+ output tokens per rollout, one
   completion reaching 148,648 characters of `Wait, ...`. Disabled via
   `chat_template_kwargs`.
3. **`max_tokens` cannot be enforced from `load_environment`.** `Environment.evaluate`
   shallow-updates the env's sampling args with the caller's, and `prime eval run`
   always passes an explicit `max_tokens` — `None` when `-t` is omitted. Always
   pass `-t 3072`. Training is unaffected: the TOMLs set `[sampling] max_tokens`.

Fix 2 mattered financially: at 40k output tokens per rollout the 300-step run
would have cost ~$1,600–3,000 against a $149 balance.

## Reward hacking watchlist

INIT.md §8 is not optional. The environment logs six zero-weight metrics;
`m_copied_baseline` and `m_length_ratio` are the two to watch.

The smoke eval changed the reading of this. Copying the baseline earns +0.1
while the untrained policy scores −0.328, so copying is not a trap the model
falls into — it is the first real improvement available and GRPO should find it
fast. Treat a rising `m_copied_baseline` as a milestone (learning not to
corrupt), and a *plateau at 1.0* as the failure. Adding a copy penalty before
that plateau appears would punish the model for making genuine progress.

Levers once it does plateau, in order: raise `improvement_bonus`, add an
explicit copy penalty, lower `cer_scale`.
