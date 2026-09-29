"""Generate notebooks/sft_then_rl.ipynb.

The notebook is generated rather than hand-edited so the cell sources stay
readable and reviewable as normal Python here. Re-run after editing:

    uv run python notebooks/build_notebook.py
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "sft_then_rl.ipynb"

cells: list[dict] = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True),
        }
    )


# ---------------------------------------------------------------- intro
md("""
# Urdu OCR — SFT then RL

Trains `Qwen/Qwen3.5-4B` to transcribe Urdu newspaper clippings, in two stages,
and measures whether the RL stage adds anything over SFT alone.

The corpus is the **Urdu Newspaper Benchmark** (829 paragraph images, 9,982
lines) from [*From Press to Pixels*](https://arxiv.org/abs/2505.13943). Published
reference points on this exact test set, so results here are comparable rather
than freestanding:

| system | CER |
|---|---:|
| TrOCR — best specialist baseline | 0.159 |
| **Qwen3.5-4B zero-shot, high-res** | **0.168** |
| Gemini-2.5-Pro zero-shot | 0.032 |

The 4B base model already sits at the specialist baseline. **Beating 0.159 is the
bar for "decent Urdu OCR from a small open model."**

## Two corrections from RL_review_v1.md, both applied here

**Resolution.** Every prior run used the low-res capture, because the original
task framing was *correcting* a low-res first pass. That stopped being the task
and nothing caught it. Over all 829 documents, same model, same prompt:

| input | CER | sub | ins |
|---|---:|---:|---:|
| low-res, 4× upscale | 0.332 | 0.155 | 0.240 |
| **high-res, 2× upscale** | **0.168** | **0.068** | **0.117** |

Better on 96.1% of documents, with substitutions halved — the model reads better,
not merely shorter. This notebook uses `rpant/ocr-urdu-splits-hr`, whose document
assignment is identical to the low-res splits.

**The vision tower trains too** (`lora_vision=True`). For a script the base model
barely saw in pretraining, freezing the encoder is the wrong default; the Manchu
low-resource work tuned both towers and reached CER 0.0219 on real documents.
Language-only is now the ablation, not the baseline.

## Why this design

Prior RL-only runs on Prime hosted **failed twice**, with different reward shapes:

| run | what happened |
|---|---|
| 100-step, cliff length penalty | CER 0.337 → 0.585, length ratio 0.93 → 2.17, then 51% empty rollouts |
| 50-step, smooth penalty + higher cap | CER 0.344 → 0.972, substitution rate 0.166 → 0.227 at matched length |

The diagnosis was not the reward function. `pass@16` on the base policy gave
best-of-16 CER **0.247** against a greedy baseline of **0.262** — a tail worth
2.1% of available headroom. GRPO can only amplify behaviour the policy already
samples, so with a tail that thin the advantage signal is dominated by length
noise and updates random-walk downhill.

SFT has no such precondition: dense per-token targets install behaviour rather
than amplifying it. A frontier VLM reads these same images at **0.022** CER, so
the information is in the pixels and the gap is model capability.

## Three arms, one held-out test set

| arm | training | question |
|---|---|---|
| **A** | SFT on `sft_train` (479) | does SFT move CER at all? |
| **B** | A → **GRPO** on `rl_train` (250) | does RL add value? |
| **C** | A → **more SFT** on `rl_train` (250) | …or would the same data as SFT do just as well? |

**B vs C is the actual experiment.** Comparing A to B alone confounds "RL helped"
with "250 more documents helped". C holds the data budget fixed.

All three are scored on the same 100 test documents, **paired per document**,
which removes between-document variance (CER std across documents is ~0.18, far
larger than the effects we are chasing).

Splits are disjoint by construction — see `data_prep/make_splits.py`, which
asserts non-overlap before pushing.

If you have decided against RL entirely, set `train_splits = ("sft_train",
"rl_train")` in §2 for **729** training documents instead of 479, and stop after
§8. Arms B and C become invalid in that configuration and the notebook refuses to
run them.

## Before the RL stage, there is a gate

Cell 9 re-runs `pass@16` on the SFT'd model and reports **greedy − best-of-16**.
That number was measured to track the thing that actually matters: on the 35B
hosted run it was 0.028 before training, and the advantage standard deviation at
step 1 of that run was 0.032 — agreement within 15%. The advantage spread *is* the
learning signal, so this probe estimates it for a few minutes of generation
instead of a 50-step run.

Below 0.05, RL will fail for the same reason it failed three times already, and
the notebook says so rather than burning an hour finding out.

---
**Runtime:** A100 40GB (Colab Pro). Set Runtime → Change runtime type → A100.
""")

# ---------------------------------------------------------------- install
md("## 1 · Environment")

code("""
# Colab: A100 40GB. Check what we actually got -- Colab silently downgrades.
!nvidia-smi --query-gpu=name,memory.total --format=csv

%pip -q install -U "transformers>=4.57" "trl>=0.24" "peft>=0.17" "datasets>=3.0" \\
    "accelerate>=1.0" "bitsandbytes>=0.44" jiwer pillow matplotlib pandas scipy
print("\\nRestart the runtime if transformers was already imported this session.")
""")

code("""
import os, json, math, time, random, gc, re, unicodedata, io, base64
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np, pandas as pd, torch
from PIL import Image
import matplotlib.pyplot as plt

def seed_all(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
""")

# ---------------------------------------------------------------- config
md("""
## 2 · Configuration

Every knob lives here.

`USE_AUGMENTATION` is **off by default**, and when enabled it applies to **arm A
only** (`sft_train`). Arms B and C both operate on the raw 250 `rl_train`
documents, because B-vs-C is a comparison of *method* at a fixed data budget —
augmenting one side would make it a comparison of dataset size instead.

Augmentation is also deliberately kept out of the RL stage on its own merits: it
adds input variance, which adds reward variance, which degrades the GRPO
advantage estimate. That signal is already the weakest link here.
""")

code('''
CFG = dict(
    # --- data ---
    # HIGH-RES splits. The low-res twin (`rpant/ocr-urdu-splits`) is what every
    # run in RL_report_v1.md used, and it was the wrong input: the corpus ships a
    # 4x-larger capture per document, and packaging only embedded the small one.
    # Paired over all 829 docs, same Qwen3.5-4B, same prompt:
    #     low-res  4x upscale   CER 0.332   sub 0.155   ins 0.240
    #     high-res 2x upscale   CER 0.168   sub 0.068   ins 0.117
    # High-res wins on 96.1% of documents. Substitutions halve, so the model
    # reads better rather than merely writing less. Document assignment is
    # byte-identical to the low-res splits (same seed), so anything already
    # measured stays comparable.
    dataset_id     = "rpant/ocr-urdu-splits-hr",  # sft_train 479 / rl_train 250 / test 100
    image_upscale  = 2,      # the measured high-res operating point
    image_format   = "jpeg", # 468KB PNG -> 133KB JPEG per example, same CER
    image_quality  = 90,

    # Which splits feed arm A. Arms B and C need `rl_train` held back, so the
    # default keeps it out. If you have decided against RL entirely, set this to
    # ("sft_train","rl_train") for 729 training documents instead of 479 -- a 52%
    # increase -- and skip sections 9-12. One-way door for this run only, since
    # the split assignment lives in the dataset and is reproducible.
    train_splits   = ("sft_train",),

    # --- augmentation (OFF by default; ARM A / sft_train ONLY) ---
    USE_AUGMENTATION   = False,
    aug_multiplier     = 2,          # extra augmented copies per source document
    aug_upscale_choices= (3, 4, 5),
    aug_quality_range  = (75, 95),
    aug_rotate_deg     = 1.0,        # +/- degrees; newspapers are near-axis-aligned
    aug_blur_prob      = 0.3,

    # --- model ---
    base_model     = "Qwen/Qwen3.5-4B",
    attn_impl      = "sdpa",
    dtype          = "bfloat16",

    # --- LoRA ---
    lora_r         = 32,
    lora_alpha     = 64,
    lora_dropout   = 0.05,
    # Language-side projection names. The vision tower is added separately at
    # runtime (see `resolve_lora_targets`) because its module names differ across
    # VLM generations and cannot be guessed offline.
    lora_targets   = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    # Train the vision encoder too. RL_report_v1.md filed this as an open
    # question; the low-resource-script literature answers it. The Manchu work
    # (Qwen2.5-VL 3B/7B, LLaMA-3.2-11B) applied PEFT to "both vision and language
    # components" at rank 64 and reached CER 0.0219 on real historical documents.
    # For a script the base model barely saw in pretraining, a frozen encoder is
    # the wrong default -- so language-only is the ablation here, not the baseline.
    lora_vision    = True,

    # --- SFT (arm A) ---
    sft_epochs         = 3,
    sft_lr             = 1e-4,
    sft_batch          = 1,
    sft_grad_accum     = 8,
    sft_warmup_ratio   = 0.03,
    sft_max_seq_len    = 2048,
    sft_eval_every     = 25,     # steps; evaluates on EVAL_SUBSET of test
    sft_eval_subset    = 32,     # keep small -- generation is the slow part

    # --- arm C (more SFT on rl_train) ---
    c_epochs           = 3,
    c_lr               = 5e-5,   # lower: continuing from an already-tuned adapter

    # --- GRPO (arm B) ---
    grpo_steps         = 60,
    grpo_num_generations = 8,    # rollouts per prompt (the GRPO group)
    grpo_lr            = 5e-6,   # deliberately low; prior runs drifted at default
    grpo_batch         = 8,
    grpo_grad_accum    = 4,
    grpo_temperature   = 0.7,
    grpo_max_new       = 1024,
    grpo_beta          = 0.04,   # KL to reference. Prior hosted runs used ~1e-3
                                 # and drifted; this anchors the policy far harder.

    # --- generation / eval ---
    gen_max_new        = 1024,
    passk_k            = 16,
    passk_docs         = 20,

    # --- output ---
    run_dir            = "/content/drive/MyDrive/ocr_sft_rl",
    seed               = 1234,
)

INSTRUCTION = ("Transcribe all text in this document image. "
               "Preserve line breaks. Output only the transcription.")

# Mount Drive so a Colab disconnect does not lose the run.
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("not on Colab or mount skipped:", e)
    CFG["run_dir"] = "./ocr_sft_rl"

RUN = Path(CFG["run_dir"]); (RUN/"metrics").mkdir(parents=True, exist_ok=True)
(RUN/"adapters").mkdir(exist_ok=True); (RUN/"figures").mkdir(exist_ok=True)
(RUN/"config.json").write_text(json.dumps(CFG, indent=2))
print("run dir:", RUN)
''')

# ---------------------------------------------------------------- metrics lib
md("""
## 3 · Metrics

CER is decomposed into **substitutions / deletions / insertions**, all normalised
by reference length. This decomposition is what made the RL failures legible:
over-generation appears *only* in insertions, misreading glyphs *only* in
substitutions. Watching CER alone cannot tell those apart, and they call for
opposite fixes.

Normalization matches the Prime environments exactly (`ocr_direct.py`) so numbers
remain comparable across platforms.
""")

code('''
import jiwer

_NORM = str.maketrans({
    "\\u064A": "\\u06CC",  # Arabic yeh   -> Farsi yeh
    "\\u0649": "\\u06CC",  # alef maksura -> Farsi yeh
    "\\u0643": "\\u06A9",  # Arabic kaf   -> Keheh
    "\\u0640": "",          # tatweel
    "\\u200C": " ",         # ZWNJ
    "\\u200D": "", "\\u200E": "", "\\u200F": "",
})
CER_CAP = 1.5

def normalize(s):
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(_NORM)
    return re.sub(r"\\s+", " ", s).strip()

_FENCE = re.compile(r"^\\s*```[a-zA-Z]*\\s*\\n(.*?)\\n?\\s*```\\s*$", re.DOTALL)
_PREAMBLE = re.compile(r"^\\s*(here (is|are)[^:\\n]*:|the transcription[^:\\n]*:|transcription:)\\s*", re.I)

def clean_output(text):
    if not text: return ""
    m = _FENCE.match(text)
    if m: text = m.group(1)
    return _PREAMBLE.sub("", text).strip()

def cer_breakdown(ref, hyp):
    """Capped CER + sub/del/ins rates. sub+del+ins == uncapped CER."""
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:  return dict(cer=0.0 if not hyp else CER_CAP, sub=0., dele=0., ins=0., ratio=0.)
    if not hyp:  return dict(cer=CER_CAP, sub=0., dele=1., ins=0., ratio=0.)
    o = jiwer.process_characters(ref, hyp); n = max(1, len(ref))
    return dict(cer=min(float(o.cer), CER_CAP), sub=o.substitutions/n,
                dele=o.deletions/n, ins=o.insertions/n, ratio=len(hyp)/n)

def summarize(records):
    # Bracket access throughout: `d.sub` resolves to DataFrame.sub (subtraction),
    # not the column, and silently returns a bound method. Same trap waits on
    # `mean`, `std`, `min`, `max`.
    d = pd.DataFrame(records)
    return dict(n=len(d), cer=d["cer"].mean(), cer_median=d["cer"].median(),
                sub=d["sub"].mean(), dele=d["dele"].mean(), ins=d["ins"].mean(),
                ratio=d["ratio"].mean(), empty=float((d["ratio"] == 0).mean()))

def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)); print("wrote", path)

def jsonl_append(obj, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\\n")

# sanity-check the metric on synthetic cases before trusting it on real output
_gt = "\\u06c1\\u0645\\u0627\\u0631\\u06d2 \\u062f\\u0641\\u0627\\u062a\\u0631 \\u0627\\u0648\\u0631"
for name, hyp in [("perfect", _gt), ("2x", _gt+" "+_gt), ("empty", "")]:
    print(f"  {name:<8}", {k: round(v,3) for k,v in cer_breakdown(_gt, hyp).items()})
''')

# ---------------------------------------------------------------- data
md("""
## 4 · Data

Splits load straight from the Hub. `sft_train`, `rl_train` and `test` are
guaranteed disjoint — re-asserted here rather than assumed, because contamination
between these three is the one error that would invalidate every number below.
""")

code('''
from datasets import load_dataset

SPLITS = {s: load_dataset(CFG["dataset_id"], split=s) for s in ["sft_train","rl_train","test"]}
ids = {k: set(v["doc_id"]) for k,v in SPLITS.items()}
for a in ids:
    for b in ids:
        assert not (a < b and ids[a] & ids[b]), f"CONTAMINATION: {a} n {b}"
print({k: len(v) for k,v in SPLITS.items()}, "| disjoint OK")

# Confirm we are on the high-res capture. Pixel dimensions cannot settle this --
# the two sources overlap in size and upscaling erases the difference -- so read
# the column the packaging step recorded.
src = set(SPLITS["sft_train"]["image_source"])
assert src == {"high"}, f"expected high-res images, got {src}; check CFG['dataset_id']"
w, h = SPLITS["sft_train"][0]["image"].size
print(f"image_source={src.pop()}  native size {w}x{h}  -> {w*CFG['image_upscale']}x{h*CFG['image_upscale']} after upscale")

gt_lens = [len(x) for x in SPLITS["sft_train"]["ground_truth"]]
print(f"ground-truth chars: min {min(gt_lens)} median {int(np.median(gt_lens))} "
      f"p90 {int(np.percentile(gt_lens,90))} max {max(gt_lens)}")
''')

code('''
def encode_image(img: Image.Image, upscale=None, quality=None, fmt=None, augment=False):
    """Upscale + re-encode. Returns a PIL image (the processor wants PIL, not a data URL)."""
    upscale = upscale or CFG["image_upscale"]
    quality = quality or CFG["image_quality"]
    fmt     = fmt or CFG["image_format"]
    im = img.convert("RGB")
    if augment:
        upscale = random.choice(CFG["aug_upscale_choices"])
        quality = random.randint(*CFG["aug_quality_range"])
        if CFG["aug_rotate_deg"]:
            im = im.rotate(random.uniform(-CFG["aug_rotate_deg"], CFG["aug_rotate_deg"]),
                           resample=Image.BILINEAR, fillcolor=(255,255,255))
        if random.random() < CFG["aug_blur_prob"]:
            from PIL import ImageFilter
            im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))
    if upscale > 1:
        im = im.resize((im.width*upscale, im.height*upscale), Image.LANCZOS)
    # round-trip through the target codec so training sees the same artifacts as eval
    buf = io.BytesIO()
    im.save(buf, format="JPEG" if fmt in ("jpeg","jpg") else fmt.upper(), quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def build_examples(split_name, augment=False, multiplier=1):
    """-> list of {image, target, doc_id}. Augmented copies are appended, never
    replacements, so the original clean view is always present.

    `split_name` accepts one split name or an iterable of them; multiple splits
    are concatenated in the order given.
    """
    names = [split_name] if isinstance(split_name, str) else list(split_name)
    rows = [ex for n in names for ex in SPLITS[n]]
    out = []
    for ex in rows:
        out.append(dict(image=encode_image(ex["image"]), target=ex["ground_truth"], doc_id=ex["doc_id"]))
    if augment:
        for _ in range(max(0, multiplier-1)):
            for ex in rows:
                out.append(dict(image=encode_image(ex["image"], augment=True),
                                target=ex["ground_truth"], doc_id=ex["doc_id"]+"_aug"))
    return out

t0=time.time()
SFT_EXAMPLES = build_examples(CFG["train_splits"], augment=CFG["USE_AUGMENTATION"],
                              multiplier=CFG["aug_multiplier"])
print(f"SFT examples: {len(SFT_EXAMPLES)} from splits {tuple(CFG['train_splits'])} "
      f"(augmentation={CFG['USE_AUGMENTATION']}) in {time.time()-t0:.0f}s")
print("image size after upscale:", SFT_EXAMPLES[0]["image"].size)

# Arms B and C compare method at a fixed data budget on `rl_train`. If arm A has
# already consumed it, that comparison no longer exists and sections 9-12 must be
# skipped rather than run on contaminated data.
RL_AVAILABLE = "rl_train" not in CFG["train_splits"]
if not RL_AVAILABLE:
    print("\\n!! rl_train is in train_splits -> arms B and C are INVALID for this run.")
    print("   Stop after section 8 and read arm A against the test split.")
''')

# ---------------------------------------------------------------- model
md("""
## 5 · Model + LoRA

⚠️ **Verify this cell before the long run.** Class names and processor behaviour
for a given VLM generation change between `transformers` releases. If
`AutoModelForVision2Seq` fails, read the traceback — the fix is usually the
model-specific class named in the error.
""")

code('''
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

DTYPE = getattr(torch, CFG["dtype"])

def load_base(adapter_path=None, for_training=True):
    proc = AutoProcessor.from_pretrained(CFG["base_model"], trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CFG["base_model"], torch_dtype=DTYPE, device_map="auto",
        attn_implementation=CFG["attn_impl"], trust_remote_code=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=for_training)
        print("loaded adapter:", adapter_path)
    return model, proc

VISION_PREFIXES = ("visual", "vision_tower", "vision_model", "image_encoder")

def resolve_lora_targets(model, include_vision=None):
    """Enumerate real nn.Linear module paths instead of guessing leaf names.

    Two reasons this is done at runtime rather than hardcoded in CFG:

    1. Vision-tower module names differ across VLM generations (`visual.blocks.
       *.attn.qkv` on Qwen2.5-VL, other layouts elsewhere). Guessing offline
       produces a config that silently matches nothing.
    2. PEFT resolves a target string by suffix -- `key == t or
       key.endswith("."+t)`. So a *leaf* name shared by both towers (`q_proj`)
       pulls the vision tower in even when we meant language-only, which would
       make `lora_vision=False` a lie and the ablation meaningless. Returning
       fully-qualified paths makes the split exact.
    """
    include_vision = CFG["lora_vision"] if include_vision is None else include_vision
    leaves = set(CFG["lora_targets"])
    lang, vis = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        is_vision = any(p in name.split(".") for p in VISION_PREFIXES)
        if is_vision:
            vis.append(name)                       # every Linear in the tower
        elif name.rsplit(".", 1)[-1] in leaves:
            lang.append(name)                      # only the named projections
    if not vis:
        print(f"  !! no vision modules matched {VISION_PREFIXES}; inspect "
              "[n for n,_ in model.named_modules()] before trusting lora_vision")
    print(f"  language targets: {len(lang)} | vision targets: {len(vis)} "
          f"({'INCLUDED' if include_vision else 'frozen'})")
    if vis:
        print(f"  vision example:   {vis[0]}")
    return lang + (vis if include_vision else [])

def attach_lora(model, include_vision=None):
    targets = resolve_lora_targets(model, include_vision)
    cfg = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                     lora_dropout=CFG["lora_dropout"], bias="none",
                     target_modules=targets, task_type="CAUSAL_LM")
    m = get_peft_model(model, cfg); m.print_trainable_parameters(); return m

def build_messages(image, target=None):
    msgs = [{"role":"user","content":[{"type":"image","image":image},
                                      {"type":"text","text":INSTRUCTION}]}]
    if target is not None:
        msgs.append({"role":"assistant","content":[{"type":"text","text":target}]})
    return msgs

model, processor = load_base()
print(type(model).__name__, "|", type(processor).__name__)
''')

# ---------------------------------------------------------------- generation + eval
md("""
## 6 · Generation and evaluation

`evaluate()` writes **one row per document** — not just aggregates — so any
comparison between arms can be recomputed or re-tested later without re-running
generation. Paired per-document records are what make the B-vs-C comparison
statistically usable.
""")

code('''
@torch.no_grad()
def generate_batch(model, processor, images, max_new=None, temperature=0.0, num_return=1):
    max_new = max_new or CFG["gen_max_new"]
    texts = [processor.apply_chat_template(build_messages(im), tokenize=False,
                                           add_generation_prompt=True) for im in images]
    inputs = processor(text=texts, images=[[im] for im in images],
                       return_tensors="pt", padding=True).to(model.device)
    kw = dict(max_new_tokens=max_new, num_return_sequences=num_return,
              pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kw.update(do_sample=False)
    out = model.generate(**inputs, **kw)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return [clean_output(t) for t in processor.batch_decode(trimmed, skip_special_tokens=True)]

def evaluate(model, processor, split="test", n=None, tag="eval", batch=4, save=True, step=None):
    ds = SPLITS[split]; n = n or len(ds)
    idx = list(range(min(n, len(ds)))); rows = []
    for i in range(0, len(idx), batch):
        chunk = [ds[j] for j in idx[i:i+batch]]
        imgs = [encode_image(c["image"]) for c in chunk]
        preds = generate_batch(model, processor, imgs, temperature=0.0)
        for c, p in zip(chunk, preds):
            r = cer_breakdown(c["ground_truth"], p)
            r.update(doc_id=c["doc_id"], pred=p, tag=tag, step=step)
            rows.append(r)
        if i % (batch*5) == 0:
            print(f"  {tag}: {i+len(chunk)}/{len(idx)}", end="\\r")
    s = summarize(rows)
    print(f"\\n[{tag}] n={s['n']} CER={s['cer']:.4f} sub={s['sub']:.4f} "
          f"del={s['dele']:.4f} ins={s['ins']:.4f} ratio={s['ratio']:.3f} empty={s['empty']:.3f}")
    if save:
        pd.DataFrame(rows).to_json(RUN/f"metrics/preds_{tag}.jsonl", orient="records", lines=True,
                                   force_ascii=False)
        jsonl_append({**s, "tag": tag, "step": step, "ts": time.time()}, RUN/"metrics/eval_log.jsonl")
    return s, rows
''')

md("### 6b · Pre-training baseline on the test set\n\nThe reference every later number is measured against.")

code('''
base_summary, base_rows = evaluate(model, processor, "test", tag="arm0_base")
jdump(base_summary, RUN/"metrics/summary_arm0_base.json")
''')

# ---------------------------------------------------------------- SFT
md("""
## 7 · Arm A — SFT on `sft_train`

Loss is masked to the assistant turn only: the model is scored on producing the
transcription, never on reproducing the instruction or the image placeholders.

Both the per-step training loss and periodic held-out CER are logged to disk.
Training loss alone is a poor guide here — it can fall steadily while CER does
nothing, which is exactly the case worth catching early.
""")

code('''
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

class VLMOcrDataset(TorchDataset):
    def __init__(self, examples): self.ex = examples
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]

def make_collator(processor):
    tok = processor.tokenizer
    def collate(batch):
        texts  = [processor.apply_chat_template(build_messages(b["image"], b["target"]),
                                                tokenize=False) for b in batch]
        images = [[b["image"]] for b in batch]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True,
                        truncation=True, max_length=CFG["sft_max_seq_len"])
        labels = enc["input_ids"].clone()
        labels[labels == (tok.pad_token_id or tok.eos_token_id)] = -100
        # mask everything before the assistant turn so loss covers only the target
        for k, b in enumerate(batch):
            prompt = processor.apply_chat_template(build_messages(b["image"]), tokenize=False,
                                                   add_generation_prompt=True)
            plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
            labels[k, :plen] = -100
        for key in ("image_token_id", "video_token_id"):
            tid = getattr(processor, key, None) or getattr(getattr(model, "config", None), key, None)
            if tid is not None: labels[enc["input_ids"] == tid] = -100
        enc["labels"] = labels
        return enc
    return collate

class MetricLogger(TrainerCallback):
    """Streams every logged scalar to JSONL, and runs a held-out CER eval on a
    fixed cadence so the two curves can be read against each other."""
    def __init__(self, path, eval_every, eval_fn):
        self.path, self.eval_every, self.eval_fn = path, eval_every, eval_fn
        self.history = []
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        rec = {"step": state.global_step, **{k: float(v) for k, v in logs.items()
               if isinstance(v, (int, float))}, "ts": time.time()}
        self.history.append(rec); jsonl_append(rec, self.path)
    def on_step_end(self, args, state, control, **kw):
        if self.eval_every and state.global_step > 0 and state.global_step % self.eval_every == 0:
            self.eval_fn(state.global_step)
''')

code('''
model = attach_lora(model)
model.config.use_cache = False
collate = make_collator(processor)

def periodic_eval(step):
    model.eval()
    try:
        evaluate(model, processor, "test", n=CFG["sft_eval_subset"],
                 tag=f"armA_step{step}", step=step)
    finally:
        model.train()

logger = MetricLogger(RUN/"metrics/sft_train_log.jsonl", CFG["sft_eval_every"], periodic_eval)

sft_args = SFTConfig(
    output_dir=str(RUN/"armA_sft"),
    num_train_epochs=CFG["sft_epochs"],
    per_device_train_batch_size=CFG["sft_batch"],
    gradient_accumulation_steps=CFG["sft_grad_accum"],
    learning_rate=CFG["sft_lr"],
    warmup_ratio=CFG["sft_warmup_ratio"],
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    bf16=True, gradient_checkpointing=True,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=[],
    seed=CFG["seed"],
)

trainer = SFTTrainer(model=model, args=sft_args,
                     train_dataset=VLMOcrDataset(SFT_EXAMPLES),
                     data_collator=collate, callbacks=[logger])
''')

md("""
### 7a · Smoke test first

Two optimizer steps on four examples. Shape and masking bugs surface here in
under a minute instead of thirty minutes into the real run.
""")

code('''
_smoke = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir=str(RUN/"_smoke"), max_steps=2,
                   per_device_train_batch_size=1, gradient_accumulation_steps=1,
                   logging_steps=1, bf16=True, gradient_checkpointing=True,
                   remove_unused_columns=False,
                   dataset_kwargs={"skip_prepare_dataset": True}, report_to=[]),
    train_dataset=VLMOcrDataset(SFT_EXAMPLES[:4]), data_collator=collate)
_smoke.train()
print("\\nsmoke OK -- loss is finite and backward() ran. Proceed.")
''')

code('''
t0 = time.time()
train_result = trainer.train()
print(f"\\nSFT done in {(time.time()-t0)/60:.1f} min")

model.save_pretrained(RUN/"adapters/armA"); processor.save_pretrained(RUN/"adapters/armA")
jdump({k: float(v) for k, v in train_result.metrics.items()}, RUN/"metrics/armA_train_metrics.json")
pd.DataFrame(logger.history).to_csv(RUN/"metrics/armA_train_log.csv", index=False)
''')

# ---------------------------------------------------------------- plots
md("""
## 8 · SFT curves

Two panels, one axis each — never a dual-axis chart. Loss and CER live on
different scales and are plotted separately so neither is implied to track the
other.
""")

code('''
# Validated categorical palette (checked for CVD separation and contrast).
C = dict(loss="#2a78d6", cer="#eb6834", sub="#1baf7a", ins="#eda100", grid="#d8d8d4",
         ink="#0b0b0b", ink2="#52514e")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink2"], "text.color": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"]})

hist = pd.DataFrame(logger.history)
ev = pd.read_json(RUN/"metrics/eval_log.jsonl", lines=True)
ev_a = ev[ev.tag.str.startswith("armA_step")].sort_values("step") if len(ev) else pd.DataFrame()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

ax = axes[0]
if "loss" in hist:
    d = hist.dropna(subset=["loss"])
    ax.plot(d["step"], d["loss"], color=C["loss"], lw=1.2, alpha=.35)
    ax.plot(d["step"], d["loss"].rolling(15, min_periods=1).mean(), color=C["loss"], lw=2,
            label="training loss (15-step mean)")
    ax.legend(frameon=False, loc="upper right")
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy"); ax.set_title("SFT training loss", loc="left")
ax.grid(axis="y", color=C["grid"], lw=.6); ax.set_axisbelow(True)

ax = axes[1]
if len(ev_a):
    for col, lab, col_c in [("cer","CER",C["cer"]),("sub","substitutions",C["sub"]),("ins","insertions",C["ins"])]:
        ax.plot(ev_a.step, ev_a[col], color=col_c, lw=2, marker="o", ms=4, label=lab)
        ax.annotate(f"{ev_a[col].iloc[-1]:.3f}", (ev_a.step.iloc[-1], ev_a[col].iloc[-1]),
                    textcoords="offset points", xytext=(6,0), color=col_c, fontsize=8, va="center")
    ax.axhline(base_summary["cer"], color=C["ink2"], lw=1, ls="--")
    ax.annotate(f"pre-SFT CER {base_summary['cer']:.3f}", (ev_a.step.iloc[0], base_summary["cer"]),
                textcoords="offset points", xytext=(0,5), color=C["ink2"], fontsize=8)
    ax.legend(frameon=False, loc="upper right")
ax.set_xlabel("step"); ax.set_ylabel("rate"); ax.set_ylim(bottom=0)
ax.set_title("Held-out error, decomposed", loc="left")
ax.grid(axis="y", color=C["grid"], lw=.6); ax.set_axisbelow(True)

plt.tight_layout(); plt.savefig(RUN/"figures/sft_curves.png", bbox_inches="tight", facecolor="white")
plt.show()

# table view -- the numbers behind the picture, also on disk
if len(ev_a):
    print(ev_a[["step","n","cer","sub","dele","ins","ratio","empty"]].round(4).to_string(index=False))
    ev_a.to_csv(RUN/"metrics/armA_eval_curve.csv", index=False)
''')

md("### 8b · Arm A on the full test set")

code('''
model.eval()
armA_summary, armA_rows = evaluate(model, processor, "test", tag="armA_final")
jdump(armA_summary, RUN/"metrics/summary_armA.json")
print(f"\\npre-SFT CER {base_summary['cer']:.4f}  ->  arm A CER {armA_summary['cer']:.4f}   "
      f"({(base_summary['cer']-armA_summary['cer'])/max(base_summary['cer'],1e-9):+.1%})")
''')

# ---------------------------------------------------------------- pass@k gate
md("""
## 9 · GATE — `pass@16` on `rl_train`

**Decide here whether to run arm B at all.**

GRPO sharpens the distribution toward modes the policy already samples. The
useful quantity is the gap between the mean rollout and the best of k: that gap
*is* the headroom RL can convert into reliability.

Two gaps get reported. They are not the same quantity and the second is the one
that decides:

- `headroom` = mean-of-k − best-of-k. How much better the tail is than a typical
  sample.
- `greedy_minus_best` = greedy − best-of-k. Whether *sampling* beats *decoding* —
  i.e. whether there is anything RL could sharpen toward that greedy does not
  already produce.

`greedy_minus_best` is primary because it was measured to predict the failure.
On the 35B hosted run it was **0.028** before training, and the advantage standard
deviation observed at step 1 of that run was **0.032** — agreement to within 15%.
That advantage spread *is* the learning signal, so this probe is a cheap estimate
of it. Independently, [a controlled null on Qwen3-VL
4B/8B](https://arxiv.org/html/2607.12640) proposes the same screen and reports
GRPO gaining 22 points where the gap exists and nothing where it does not.

| `greedy_minus_best` | action |
|---|---|
| ≥ 0.10 | real tail → **run arm B** |
| 0.05 – 0.10 | marginal → only with a discretized reward (see §10) |
| < 0.05 | **skip arm B.** 0.028 and 0.015 both collapsed; believe it this time |
""")

code('''
@torch.no_grad()
def pass_at_k(model, processor, split="rl_train", k=None, n_docs=None, temperature=0.7):
    k = k or CFG["passk_k"]; n_docs = n_docs or CFG["passk_docs"]
    ds = SPLITS[split]; rows = []
    for i in range(min(n_docs, len(ds))):
        ex = ds[i]; im = encode_image(ex["image"])
        greedy = generate_batch(model, processor, [im], temperature=0.0)[0]
        samples = []
        for _ in range(k):
            samples += generate_batch(model, processor, [im], temperature=temperature)
        cers = [cer_breakdown(ex["ground_truth"], s)["cer"] for s in samples]
        g = cer_breakdown(ex["ground_truth"], greedy)["cer"]
        rows.append(dict(doc_id=ex["doc_id"], greedy=g, mean=float(np.mean(cers)),
                         best=float(np.min(cers)), worst=float(np.max(cers)),
                         std=float(np.std(cers)), n_better_than_greedy=int(sum(c < g for c in cers))))
        print(f"  {i+1}/{n_docs} greedy={g:.3f} mean={np.mean(cers):.3f} best={np.min(cers):.3f}", end="\\r")
    d = pd.DataFrame(rows)
    res = dict(k=k, n_docs=len(d), greedy=d["greedy"].mean(), mean=d["mean"].mean(),
               best=d["best"].mean(), headroom=float(d["mean"].mean()-d["best"].mean()),
               greedy_minus_best=float(d["greedy"].mean()-d["best"].mean()),
               frac_docs_improved=float((d["best"] < d["greedy"]).mean()),
               frac_rollouts_better=float(d["n_better_than_greedy"].sum()/(len(d)*k)))
    d.to_csv(RUN/"metrics/passk_rl_train.csv", index=False); jdump(res, RUN/"metrics/passk_rl_train.json")
    return res, d

assert RL_AVAILABLE, ("rl_train was used for SFT, so pass@k on it measures "
                      "memorisation, not headroom. Skip sections 9-12.")
passk, passk_df = pass_at_k(model, processor)
print(f"""
greedy CER         {passk['greedy']:.4f}
mean-of-{passk['k']} CER     {passk['mean']:.4f}
best-of-{passk['k']} CER     {passk['best']:.4f}
headroom           {passk['headroom']:.4f}   (mean - best)
GREEDY - BEST      {passk['greedy_minus_best']:.4f}   <- the gate
docs improved      {passk['frac_docs_improved']:.1%}
rollouts better    {passk['frac_rollouts_better']:.1%}

reference: 35B hosted run had greedy-best 0.028 pre-training, advantage sd 0.032
at step 1, and collapsed to CER 1.500 on held-out data by step 25.
""")
_g = passk["greedy_minus_best"]
print("VERDICT:", "run arm B -- real tail to sharpen" if _g >= 0.10
      else "MARGINAL -- arm B only with a discretized reward (see section 10)" if _g >= 0.05
      else "SKIP arm B -- tail too thin; RL will drift as it did on hosted")
''')

# ---------------------------------------------------------------- GRPO
md("""
## 10 · Arm B — GRPO on `rl_train`

Reward is `1 − CER` against **human** ground truth, capped at 1.5 so a single
degenerate rollout cannot dominate its group's advantage, plus a weak smooth
length term. No cliff penalties: the hosted run's `−0.5` step function at ratio
2.0 was flat inside the band (nothing pulled the policy back as it drifted) and a
wall at the edge (which whipsawed it into emitting nothing).

Two deliberate departures from the hosted config, both aimed at the drift:
`grpo_lr` 5e-6 and `beta` 0.04 — a KL anchor ~40× stronger than the `kl_tau`
1e-3 those runs used.

> ⚠️ **This reward is still the continuous one, and `RL_review_v1.md` argues it is
> the wrong shape.** The case against it, briefly: a continuous per-page CER has a
> within-group spread (~0.03) at or below its own sampling noise floor, so the
> group comparison has nothing to bite on. It also makes the `zero_advantage`
> curriculum filter a no-op, because exact reward ties never occur across 8
> rollouts — the same silent pass-through that took GSM8K from 0.754 to 0.040 in
> [the phantom-advantage study](https://arxiv.org/html/2609.13866). Every
> successful OCR RL result in the literature uses a **discrete** reward instead:
> [olmOCR 2](https://arxiv.org/abs/2510.19817) scores pages by pass-fraction over
> binary unit tests, and argues edit distance "may penalize equally correct
> outputs."
>
> One caveat specific to TRL, and it matters: prime-rl mean-centres only
> (`advantages = rewards - rewards.mean()`), but TRL's `GRPOTrainer` defaults to
> `scale_rewards="group"`, which **divides by the group standard deviation**. That
> makes a low-variance reward strictly more dangerous here than it was on hosted —
> ε-scale differences become O(1) advantages, and turning the shaping coefficient
> down provably does not help, because z-scoring cancels it
> ([the Dark Room result](https://arxiv.org/html/2607.21273v1)).
>
> The trap: `loss_type="dr_grpo"` alone does *not* turn this off — the group
> scaling is a separate knob and stays on. `scale_rewards` is set to `"none"`
> below, with a fallback for older TRL versions that expect a bool.
>
> So: if the gate returns MARGINAL, discretize before running this. Per-line exact
> match over the ground-truth lines is the cheapest discrete reward available and
> needs no synthetic data. The continuous version is left here as the documented
> baseline, not as a recommendation.
""")

code('''
from trl import GRPOConfig, GRPOTrainer

GT_BY_ID = {ex["doc_id"]: ex["ground_truth"] for ex in SPLITS["rl_train"]}
rl_reward_log = []

def ocr_reward(completions, doc_id=None, **kw):
    """1 - CER, plus -0.1*|ln(len ratio)|. Reward is always against human labels."""
    out = []
    for i, c in enumerate(completions):
        text = clean_output(c if isinstance(c, str) else c[0]["content"])
        gt = GT_BY_ID[doc_id[i]] if doc_id else ""
        b = cer_breakdown(gt, text)
        lr_pen = 0.0
        if b["ratio"] > 0:
            lr_pen = -0.1 * abs(math.log(max(b["ratio"], 1e-6)))
        r = (1.0 - b["cer"]) + lr_pen
        out.append(float(r))
        rl_reward_log.append(dict(doc_id=doc_id[i] if doc_id else None, reward=r, **b))
    return out

rl_ds = SPLITS["rl_train"].map(lambda ex: {
    "prompt": [{"role":"user","content":[{"type":"image"},{"type":"text","text":INSTRUCTION}]}],
    "image": encode_image(ex["image"]), "doc_id": ex["doc_id"]},
    remove_columns=[c for c in SPLITS["rl_train"].column_names if c != "doc_id"])

_grpo_kwargs = dict(
    output_dir=str(RUN/"armB_grpo"),
    max_steps=CFG["grpo_steps"],
    per_device_train_batch_size=CFG["grpo_batch"],
    gradient_accumulation_steps=CFG["grpo_grad_accum"],
    num_generations=CFG["grpo_num_generations"],
    learning_rate=CFG["grpo_lr"],
    beta=CFG["grpo_beta"],
    temperature=CFG["grpo_temperature"],
    max_completion_length=CFG["grpo_max_new"],
    logging_steps=1, save_steps=20, save_total_limit=2,
    bf16=True, gradient_checkpointing=True, report_to=[], seed=CFG["seed"],
)

# Disable group-std advantage scaling. TRL >= 0.22 spells this scale_rewards="none";
# older versions take a bool; oldest have no knob at all. Probe rather than assume,
# because guessing wrong here silently leaves the amplification on -- and with a
# within-group spread of ~0.03 that is the difference between a tiny gradient and
# an O(1) one built entirely from noise.
import inspect
_fields = set(inspect.signature(GRPOConfig.__init__).parameters)
if "scale_rewards" in _fields:
    for _v in ("none", False):
        try:
            grpo_args = GRPOConfig(scale_rewards=_v, **_grpo_kwargs)
            print(f"  scale_rewards={_v!r} -> group-std scaling OFF (mean-centred advantages)")
            break
        except Exception as e:
            print(f"  scale_rewards={_v!r} rejected: {type(e).__name__}")
    else:
        grpo_args = GRPOConfig(**_grpo_kwargs)
        print("  !! could not disable scale_rewards; advantages ARE std-normalised")
else:
    grpo_args = GRPOConfig(**_grpo_kwargs)
    print("  !! this TRL has no scale_rewards knob; advantages ARE std-normalised.")
    print("     Upgrade TRL or expect the low-variance amplification described above.")

grpo_logger = MetricLogger(RUN/"metrics/grpo_train_log.jsonl", 0, lambda s: None)
grpo_trainer = GRPOTrainer(model=model, args=grpo_args, train_dataset=rl_ds,
                           reward_funcs=[ocr_reward], processing_class=processor,
                           callbacks=[grpo_logger])
print("GRPO configured. If this OOMs, drop grpo_batch to 4 or num_generations to 4.")
''')

code('''
model.train()
t0 = time.time(); grpo_trainer.train()
print(f"GRPO done in {(time.time()-t0)/60:.1f} min")

model.save_pretrained(RUN/"adapters/armB")
pd.DataFrame(rl_reward_log).to_csv(RUN/"metrics/armB_rollout_rewards.csv", index=False)
pd.DataFrame(grpo_logger.history).to_csv(RUN/"metrics/armB_train_log.csv", index=False)

armB_summary, armB_rows = evaluate(model, processor, "test", tag="armB_final")
jdump(armB_summary, RUN/"metrics/summary_armB.json")
''')

md("""
### 10b · GRPO dynamics

The panel that matters is the second one. If `insertions` climbs while
`substitutions` is flat or rising, the policy is writing more rather than reading
better — the exact signature of both hosted failures. Stop the run if you see it.
""")

code('''
gl = pd.DataFrame(grpo_logger.history)
rl = pd.DataFrame(rl_reward_log)
if len(rl):
    rl["bucket"] = (np.arange(len(rl)) // max(1, len(rl)//40))
    agg = rl.groupby("bucket")[["reward","cer","sub","dele","ins","ratio"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    if "reward" in gl:
        d = gl.dropna(subset=["reward"])
        ax.plot(d["step"], d["reward"], color=C["loss"], lw=1.2, alpha=.35)
        ax.plot(d["step"], d["reward"].rolling(5, min_periods=1).mean(), color=C["loss"], lw=2,
                label="mean reward (5-step)")
        ax.legend(frameon=False)
    ax.set_xlabel("step"); ax.set_ylabel("reward"); ax.set_title("GRPO reward", loc="left")
    ax.grid(axis="y", color=C["grid"], lw=.6); ax.set_axisbelow(True)

    ax = axes[1]
    for col, lab, cc in [("sub","substitutions",C["sub"]),("ins","insertions",C["ins"]),("cer","CER",C["cer"])]:
        ax.plot(agg.bucket, agg[col], color=cc, lw=2, label=lab)
        ax.annotate(f"{agg[col].iloc[-1]:.3f}", (agg.bucket.iloc[-1], agg[col].iloc[-1]),
                    textcoords="offset points", xytext=(6,0), color=cc, fontsize=8, va="center")
    ax.legend(frameon=False); ax.set_xlabel("rollouts (binned)"); ax.set_ylabel("rate")
    ax.set_title("Error decomposition during RL", loc="left")
    ax.grid(axis="y", color=C["grid"], lw=.6); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(RUN/"figures/grpo_curves.png", bbox_inches="tight", facecolor="white")
    plt.show()
    agg.to_csv(RUN/"metrics/armB_binned.csv", index=False)
    print(agg.round(4).to_string(index=False))
''')

# ---------------------------------------------------------------- arm C
md("""
## 11 · Arm C — the control

Reload the **arm A** adapter and continue SFT on `rl_train`. Same 250 documents
arm B saw, **unaugmented**, same starting point — supervised instead of
reinforced.

Without this, "arm B beat arm A" only shows that 250 extra documents helped.
The assert below enforces the parity that makes the comparison mean anything.
""")

code('''
del model, trainer, grpo_trainer; gc.collect(); torch.cuda.empty_cache()

modelC, processorC = load_base(adapter_path=str(RUN/"adapters/armA"), for_training=True)
modelC.config.use_cache = False
collateC = make_collator(processorC)

# NO augmentation here, regardless of CFG["USE_AUGMENTATION"]. Arm B (GRPO)
# rolls out on the 250 raw rl_train documents; if arm C trained on augmented
# copies of the same 250 it would see a larger effective dataset, and B-vs-C
# would stop being a comparison of *method* at a fixed data budget -- which is
# the only reason arm C exists. Augmentation belongs to arm A alone.
C_EXAMPLES = build_examples("rl_train", augment=False)

loggerC = MetricLogger(RUN/"metrics/armC_train_log.jsonl", 0, lambda s: None)
assert len(C_EXAMPLES) == len(SPLITS["rl_train"]), (
    "arm C must see exactly the documents arm B saw, once each")
argsC = SFTConfig(output_dir=str(RUN/"armC_sft"), num_train_epochs=CFG["c_epochs"],
                  per_device_train_batch_size=CFG["sft_batch"],
                  gradient_accumulation_steps=CFG["sft_grad_accum"],
                  learning_rate=CFG["c_lr"], lr_scheduler_type="cosine",
                  logging_steps=1, bf16=True, gradient_checkpointing=True,
                  remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
                  report_to=[], seed=CFG["seed"])
trainerC = SFTTrainer(model=modelC, args=argsC, train_dataset=VLMOcrDataset(C_EXAMPLES),
                      data_collator=collateC, callbacks=[loggerC])
trainerC.train()
modelC.save_pretrained(RUN/"adapters/armC")
pd.DataFrame(loggerC.history).to_csv(RUN/"metrics/armC_train_log.csv", index=False)

modelC.eval()
armC_summary, armC_rows = evaluate(modelC, processorC, "test", tag="armC_final")
jdump(armC_summary, RUN/"metrics/summary_armC.json")
''')

# ---------------------------------------------------------------- comparison
md("""
## 12 · Paired comparison

Every arm scored the same 100 documents, so differences are tested **per
document**. A paired test removes between-document variance, which here (CER std
≈ 0.18) dwarfs the effects being measured — an unpaired comparison on n=100 would
not resolve them.
""")

code('''
from scipy import stats

def paired(rows_a, rows_b, name_a, name_b):
    a = pd.DataFrame(rows_a).set_index("doc_id")["cer"]
    b = pd.DataFrame(rows_b).set_index("doc_id")["cer"]
    common = a.index.intersection(b.index); a, b = a[common], b[common]
    d = a - b
    t, p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b) if len(common) > 10 else None
    return dict(comparison=f"{name_a} -> {name_b}", n=len(common),
                cer_a=float(a.mean()), cer_b=float(b.mean()), delta=float(d.mean()),
                pct_change=float(d.mean()/max(a.mean(),1e-9)),
                docs_improved=float((b < a).mean()),
                cohens_d=float(d.mean()/max(d.std(),1e-9)),
                t_p=float(p), wilcoxon_p=float(w.pvalue) if w else None)

comparisons = [
    paired(base_rows, armA_rows, "base", "armA_SFT"),
    paired(armA_rows, armB_rows, "armA_SFT", "armB_SFT+RL"),
    paired(armA_rows, armC_rows, "armA_SFT", "armC_SFT+SFT"),
    paired(armC_rows, armB_rows, "armC_SFT+SFT", "armB_SFT+RL"),   # <- the experiment
]
comp = pd.DataFrame(comparisons)
comp.to_csv(RUN/"metrics/paired_comparisons.csv", index=False)
jdump(comparisons, RUN/"metrics/paired_comparisons.json")
print(comp.round(4).to_string(index=False))

summ = pd.DataFrame([{"arm":"base (no training)", **base_summary},
                     {"arm":"A: SFT", **armA_summary},
                     {"arm":"B: SFT+RL", **armB_summary},
                     {"arm":"C: SFT+SFT", **armC_summary}])
summ.to_csv(RUN/"metrics/arm_summaries.csv", index=False)
print("\\n", summ.round(4).to_string(index=False))
''')

code('''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

ax = axes[0]
arms = summ["arm"].tolist(); vals = summ["cer"].tolist()
bars = ax.bar(range(len(arms)), vals, color=[C["ink2"], C["loss"], C["cer"], C["sub"]],
              width=.62, zorder=3)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0,4),
                ha="center", fontsize=9, color=C["ink"])
ax.set_xticks(range(len(arms))); ax.set_xticklabels([a.split(":")[0] for a in arms])
ax.set_ylabel("CER (lower is better)"); ax.set_title("Held-out CER by arm", loc="left")
ax.grid(axis="y", color=C["grid"], lw=.6); ax.set_axisbelow(True)

ax = axes[1]
stack = summ[["sub","dele","ins"]].values
bottom = np.zeros(len(summ))
for j,(lab,cc) in enumerate([("substitutions",C["sub"]),("deletions",C["loss"]),("insertions",C["ins"])]):
    ax.bar(range(len(summ)), stack[:,j], bottom=bottom, color=cc, width=.62,
           label=lab, zorder=3, edgecolor="white", linewidth=2)
    bottom += stack[:,j]
ax.set_xticks(range(len(summ))); ax.set_xticklabels([a.split(":")[0] for a in arms])
ax.set_ylabel("rate"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Where the errors are", loc="left")
ax.grid(axis="y", color=C["grid"], lw=.6); ax.set_axisbelow(True)

plt.tight_layout(); plt.savefig(RUN/"figures/arm_comparison.png", bbox_inches="tight", facecolor="white")
plt.show()
''')

md("""
## 13 · Readout

**The headline is the last row of the comparison table: `armC → armB`.** That is
RL versus the same data spent on supervision.

| result | interpretation |
|---|---|
| B better than C, p < 0.05 | RL adds real value beyond the data it consumed |
| B ≈ C | SFT→RL is not worth the complexity here; ship arm C |
| B worse than C | RL is degrading a good policy — consistent with both hosted runs |

Read the second panel too. If arm B's improvement comes from **insertions**
falling rather than **substitutions**, it learned to stop rambling, not to read
better — a real but much less interesting result than it looks.

**Reference points for arm A's CER**, on this same benchmark: 0.168 is the
untrained high-res baseline, **0.159 is TrOCR** (the bar), 0.032 is
Gemini-2.5-Pro. A frontier VLM reads these images at 0.022, so the information is
in the pixels and any shortfall is model capability, not data.

**If arm A barely moved,** in order of what to check:

1. **Confirm the vision tower actually trained.** `lora_vision=True` is the
   default now, but `resolve_lora_targets` prints how many vision modules it
   matched — if that count is 0, the prefix list missed this architecture and you
   silently trained language-only. That single line is the check.
2. **Run the language-only ablation** (`lora_vision=False`). If it matches, the
   encoder was never the bottleneck and the ceiling is elsewhere.
3. **Suspect data volume before architecture.** 479 page-level examples is below
   the range where this is known to work: comparable low-resource results used 911
   samples (Jawi, Qwen2-VL-2B → CER 0.0866) and 60k (Manchu). The corpus holds
   **9,982 ground-truth-aligned lines**, and projection-profile segmentation
   recovers an exact line count on ~61% of documents — roughly 5,000 usable
   line-level examples, at ~40 output tokens each instead of ~1,000. That is the
   next lever, and it is larger than any hyperparameter here.
4. **Resolution, again.** 2× on high-res is the measured operating point but the
   sweep was coarse. 1× and 3× are one config change each.

Everything written to disk under `run_dir`:

```
metrics/
  sft_train_log.jsonl        per-step loss/lr/grad-norm (arm A)
  armA_train_log.csv         same, tabular
  armA_eval_curve.csv        periodic held-out CER during SFT
  eval_log.jsonl             every evaluation, all arms
  preds_<tag>.jsonl          per-document predictions + CER breakdown
  passk_rl_train.{csv,json}  the arm-B gate
  armB_rollout_rewards.csv   per-rollout reward + decomposition
  armB_binned.csv            binned GRPO dynamics
  arm_summaries.csv          one row per arm
  paired_comparisons.{csv,json}
figures/
  sft_curves.png  grpo_curves.png  arm_comparison.png
adapters/  armA/  armB/  armC/
```
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
