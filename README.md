# autoresearch — Jetson Orin Nano Super (ARM64) + Workflow Improvements

![teaser](progress.png?raw=true)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Full credit to [@karpathy](https://github.com/karpathy) for the core idea: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies `train.py`, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. The metric is val_bpb (validation bits per byte) — lower is better.

This fork:
1. Ports autoresearch to **NVIDIA Jetson Orin Nano Super** (ARM64, 8GB unified memory)
2. Adds **automated workflow integrity checks** to ensure reproducible experiments

## Hardware

NVIDIA Jetson Orin Nano Super Developer Kit — 8 GB unified memory (CPU + GPU shared), 1024 CUDA cores, JetPack 6.1 / CUDA 12.6, ARM64 (aarch64). SDPA (no Flash Attention 3). `torch.compile` unavailable — Triton is not supported on ARM64. ~1.4 GB model footprint, ~1350 steps per 5-minute experiment.

## Results

> ⚠️ **Note:** This fork has two branches with different purposes (see [Branches](#branches) section below). Complete results are on `autoresearch/apr4` branch.

**Final results** (from `autoresearch/apr4` branch):

- Total experiments: 102
- Baseline: 1.454936 bpb
- Best: 1.404085 bpb (commit `2e6bd5b`)
- Improvement: **3.50%**
- Keep rate: 19.6% (20/102)
- Full log: [**results.tsv**](https://github.com/radozaprazny/autoresearch-jetson/blob/autoresearch/apr4/results.tsv) ← on `apr4` branch

**Comparison:**
- `master` branch: 13 keeps (baseline port only)
- `autoresearch/apr4` branch: 20 keeps (with integrity checks)

## Workflow Improvements (apr4 branch)

This fork adds automated workflow guards to prevent common experiment failures and ensure reproducibility:

### 1. **Git Dirty Check** (`train.py:458-465`)
**Problem:** Running experiments with uncommitted changes makes results non-reproducible.  
**Solution:** Script refuses to run unless working directory is clean.  
**Impact:** 100% compliance — all 102 experiments have matching git commits.

```python
# Before training starts
result = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD'], capture_output=True)
if result.returncode != 0:
    print("❌ ERROR: Uncommitted changes detected")
    print("   WORKFLOW: git commit → python train.py")
    sys.exit(1)
```

**Enforces workflow:** `git commit` → `train.py` → record result

### 2. **VRAM Pre-flight Check** (`train.py:504-507`)
**Problem:** Silent OOM crashes waste hours on 8GB unified memory.  
**Solution:** Estimate VRAM usage before training starts, warn if approaching limits.  
**Impact:** Zero OOM failures across 102 experiments.

```python
# VRAM estimate - soft warning for Jetson 8GB
_vram_est_gb = num_params * 2 / 1e9 * 3.5
if _vram_est_gb > 6.8:
    print(f"⚠️  WARNING: VRAM Est ~{_vram_est_gb:.1f}GB -> Potential OOM")
```

### 3. **Script Hash Tracking** (`train.py:651-652`)
**Problem:** Accidentally editing `train.py` after commit invalidates results.  
**Solution:** Log SHA-256 hash of script at experiment end.  
**Impact:** Easy detection of post-commit modifications in experiment logs.

```python
with open(__file__, 'rb') as f:
    _script_hash = hashlib.sha256(f.read()).hexdigest()[:8]
print(f"script_hash:      {_script_hash}")
```

### 4. **Anti-Stagnation Rule** (`program.md:36-49`)
**Problem:** Agent gets stuck micro-tuning hyperparameters in noise floor.  
**Solution:** After 5 consecutive discards with <0.002 improvement, force structural changes.  
**Impact:** Prevents hyperparameter dead-ends, encourages architectural exploration.

**Forbidden:** LR tweaks, weight decay adjustments  
**Required:** Activation functions, attention mechanisms, architecture changes

## Platform Adaptations

Code changes from upstream required for Jetson ARM64:

**`train.py`:**
- Removed Flash Attention 3 kernel import (`from kernels import get_kernel`)
- Replaced with `F.scaled_dot_product_attention` (PyTorch native SDPA)
- Disabled `torch.compile` (Triton not supported on ARM64)
- Reduced model: DEPTH=3 (vs 8 on H100), ASPECT_RATIO=64

**`prepare.py`:**
- `MAX_SEQ_LEN=2048` (same as upstream)
- Reduced `EVAL_TOKENS` to fit 8GB VRAM

## Branches

This fork maintains **two branches** to show incremental value:

### `master` — Baseline Jetson Port

**Purpose:** Minimal changes to run on ARM64 (platform compatibility only)  
**What's different from upstream:**
- Flash Attention → PyTorch SDPA
- Disabled `torch.compile` (ARM64 lacks Triton)
- Reduced model size (DEPTH=3)

**Status:** 13 keep experiments tracked in git history  
**Note:** No `results.tsv` file committed (following upstream's `.gitignore`)

### `autoresearch/apr4` — **With Workflow Integrity Checks** ⭐

**Purpose:** Everything from `master` + automated workflow improvements  
**Additional changes:**
- 4 workflow guards (git dirty check, VRAM pre-flight, script hash, anti-stagnation)
- Enforces commit-before-run workflow

**Status:** 20 keep experiments, 102 total runs  
**Files:**
- [**results.tsv**](https://github.com/radozaprazny/autoresearch-jetson/blob/autoresearch/apr4/results.tsv) — complete experiment log (103 lines)
- [progress.png](https://github.com/radozaprazny/autoresearch-jetson/blob/autoresearch/apr4/progress.png) — generated by `analysis.ipynb`

**Tag:** `jetson-apr4-final`

---

> 💡 **For new visitors:** If you want to see the complete results and workflow improvements, check out the **`autoresearch/apr4`** branch.  
> If you're only interested in ARM64 platform compatibility changes, `master` has minimal diff from upstream.

## Repository Files

- `train.py` — Training script with integrity checks (apr4 branch)
- `program.md` — Instructions for AI agent (includes anti-stagnation rule)
- `prepare.py` — Data download and tokenizer prep
- `analysis.ipynb` — Generates `progress.png` from `results.tsv` (visualization)
- `results.tsv` — Complete experiment log (only on `autoresearch/apr4` branch)
- `progress.png` — Progress graph showing all keep improvements

## Quick start

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), NVIDIA Jetson with JetPack 6.1+.

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/radozaprazny/autoresearch-jetson.git
cd autoresearch-jetson

# 3. Choose branch (master = baseline, apr4 = with integrity checks)
git checkout autoresearch/apr4  # recommended

# 4. Install dependencies (JetPack PyTorch wheel uses non-standard filename)
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv sync

# 5. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 6. Run single experiment (~5 min)
uv run train.py
```

Then point Claude/Copilot or another coding agent at `program.md` and let it run the loop.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the original idea
- Community contributors — see git history for full credits

## License

MIT

