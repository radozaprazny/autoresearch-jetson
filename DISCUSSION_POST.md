# Jetson port + workflow integrity checks that made 102 experiments fully reproducible (3.50% improvement)

## 🎯 The Problem

While porting autoresearch to Jetson, I kept hitting the same workflow failures:

- 🔴 Running experiments with uncommitted changes → impossible to reproduce which commit produced result
- 🔴 OOM crashes from oversized models → hours wasted on 8GB unified memory
- 🔴 Accidentally editing code mid-experiment → corrupted results with no trace
- 🔴 Agent stuck micro-tuning hyperparameters → 10+ discards in noise floor

**After 10+ failed runs, I added workflow guards to make these mistakes impossible.**

## ✅ The Solution: Automated Workflow Improvements

Added 4 integrity checks and workflow rules that enforced reproducible experiments across 102 runs with zero failures:

### 1. **Git Dirty Check** — Reproducibility Guard
```python
# train.py:458-465
result = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD'], capture_output=True)
if result.returncode != 0:
    print("❌ ERROR: Uncommitted changes detected")
    print("   WORKFLOW: git commit → python train.py")
    sys.exit(1)
```
**Impact:** Every experiment has matching git commit. Zero ambiguity. Script refuses to run on uncommitted changes.

### 2. **VRAM Pre-flight Check** — OOM Prevention
```python
# train.py:504-507
_vram_est_gb = num_params * 2 / 1e9 * 3.5
if _vram_est_gb > 6.8:
    print(f"⚠️  WARNING: VRAM Est ~{_vram_est_gb:.1f}GB -> Potential OOM")
```
**Impact:** Catches oversized models before training starts. Zero OOM crashes across 102 experiments on 8GB Jetson.

### 3. **Script Hash Tracking** — Detect Post-Commit Edits
```python
# train.py:651-652
with open(__file__, 'rb') as f:
    _script_hash = hashlib.sha256(f.read()).hexdigest()[:8]
print(f"script_hash:      {_script_hash}")
```
**Impact:** Easy detection of accidental code modifications after commit. Logged in every experiment output.

### 4. **Anti-Stagnation Rule** — Escape Noise Floor
After 5 consecutive discards with <0.002 improvement (program.md):
- ❌ **Forbidden:** LR tweaks, weight decay micro-adjustments
- ✅ **Required:** Activation functions, attention mechanisms, architectural changes

**Impact:** Agent forced to explore structure instead of getting stuck in hyperparameter noise. Prevents dead-ends.

## 📊 Results: Jetson Orin Nano Super (ARM64)

**Hardware:** 8GB unified memory, 1024 CUDA cores, ARM64, JetPack 6.1  
**Model:** 1.6 GB (27× smaller than H100's 44GB model)  
**Steps:** ~1350 per 5-min experiment (vs ~8000 on H100)

- ✅ Total experiments: 102
- ✅ Baseline: 1.454936 bpb
- ✅ Best: 1.404085 bpb (commit `2e6bd5b`)
- ✅ Improvement: **3.50%**
- ✅ Keep rate: 19.6% (20/102)
- ✅ **100% git compliance** — every experiment has clean git state
- ✅ **Zero crashes** — 0 OOM errors, 0 config violations, all experiments completed

## 🔧 Code Changes

**Platform (ARM64 compatibility):**
- Flash Attention 3 → PyTorch SDPA
- Disabled `torch.compile` (no Triton on ARM64)
- Reduced model depth 8→4, aspect ratio 96→64 (apr4) / 60 (master)

**Workflow improvements (platform-independent):**
- `train.py:458-465` — git dirty check (enforces commit-before-run)
- `train.py:504-507` — VRAM pre-flight check (prevent OOM)
- `train.py:651-652` — script hash logging (detect post-commit edits)
- `program.md:36-49` — anti-stagnation rule (force structural exploration)

## 🤔 Questions for Community

1. **Would these workflow improvements be useful on other platforms?** (H100, RTX, MPS, etc.)
   - Git dirty check, VRAM pre-flight, script hash — all platform-agnostic
   - Zero dependencies, ~30 lines of code total

2. **Interest in upstreaming?**
   - Could make them optional via command-line flag (`--strict-workflow`)
   - Especially git dirty check + anti-stagnation rule — these caught real problems

3. **What other workflow pitfalls should I automate away?**
   - I'm sure there are more mistakes I haven't encountered yet

## 📦 Repo

**Fork:** [radozaprazny/autoresearch-jetson](https://github.com/radozaprazny/autoresearch-jetson)

**Two branches to show incremental value:**

| Branch | Purpose | Files |
|--------|---------|-------|
| `master` | Baseline Jetson port (ARM64 compatibility only) | Only platform changes |
| `autoresearch/apr4` ⭐ | **With integrity checks** (recommended) | + [results.tsv](https://github.com/radozaprazny/autoresearch-jetson/blob/autoresearch/apr4/results.tsv) (102 experiments) |

**Progress graph:**

![Jetson apr4 progress](https://github.com/radozaprazny/autoresearch-jetson/raw/master/progress.png?raw=true)

*Shows validation loss (bpb) over 102 experiments. Green points = kept improvements, gray = discarded. Annotations show commit messages.*

> 💡 Check out `autoresearch/apr4` branch to see complete results and workflow implementation.

---

Happy to answer questions about Jetson specifics or the integrity check implementation! The checks caught every single mistake I made during the 102-experiment run. 🎯
