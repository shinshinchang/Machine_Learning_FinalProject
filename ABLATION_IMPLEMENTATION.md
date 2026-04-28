# CEINN Ablation Experiment Implementation

## Overview
Implemented complete ablation experiment support for the CEINN model with the following ablation modes:
- `none`: Full CEINN model
- `w/o_pt`: Without Prospect Theory
- `w/o_hd`: Without Hyperbolic Discounting
- `w/o_mtl`: Without Multi-Task Learning
- `w/o_causal`: Without Causal Adjustment

## Files Modified

### 1. `train_ceinn.py`
**Changes made:**

#### a. Updated argument parser (Line 18)
```python
# Before
parser.add_argument('--ablation', type=str, default='none', 
                   choices=['none', 'no_pt', 'no_hd', 'no_mtl', 'no_causal'])

# After
parser.add_argument('--ablation', type=str, default='none', 
                   choices=['none', 'w/o_pt', 'w/o_hd', 'w/o_mtl', 'w/o_causal'])
```

#### b. Updated ablation_to_flags function (Lines 23-33)
```python
def ablation_to_flags(name: str) -> AblationFlags:
    flags = AblationFlags(True, True, True, True)
    if name == 'w/o_pt':      # Changed from 'no_pt'
        flags.use_pt = False
    elif name == 'w/o_hd':    # Changed from 'no_hd'
        flags.use_hd = False
    elif name == 'w/o_mtl':   # Changed from 'no_mtl'
        flags.use_mtl = False
    elif name == 'w/o_causal': # Changed from 'no_causal'
        flags.use_causal = False
    return flags
```

#### c. Added sanitize_ablation_name function (Lines 36-38)
```python
def sanitize_ablation_name(name: str) -> str:
    """Convert ablation name to filesystem-safe format (w/o_pt -> w_o_pt)"""
    return name.replace('/', '_')
```
**Reason:** Windows/Unix filesystems don't allow `/` in filenames; converts `w/o_pt` → `w_o_pt` for file paths.

#### d. Updated file saving (Lines 161-162, 187)
```python
# Before
best_path = output_dir / f'best_model_{args.ablation}.pt'
metrics_path = output_dir / f'metrics_{args.ablation}.json'

# After
ablation_safe = sanitize_ablation_name(args.ablation)
best_path = output_dir / f'best_model_{ablation_safe}.pt'
metrics_path = output_dir / f'metrics_{ablation_safe}.json'
```

### 2. `model.py`
**Changes made:**

#### Updated forward() method (Lines 208-246)
**Key change:** Proper handling of w/o_pt by skipping dynamic reference point calculation

```python
# Before: Dynamic reference always calculated
ref = self.dynamic_reference(economic_state, do_abs_util)
delta = do_abs_util - ref.unsqueeze(-1)
long_score = self.smooth_prospect_value(delta, use_pt=flags.use_pt)

# After: Conditionally calculate based on ablation
if flags.use_pt:
    ref = self.dynamic_reference(economic_state, do_abs_util)
    delta = do_abs_util - ref.unsqueeze(-1)
    long_score = self.smooth_prospect_value(delta, use_pt=True)
else:
    # For w/o_pt ablation: use raw utility directly as long-term score
    long_score = do_abs_util
    ref = torch.zeros(do_abs_util.size(0), device=do_abs_util.device)
```

**Rationale:**
- `use_pt=True`: Applies Prospect Theory transformation (reference point + smoothed value function)
- `use_pt=False`: Uses raw utility logits directly, bypassing reference-dependence

### 3. `README.md`
**Changes made:**

#### Updated supported features section (Lines 19-23)
```markdown
# Before
- 消融開關：
  - `--ablation none`
  - `--ablation no_pt`
  - `--ablation no_hd`
  - `--ablation no_mtl`
  - `--ablation no_causal`

# After
- 消融實驗支援（Ablation Studies）：
  - `--ablation none` : 完整 CEINN 模型
  - `--ablation w/o_pt` : 無 Prospect Theory（直接用原始效用，無動態參考點）
  - `--ablation w/o_hd` : 無 Hyperbolic Discounting（基本因果注意力遮罩）
  - `--ablation w/o_mtl` : 無多任務學習（僅經濟頭，無語意頭）
  - `--ablation w/o_causal` : 無因果調整（無 SNIPS 權重、無潛在混淆因子）
```

#### Updated ablation examples section (Lines 50-61)
```markdown
# Before
## 4. 消融
python train_ceinn.py --config config.yaml --ablation no_pt
...

# After
## 4. 消融實驗
```bash
# 無 Prospect Theory（基本效用評分）
python train_ceinn.py --config config.yaml --ablation w/o_pt

# 無 Hyperbolic Discounting（基本自注意力）
python train_ceinn.py --config config.yaml --ablation w/o_hd

# 無多任務學習（僅長期經濟信號）
python train_ceinn.py --config config.yaml --ablation w/o_mtl

# 無因果調整（無 SNIPS、無混淆因子）
python train_ceinn.py --config config.yaml --ablation w/o_causal
```
```

## Ablation Implementation Details

### 1. `w/o_pt` (Without Prospect Theory)
- **What's disabled:** 
  - Dynamic reference point calculation
  - Prospect Theory utility transformation
  - Loss aversion parameter λ
- **What's used instead:** Raw utility logits from causal intervention (economic stream)
- **Code path:** `model.py:214-218` → `long_score = do_abs_util`

### 2. `w/o_hd` (Without Hyperbolic Discounting)
- **What's disabled:** Hyperbolic discount bias in attention
- **Implementation:** `encode_sequence(seq, use_hd=False)` in `EconomicSelfAttention`
- **Effect:** Uses only causal mask, no position-dependent bias
- **Existing code:** Already implemented in model, just uses flag

### 3. `w/o_mtl` (Without Multi-Task Learning)
- **What's disabled:** Short-term semantic head contribution
- **What remains:** Only long-term economic head
- **Code path:** `train_ceinn.py:54 & model.py:230-231` → `short_score = torch.zeros_like(...)`
- **Effect:** `total_score = 0 + long_score / (1 + softplus(κ))`

### 4. `w/o_causal` (Without Causal Adjustment)
- **What's disabled:**
  - SNIPS propensity weighting
  - Latent confounder prototype adjustment
  - Causal intervention utility
- **Implementation:** 
  - `do_intervention_utility(... use_causal=False)` → simplified utility
  - `compute_loss()` uses `weights = torch.ones_like(...)` instead of SNIPS weights
- **Effect:** Standard empirical risk minimization

## Output Files

For each ablation, the following files are generated:

| Ablation | Model File | Metrics File |
|----------|-----------|--------------|
| none | `outputs/best_model_none.pt` | `outputs/metrics_none.json` |
| w/o_pt | `outputs/best_model_w_o_pt.pt` | `outputs/metrics_w_o_pt.json` |
| w/o_hd | `outputs/best_model_w_o_hd.pt` | `outputs/metrics_w_o_hd.json` |
| w/o_mtl | `outputs/best_model_w_o_mtl.pt` | `outputs/metrics_w_o_mtl.json` |
| w/o_causal | `outputs/best_model_w_o_causal.pt` | `outputs/metrics_w_o_causal.json` |

## Usage Examples

### Train full CEINN model
```bash
python train_ceinn.py --config config.yaml --ablation none
```

### Train without Prospect Theory
```bash
python train_ceinn.py --config config.yaml --ablation w/o_pt
```

### Train without Hyperbolic Discounting
```bash
python train_ceinn.py --config config.yaml --ablation w/o_hd
```

### Train without Multi-Task Learning
```bash
python train_ceinn.py --config config.yaml --ablation w/o_mtl
```

### Train without Causal Adjustment
```bash
python train_ceinn.py --config config.yaml --ablation w/o_causal
```

## Verification

All ablation modes have been tested to ensure:
- ✅ Argument parser accepts all 5 ablation modes
- ✅ AblationFlags are correctly set for each mode
- ✅ Model forward() respects all flags
- ✅ File naming is filesystem-safe (no `/` in paths)
- ✅ Metrics are saved with correct ablation name
- ✅ No component is used unless its flag is True

## Backward Compatibility

- The default ablation is `none` (full model), maintaining backward compatibility
- Old `no_*` naming is no longer supported (breaks backward compatibility but improves clarity)
- Config file remains unchanged
