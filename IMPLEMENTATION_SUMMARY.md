# Checkpoint CLI Overrides Implementation Summary

## Overview
Fixed `uv run train --init-from-checkpoint` to properly override experiment knobs and run-control fields from CLI arguments while preserving checkpoint-owned architecture fields for weight compatibility.

## Changes Made

### 1. CLI Argument Tracking (`src/training/train.py`)
- **Modified `parse_args()`** to return a 4-tuple instead of 3-tuple:
  - Now returns: `(config, verbose, init_checkpoint, explicitly_provided)`
  - `explicitly_provided` is a `set[str]` containing field names that were explicitly provided on the command line
  - Tracks arguments by scanning `sys.argv` for `--` prefixed arguments before calling `parser.parse_args()`
  - Normalizes CLI keys (e.g., `--learning-rate` → `learning_rate`)

### 2. Refactored Merge Logic (`src/training/train.py`)
- **Enhanced `merge_config_from_init_checkpoint()`** with three explicit field categories:

#### Checkpoint-Owned Fields (Architecture - Never Overridden)
These must come from checkpoint for weight compatibility:
- `value_network`
- `max_exponent`
- `embedding_dim`
- `hidden_dim`

#### CLI-Overridable Fields (Experiment + Run-Control)
These override checkpoint only when explicitly supplied on CLI:
- **Experiment knobs**: `learning_rate`, `gamma`, `batch_size`, `target_update_interval`, `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`, `exploration`, `grad_clip`
- **Run-control**: `seed`, `steps`, `model_dir`, `device`, `replay_capacity`, `checkpoint_interval`, `eval_interval`, `learning_starts`, `train_frequency`

#### Other Fields
All other fields stay checkpoint-derived (via `train_config_from_dict` normalization).

### 3. Updated Training Loop (`src/training/train.py`)
- **Modified `train()` function signature** to accept:
  - `explicitly_provided: set[str] | None` — tracks which CLI args were explicit
  - `learning_rate_was_overridden: bool` — flag for optimizer state handling
- **Implemented optimizer state learning rate reset**:
  - When `learning_rate` is explicitly overridden via CLI, the function resets all parameter-group learning rates after loading optimizer state
  - Ensures optimizer state from checkpoint doesn't interfere with new learning rate

### 4. Updated Main Entry Point (`src/training/train.py`)
- **Modified `main()`** to unpack and pass through the new return values from `parse_args()`

### 5. Test Coverage
Created comprehensive test suites:

#### `tests/test_train_checkpoint_overrides.py` (8 tests)
- Architecture fields always preserved from checkpoint
- Experiment fields overridden only when explicitly supplied
- Run-control fields overridden only when explicitly supplied
- Empty explicitly_provided set keeps all from checkpoint
- None explicitly_provided achieves backward compatibility
- Epsilon schedule override works
- Exploration strategy override works
- All experiment fields can be overridden together

#### `tests/test_train_optimizer_state.py` (2 tests)
- Optimizer state learning rate reset when explicitly overridden
- Optimizer state unchanged when learning_rate not explicitly provided

#### Updated `tests/test_imitation.py`
- Fixed `test_merge_configs_from_ckpt_stub` to explicitly provide fields when testing merge behavior

## Behavior Examples

### Before (Bug)
```bash
# Checkpoint had learning_rate=5e-4, epsilon_start=0.8
uv run train --init-from-checkpoint checkpoint.pt --learning-rate 1e-3 --epsilon-start 0.5
# Result: silently ignored CLI values, used checkpoint values
# Config: learning_rate=5e-4, epsilon_start=0.8
```

### After (Fixed)
```bash
# Checkpoint had learning_rate=5e-4, epsilon_start=0.8, value_network=qcnn
uv run train --init-from-checkpoint checkpoint.pt --learning-rate 1e-3 --epsilon-start 0.5
# Result: applies CLI overrides while preserving architecture
# Config: learning_rate=1e-3, epsilon_start=0.5, value_network=qcnn
```

### Mixed Scenario
```bash
# Checkpoint: learning_rate=5e-4, batch_size=256, steps=50000
uv run train --init-from-checkpoint checkpoint.pt --learning-rate 1e-3
# Result: only learning_rate overridden, batch_size and steps stay from checkpoint
# Config: learning_rate=1e-3, batch_size=256, steps=50000
```

## Architecture Compliance
- ✓ `game.py` remains dependency-free
- ✓ Training code stays in `src/training/`
- ✓ Follows existing patterns in codebase
- ✓ No breaking changes to checkpoint format

## Testing Results
- ✓ All 92 tests pass
- ✓ 8 new checkpoint override tests
- ✓ 2 new optimizer state tests
- ✓ Updated 1 existing test for backward compatibility
- ✓ Real checkpoint verification (current_best_imitation_checkpoint.pt works)

## Backward Compatibility
- Function signatures are backward compatible when `explicitly_provided=None`
- Old code calling `merge_config_from_init_checkpoint(config, path)` still works
- When `explicitly_provided=None`, behavior defaults to: explicitly provided set is empty, so unspecified fields come from checkpoint

## Key Implementation Details

### CLI Parsing Strategy
Uses `sys.argv` scanning before argparse to identify explicitly provided arguments. This is more reliable than using argparse's action tracking, as argparse defaults are set at definition time.

### Field Category Strategy
Three-tier classification (checkpoint-owned, CLI-overridable, other) provides clear semantics:
1. Architecture fields protected from CLI interference
2. Experiment/run-control fields override only when explicit
3. Other fields inherit checkpoint values or defaults

### Optimizer State Handling
Reset parameter-group learning rates only when explicitly overridden, ensuring:
- Old optimizer state can be reused with new learning rate
- Checkpoint gradient accumulators still valid
- No unexpected learning rate jumps from stale optimizer state

## Files Modified
1. `src/training/train.py` — core implementation
2. `tests/test_imitation.py` — updated one test for new behavior
3. `tests/test_train_checkpoint_overrides.py` — new comprehensive tests
4. `tests/test_train_optimizer_state.py` — new optimizer state tests
