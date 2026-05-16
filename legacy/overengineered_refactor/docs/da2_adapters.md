# DA2 adapters and minimal checkpoints

This document describes the LoRA/adapter support introduced in Phase 6 of the
DA2 refactor.

## Design principle

LoRA is treated as an **adapter axis**, separate from the original DA2 parameter
scope.  There is no separate high-level `tuning.mode` label in the final config;
the effective behavior is exactly the combination of `base` and `adapter`.

```yaml
base:
  trainable_scope: refinenets_output  # original DA2 parameters to train

adapter:
  enabled: true                       # optional adapter insertion
  type: lora
```

This means these combinations are intentionally valid:

- LoRA-only: `base.trainable_scope: frozen` and `adapter.enabled: true`;
- decoder + LoRA;
- refinenets_output + LoRA;
- full + LoRA.

## LoRA config

Example LoRA-only decoder adapter config:

```yaml
base:
  trainable_scope: frozen

adapter:
  enabled: true
  type: lora
  rank: 8
  alpha: 16
  dropout: 0.05
  target:
    mode: decoder
```

Example LoRA on the currently trainable base scope:

```yaml
base:
  trainable_scope: refinenets_output

adapter:
  enabled: true
  type: lora
  rank: 8
  alpha: 16
  dropout: 0.05
  target:
    mode: trainable_scope
```

Supported target modes:

- `trainable_scope`: wrap supported modules whose direct parameters are already
  trainable after `base.trainable_scope` is applied;
- `decoder`: wrap supported modules under `depth_head`;
- `all_linear`: wrap all `nn.Linear` modules;
- `regex`: wrap supported modules whose full module name matches one of
  `adapter.target.patterns`.

Supported module types in the current implementation:

- `torch.nn.Linear`;
- `torch.nn.Conv2d` with `groups=1`.

## Checkpoint policy

The default checkpoint policy is `trainable_only`:

```yaml
checkpoint:
  save_policy: trainable_only
  keep_latest: true
  keep_best: true
  save_optimizer: true
```

A trainable-only checkpoint is reconstructed from:

```text
base DA2 checkpoint + effective_config.yaml + trainable checkpoint payload
```

The payload format is:

```text
format: da2_trainable_checkpoint_v1
adapter:          LoRA parameters only
trainable_base:   changed original DA2 trainable parameters only
trainable:        adapter + trainable_base convenience union
base_checkpoint:  path to the base DA2 checkpoint used for reconstruction
config:           effective training config
```

For LoRA-only runs, `trainable_base` is empty.  For decoder+LoRA or
refinenets_output+LoRA, `trainable_base` stores only the changed original DA2
parameters.

`latest.pth` may include optimizer/scaler state for resume.  `best.pth` is saved
without optimizer/scaler by default so it stays small for evaluation and reuse.

Full-model saving remains available through:

```yaml
checkpoint:
  save_policy: full_model
```

but should be used intentionally because it is much larger.

## Example configs

Phase 6 added:

- `configs/experiments/da2_vits_lora_decoder.yaml`
  - LoRA-only adapter on the DA2 decoder;
- `configs/experiments/da2_vits_refinenets_output_lora.yaml`
  - refinenets_output base fine-tuning plus LoRA on the trainable scope.

## Evaluation

`scripts/eval.py` can evaluate trainable-only checkpoints directly:

```bash
python scripts/eval.py \
  --checkpoint /work/scratch/$USER/cil-visionavengers-depth/checkpoints/<run>/<timestamp>/best.pth \
  --protocol native_resolution
```

The evaluator reads the config stored inside the checkpoint, reconstructs the
base DA2 model, inserts adapters, loads the trainable payload, and then evaluates
using the selected protocol.

Intermediate refactor schemas such as `finetune.method` and `checkpointing` are
not supported by the final canonical path; use `tuning`, `base`, and
`checkpoint`.
