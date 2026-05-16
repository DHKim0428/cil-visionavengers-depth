# DA2 Fine-Tuning Refactor Plan

## Why this document exists

This document records the design decisions for the Depth Anything V2 (DA2)
refactor before implementation starts in earnest.  The goal is to preserve the
reasoning behind the refactor, not only the final file layout, so that later
changes remain coherent even after several rounds of experimentation.

The branch for this work is intended to be a refactor branch, not a one-off LoRA
experiment branch.  LoRA is an important target use case, but the larger goal is
to turn the current ad hoc DA2 experimentation path into a reproducible,
configurable, and extensible training subsystem.

## Current high-level diagnosis

The repository already contains useful DA2 work, but it grew through separate
experiments:

- there are two independent DA2 fine-tuning scripts;
- there are multiple DA2 dataset/evaluation assumptions;
- model and checkpoint paths are hard-coded to individual users;
- there is no repository-owned DA2 setup procedure;
- experiment defaults are spread across Python scripts and shell wrappers;
- augmentation, supervision policy, model preprocessing, and training logic are
  not yet separated cleanly.

The refactor should preserve useful existing behavior while removing accidental
divergence.

## Decisions already made

### 1. DA2 storage and setup on the student cluster

The repository itself lives at:

```text
~/workspace/cil-visionavengers-depth
```

DA2 upstream code and checkpoints should be reproducible from one setup
script, but they live in different places for different reasons:

```text
~/workspace/cil-visionavengers-depth/
└── external/
    └── Depth-Anything-V2/              # upstream source checkout, gitignored

/work/scratch/$USER/cil-visionavengers-depth/
└── models/
    └── da2/
        ├── depth_anything_v2_vits.pth
        ├── depth_anything_v2_vitb.pth
        └── ...
```

The DA2 source checkout stays inside this repository tree so later code can use
a stable repo-relative path during partial fine-tuning.  Large checkpoint files
stay in scratch.  The setup script should be idempotent: rerunning it keeps an
existing upstream checkout and already-downloaded checkpoints.

Project-owned Python dependency installation should be handled earlier in the
repository setup flow through the repository `requirements.txt`, not hidden
inside the DA2 asset setup script.  Installing the DA2 upstream repo's own
`requirements.txt` is intentionally deferred unless later partial fine-tuning
smoke tests reveal that the course environment plus project requirements are not
enough.

### 2. DA2 model family in scope

The current repository uses the **relative-depth** DA2 family, not the separate
metric-depth DA2 models.  The current code already references:

- `vits`
- `vitb`
- `vitl`

LoRA and fine-tuning work should continue on this relative-depth path first.
Metric-depth models are a separate family and are not part of the first refactor
pass.

### 3. DA2 fine-tuning scripts should be unified

The current two DA2 fine-tuning scripts contain complementary strengths:

- `fine-tune/finetune_depth_anything_sirmse.py`
  - AMP
  - gradient accumulation
  - validation image logging
- `fine-tune/train_cil.py`
  - encoder selection
  - partial fine-tuning strategies
  - resume support
  - structured run logging

The long-term target is one canonical DA2 trainer, preferably named around the
task rather than one specific method, for example:

```text
scripts/train_da2.py
```

LoRA should eventually be one supported fine-tuning method of that trainer, not
the reason to create a third diverging DA2 script.

### 4. Both existing dataset pipelines should remain supported for now

The current DA2 work contains two different dataset/evaluation contracts written
by different contributors:

- a legacy square-resize pipeline;
- a DPT-style/native-resolution pipeline.

During the refactor, both should remain selectable.  The refactor should make
the choice explicit rather than deciding immediately which is better.  The
relative merits of the two pipelines should be revisited only after the common
infrastructure is in place.

### 5. Evaluation protocol must be explicit

The intended canonical comparison principle is:

1. keep ground-truth depth on its native grid when possible;
2. let the model operate on its own input grid;
3. resize the prediction back to the ground-truth grid;
4. compare there with a consistent valid mask and metric implementation.

The refactor should also preserve a legacy square evaluation mode for result
reproduction, but the distinction between evaluation protocols must be visible
in configuration and logs.

### 6. Repository requirements should be explicit

The student cluster provides the heavy shared runtime through the course Conda
environments, including the PyTorch stack.  The repository should still own a
small `requirements.txt` for extra project dependencies layered on top of that
environment.  The agreed initial policy is:

- rely on the course environment for `torch` and `torchvision`;
- keep repository extras such as `numpy`, `opencv-python`, `Pillow`,
  `matplotlib`, `tqdm`, `tensorboard`, and `wandb` in `requirements.txt`;
- install repository requirements before DA2 asset setup;
- defer installing the DA2 upstream repo requirements unless a later partial
  fine-tuning smoke test proves they are needed.

### 7. Configuration should move out of scripts

Experiment-defining values should live in config files rather than being
duplicated across Python defaults and shell wrappers.  The first pass should be
simple rather than over-engineered, for example:

```text
configs/
├── experiments/
└── augmentations/
```

Configs should describe:

- model family and encoder;
- tuning mode and base trainable scope;
- train dataset pipeline;
- evaluation protocol;
- augmentation preset;
- supervision options;
- optimization settings;
- runtime defaults.

Machine-specific absolute paths should not be hard-coded into shared configs.
They should come from a setup convention, environment variables, or CLI
overrides.

### 8. Weights & Biases should become the canonical logging backend

The refactored training path should use **Weights & Biases (W&B)** as the
default experiment tracker rather than treating logging as an afterthought.
Current scripts mix stdout logging, TensorBoard, hand-written text summaries,
and ad hoc checkpoint folders.  The canonical path should standardize on the
shared W&B destination `cil-visionavengers/cil-visionavengers-depth` for:

- scalar metrics;
- hyperparameters/config snapshots;
- run names and tags;
- selected qualitative validation images;
- references to checkpoints and important artifacts.

Local text logs can still exist when operationally useful, but W&B should be
the default and expected path for new experiments.  Setup documentation must
therefore include W&B installation/login expectations for the student cluster.

The intended user experience is:

1. documentation tells users to run `wandb login` before launching training;
2. canonical training entrypoints check login state at startup;
3. if login is missing, they emit a clear warning telling the user to run
   `wandb login` so the experiment is tracked properly;
4. logging remains a visible first-class concern rather than silently degrading
   without explanation.

### 9. Teacher masks are supervision, not augmentation

Teacher reliability masks should conceptually be treated as supervision policy.
Operationally:

1. raw loading reads RGB, depth, dataset validity mask, and optional teacher
   mask;
2. paired spatial transforms move all spatial tensors together;
3. the final training mask is composed afterward, for example
   `dataset_valid_mask & teacher_mask`;
4. model-specific preprocessing happens later.

This keeps alignment correct without conflating mask-based supervision with data
augmentation.

### 10. Existing code should not be moved to `legacy/` yet

The branch was created specifically so refactoring can happen safely while the
current code remains available as a reference implementation.  Existing code
should stay in place until the canonical path has feature parity and has been
validated.  Only superseded old entrypoints should later move to `legacy/`.

## Target conceptual separation

The refactor should move toward the following responsibilities:

### Raw sample layer

- read RGB/depth files;
- load dataset validity masks;
- optionally load teacher masks;
- remain model-agnostic.

### Paired spatial transform layer

- horizontal flip;
- rotation;
- crop;
- geometry-aware tilt;
- any operation that must keep RGB, depth, and masks aligned.

### Supervision layer

- combine native validity and teacher reliability masks;
- define what pixels contribute to the loss.

### Model-specific preprocessing layer

- DA2/DPT preprocessing;
- U-Net preprocessing;
- output target representation decisions.

### Training utilities

- losses;
- metrics;
- checkpoint policy;
- W&B logging;
- split management;
- seed handling.

### Thin entrypoints

- `train_da2.py`
- `eval_da2.py`
- `train_unet.py`

The goal is not to force every model through a single over-generalized engine.
Prediction semantics differ across the current models, so shared utilities are
preferred over premature abstraction.

## Proposed target structure

```text
configs/
  experiments/
  augmentations/

dataset/
  raw_cil.py
  da2_pipelines.py
  unet_pipelines.py
  augmentations.py
  loaders.py

models/
  da2.py
  unet.py
  adapters.py

training/
  losses.py
  metrics.py
  checkpoints.py
  splits.py

scripts/
  setup_da2.sh
  train_da2.py
  eval_da2.py
  train_unet.py

docs/
  da2_setup.md
  da2_current_state.md
  da2_finetuning.md
```

This is a destination map, not a requirement to move every file immediately.

## Refactor phases

### Phase 0 — Document the current state

Goal: freeze the current experimental contracts before changing them.

Deliverables:

- DA2 current-state document;
- table of current scripts and responsibilities;
- table of current dataset pipelines;
- table of current evaluation protocols;
- record of currently used DA2 variants and hard-coded paths.

### Phase 1 — Reproducible DA2 setup

Goal: make the DA2 dependency reproducible on the student cluster.

Deliverables:

- `requirements.txt` for repository extras on top of the course environment;
- `scripts/setup_da2.sh`;
- documented repo-local external checkout path;
- documented scratch checkpoint layout;
- documented environment/path contract;
- a clearly separated repository environment-setup path before DA2 asset setup;
- documented W&B setup/login expectation for new experiments;
- documented startup warning policy when W&B is not logged in;
- removal plan for personal absolute paths;
- a real DA2 asset setup smoke test before moving to Phase 2.

Before Phase 1 is considered fully settled, the repository should separately
choose how project-owned Python dependencies are declared and installed.  The
preferred user-facing order is:

1. activate the project environment;
2. install the repository requirements;
3. log in to W&B;
4. run the DA2 asset setup script.

This policy has now been chosen for the refactor: project requirements are
installed explicitly from the repository root, while `setup_da2.sh` handles only
DA2 source checkout and checkpoint assets.

### Phase 2 — Configuration layer

Goal: move experiment meaning out of scripts.

Deliverables:

- initial config schema;
- representative DA2 experiment configs;
- representative augmentation configs;
- logging config defaults, including W&B entity/project/name/tag policy;
- documented CLI override policy.

Implemented in Phase 2:

- `configs/README.md`;
- DA2 experiment configs for the existing full, decoder, and
  `refinenets_output` paths;
- augmentation presets for `none`, `basic`, and `tilt_geometry`;
- `docs/configuration.md`;
- YAML parse smoke tests for all added configs.

### Phase 3 — Data layer separation

Goal: support both current DA2 pipelines without embedding data orchestration in
training scripts.

Deliverables:

- dataloader factory design;
- explicit selection of legacy square vs DPT-style pipeline;
- split management outside dataset classes;
- teacher-mask handling modeled as supervision.

Deferred on purpose:

- deciding which pipeline is ultimately better.

Implemented in Phase 3:

- `dataset/raw_cil.py` centralizes raw CIL sample discovery and RGB/depth loading;
- `dataset/supervision.py` centralizes dataset-valid-mask creation, optional
  teacher-mask loading, and final supervision-mask composition;
- `dataset/da2_pipelines.py` introduces explicit `legacy_square` and
  `dpt_native` dataset implementations with a shared returned sample contract:
  `image`, `depth`, `valid_mask`, and `name`;
- `training/splits.py` moves deterministic filename-based train/val splitting
  outside dataset classes;
- `dataset/loaders.py` adds a DA2 dataloader factory that chooses the pipeline
  explicitly and keeps teacher masks training-only;
- `dataset/transform.py` now propagates `dataset_valid_mask` and `teacher_mask`
  through paired spatial transforms before supervision masks are composed.

This phase intentionally leaves the old DA2 entrypoints untouched.  The new data
layer is the path intended for the future canonical trainer, while the old code
remains available for reproduction until later phases.

### Phase 4 — Unified DA2 trainer

Goal: merge the two existing DA2 fine-tuning paths.

Deliverables:

- one canonical DA2 trainer;
- support for current fine-tuning strategies;
- AMP, gradient accumulation, resume, logging, and visualization under one path;
- W&B as the default logging backend for new runs;
- documentation of supported options.

Implemented in Phase 4:

- `scripts/train_da2.py` is the new config-driven canonical DA2 trainer;
- `--config` defaults to `configs/experiments/da2_vits_refinenets_output.yaml`;
- minimal CLI overrides exist for run names, paths, smoke/debug sizing, resume,
  and checkpoint selection;
- current base trainable scopes are supported: `full`, `decoder`, and
  `refinenets_output`;
- DA2 model loading and parameter-freezing logic lives in `models/da2.py`;
- DA2 siRMSE loss/eval logic lives in `training/da2_losses.py`;
- YAML loading and environment-variable expansion live in `training/config.py`;
- the trainer uses the Phase 3 dataloader factory and logs effective configs,
  checkpoints, scalar metrics, and validation image grids;
- W&B is the default backend, with a clear warning if login is missing;
- `docs/da2_finetuning.md` documents the new entrypoint and supported options.

Validation completed for Phase 4 with static compilation, default-config
dry-run, a real DA2 helper setup smoke test, and a tiny one-epoch trainer-loop
smoke run using `vits`, `img_size=56`, `max_samples=2`, and W&B logging.

### Phase 5 — Evaluation unification

Goal: make DA2 comparisons fair and explicit.

Deliverables:

- canonical native-resolution evaluation protocol;
- explicit legacy evaluation mode;
- metadata logging of train pipeline and eval protocol.

Implemented in Phase 5:

- `scripts/eval_da2.py` is the canonical DA2 evaluation entrypoint;
- `training/da2_eval.py` centralizes evaluation helpers, prediction resizing,
  raw-infer evaluation, visualization export, and summary statistics;
- `native_resolution` evaluates transformed DA2 tensors while resizing
  predictions back to the native GT grid;
- `legacy_square` preserves square-grid validation for reproduction;
- `raw_infer_native` preserves the older zero-shot `model.infer_image(...)`
  evaluation path from `comparison/script/evaluate_depth_anything.py`;
- evaluation outputs include `effective_config.yaml`, `eval_summary.json`,
  `eval_summary.txt`, optional visualizations, checkpoint metadata, data
  pipeline metadata, eval protocol metadata, sample names, and per-sample scores;
- `models/da2.py` now accepts official DA2 checkpoints, `best.pth`, and
  canonical `latest.pth` checkpoints for evaluation.

Validation completed for Phase 5 with static compilation, evaluator dry-run, and
tiny smoke evaluations for `native_resolution`, `legacy_square`, and
`raw_infer_native` including one raw-infer visualization export.

### Phase 6 — LoRA / adapter support

Goal: add LoRA only after the infrastructure is stable, while keeping it
decoupled from the base fine-tuning strategy.  LoRA should be an adapter axis,
not a mutually exclusive replacement for base parameter tuning.

Design decisions before implementation:

- `base.trainable_scope` controls which original DA2 parameters are trainable;
- `adapter` controls whether LoRA modules are inserted and where;
- LoRA should support combinations such as LoRA-only, decoder+LoRA,
  refinenets_output+LoRA, and full+LoRA rather than hard-coding one pairing;
- target selection should be configurable, with an important mode where LoRA is
  applied automatically within the currently trainable scope;
- adapter hyperparameters such as rank, alpha, dropout, and target mode belong
  in config.

Checkpoint policy:

- storage is a constraint on the student cluster, so full-model checkpoint saving
  must **not** be the default for adapter experiments;
- the default checkpoint representation should be `trainable_only`;
- an eval/resume checkpoint is reconstructed from:

  ```text
  base DA2 checkpoint + effective_config.yaml + trainable checkpoint payload
  ```

- LoRA-only checkpoints should store adapter weights plus training metadata;
- LoRA + base fine-tuning checkpoints should store adapter weights plus only the
  changed original DA2 trainable parameters;
- full-model saving should be explicit opt-in, for example through
  `checkpoint.save_policy: full_model`, because it is much larger;
- `latest.pth` may include optimizer/scaler state for resume, while `best.pth`
  should remain as small as possible for evaluation/reuse.

Deliverables:

- adapter injection utilities;
- LoRA config options;
- LoRA-only and LoRA-plus-decoder experiment support;
- configurable LoRA target modes such as `trainable_scope`, `decoder`,
  `all_linear`, and possibly regex-based selection;
- trainable parameter accounting that separates original DA2 params from adapter
  params;
- trainable-only checkpoint save/load utilities;
- eval/resume logic that reconstructs models from base checkpoint + config +
  trainable checkpoint payload;
- explicit docs on the minimal checkpoint format and storage tradeoff.

Implemented in Phase 6:

- `models/adapters.py` adds lightweight LoRA wrappers for `nn.Linear` and
  `nn.Conv2d(groups=1)` modules;
- LoRA is configured through `adapter` and remains independent from
  `base.trainable_scope`;
- `tuning.mode: lora` with `base.trainable_scope: frozen` enables LoRA-only training with the base DA2 model
  frozen;
- target modes currently include `trainable_scope`, `decoder`, `all_linear`, and
  `regex`;
- `training/checkpoints.py` implements `da2_trainable_checkpoint_v1`;
- `best.pth` and `latest.pth` now use trainable-only payloads by default, while
  `checkpoint.save_policy: full_model` remains available as explicit opt-in;
- `scripts/train_da2.py` injects adapters after applying the base fine-tuning
  strategy, logs base-vs-adapter trainable parameter counts, and saves minimal
  checkpoints;
- `scripts/eval_da2.py` can reconstruct and evaluate trainable-only adapter
  checkpoints from base checkpoint + stored config + trainable payload;
- new configs were added for LoRA-only decoder training and
  refinenets_output+LoRA training;
- `docs/da2_adapters.md` documents LoRA target modes and checkpoint format.

Validation completed for Phase 6 with static compilation, LoRA config dry-run,
LoRA-only one-epoch trainer smoke, checkpoint payload inspection, trainable-only
checkpoint evaluation smoke, and refinenets_output+LoRA target-selection smoke.

### Phase 7 — Shared augmentation experiments

Goal: connect existing augmentation research to DA2 fine-tuning.

Deliverables:

- model-agnostic paired augmentation presets;
- reuse of augmentation definitions across U-Net and DA2;
- later comparison of dataset/augmentation choices under a stable trainer.

Implemented in Phase 7:

- `dataset/augmentations.py` now owns the shared `DepthAugmentation`
  implementation and a config-to-augmentation builder;
- the existing U-Net dataset imports `DepthAugmentation` from the shared module;
- `training/config.py` resolves `augmentation.preset` from
  `configs/augmentations/<preset>.yaml`;
- `dataset/loaders.py` passes the resolved augmentation config into the DA2
  training dataset only;
- `LegacySquareDA2Dataset` and `DPTNativeDA2Dataset` apply the shared paired
  augmentation stack before ImageNet normalization;
- validation and evaluation datasets remain unaugmented;
- this phase intentionally does **not** decide which preset is best.

Validation completed for Phase 7 with static compilation, trainer dry-run, and a
synthetic dataloader smoke test covering `none`, `basic`, and `tilt_geometry` on
both `legacy_square` and `dpt_native`.

### Phase 8 — Move superseded entrypoints to legacy

Goal: clean up only after the canonical path is proven.

Deliverables:

- move only truly superseded entrypoints into `legacy/`;
- keep documentation of how to reproduce older experiments;
- avoid a big-bang rewrite.

Implemented in Phase 8:

- audited the old DA2 scripts against the canonical modular path before moving
  anything;
- moved superseded DA2 entrypoints and personal wrappers to `legacy/da2/`:
  - `train_cil.py`;
  - `finetune_depth_anything_sirmse.py`;
  - `evaluate_depth_anything.py`;
  - `run_exp.sh`;
  - `submit_train_cil.sh`;
- kept historical result files in `comparison/results/` untouched because they
  are artifacts rather than runnable entrypoints;
- added `legacy/da2/README.md` with a migration audit mapping old behavior to
  canonical replacements;
- canonical DA2 training/evaluation now lives under `scripts/train_da2.py`,
  `scripts/eval_da2.py`, `configs/`, `dataset/`, `models/`, and `training/`.

Functionality audit summary:

- `fine-tune/train_cil.py` is covered by config-driven encoder selection,
  `base.trainable_scope`, poly LR, resume, trainable-only checkpoints, and W&B
  logging in `scripts/train_da2.py`;
- `fine-tune/finetune_depth_anything_sirmse.py` is covered by AMP, gradient
  accumulation, validation image logging, `legacy_square`, and full/partial DA2
  configs in `scripts/train_da2.py`;
- `comparison/script/evaluate_depth_anything.py` is covered by
  `scripts/eval_da2.py --protocol raw_infer_native`;
- old shell wrappers were personal cluster launchers with hard-coded home paths,
  so they are preserved only as historical examples.

Validation completed for Phase 8 with canonical module static compilation plus
`train_da2.py` and `eval_da2.py` dry-runs after the legacy move.

### Phase 9 — Legacy parity and supervision/filtering audit

Goal: verify that the canonical DA2 path reproduces the intent of legacy
experiments closely enough to trust it, and isolate any remaining supervision or
filtering pipeline work.

Phase 9A — DA2 legacy result parity:

- move DA2 result artifacts produced by old scripts into `legacy/da2/results/`;
- map each legacy result to a canonical config/evaluation command;
- compare cheap invariants first, such as trainable parameter counts, protocol,
  sample count/fraction, valid-mask contract, and checkpoint variant;
- run full metric parity only when the required checkpoint/split information is
  available;
- if metrics diverge, diagnose pipeline/protocol/mask/checkpoint differences
  before changing canonical code.

Phase 9B — filtering / teacher-mask supervision audit:

- audit DA3 reliability-mask generation and validation scripts;
- connect precomputed mask artifacts cleanly to `supervision.teacher_mask`;
- document mask file format, split assumptions, and filtering policy;
- decide what belongs in the canonical path versus experiment/debug scripts.

Implemented so far in Phase 9A:

- moved historical DA2 result artifacts from `comparison/results/` to
  `legacy/da2/results/`;
- added `legacy/da2/results/README.md`;
- added `docs/da2_legacy_parity.md` with canonical command skeletons and a
  parity checklist;
- verified canonical module compilation and dry-ran the mapped `vits`
  refinenets training skeleton plus `vits`/`vitb` raw-infer evaluation
  skeletons;
- added generic SLURM wrappers `scripts/slurm/train_da2.sbatch` and
  `scripts/slurm/eval_da2.sbatch` for real 9A parity/W&B validation runs;
- dry-ran both SLURM wrappers directly with `DRY_RUN=1` to confirm config/env
  overrides are forwarded to the canonical Python entrypoints.
- completed the first real SLURM validation round for DA2 smoke/full-count
  zero-shot evaluation: W&B/log/checkpoint plumbing works, `vits` and `vitb`
  raw/native eval run at the intended sample counts, and zero-shot metric
  discrepancies versus legacy are now documented rather than treated as solved.
- started longer Phase 9A training validation: `phase9a_refinenets_full` is
  running and matches legacy startup invariants, while decoder and VITB
  legacy-square training jobs are queued.
- added Phase 9B/current-cleanup planning docs:
  `docs/da3_teacher_mask_current_state.md` audits the DA3 reliability-mask path,
  and `docs/repo_cleanup_plan.md` defines the route toward one canonical train
  entrypoint, one eval entrypoint, shared augmentation policy, and modular model
  engines.
- consolidated remaining work in `docs/remaining_refactor_plan.md`, with
  siRMSE-only unified evaluation as the next organizing principle before further
  comparison runs.

## Deferred decisions

These questions are intentionally postponed until after the infrastructure is
clean enough to compare them fairly:

- whether the legacy square pipeline or the DPT-style pipeline should become the
  long-term default;
- whether DA2 fine-tuning should use `vits` or `vitb` as the default everyday
  research model;
- which augmentation presets actually help DA2 rather than only U-Net;
- whether LoRA alone is sufficient or should usually be paired with decoder
  fine-tuning.

## Working principle for the refactor

Prefer:

- explicit contracts over hidden conventions;
- reproducibility over personal machine assumptions;
- thin entrypoints over copied scripts;
- feature parity before cleanup;
- staged migration over a rewrite.



### Phase 6 naming/schema cleanup

After LoRA support landed, the config naming was clarified to avoid implying
that LoRA-only runs are not fine-tuning:

- `finetune.method` was retired and replaced by two explicit axes:
  - `tuning.mode`: `base`, `lora`, or `mixed`;
  - `base.trainable_scope`: `frozen`, `full`, `decoder`, or `refinenets_output`;
- `checkpointing` was retired in favor of `checkpoint`;
- `training/checkpointing.py` was renamed to `training/checkpoints.py`;
- intermediate refactor configs/checkpoints using retired schema keys are not
  supported by the final canonical path.
