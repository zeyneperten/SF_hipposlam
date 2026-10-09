# Initialize a high-level RNN from an oracle controller

## Note for Zeynep

This adds a way to start a new high-level RNN experiment from **one oracle checkpoint that you choose**. The checkpoint supplies the visual encoder, DG, base sequence core, decoder, action head, critic, and observation-normalization state. Its first two trained decoder modes become the two learned modes. The high-level RNN, optimizer, and training counters start fresh. The transferred action controller is frozen by default while the high-level RNN and critic train.

In your existing `experiments/exp_ymaze_HLRNN.py` launcher, give the new experiment a unique name and add this option to its `cli` string:

```text
--oracle_init_checkpoint=/absolute/path/to/oracle_run/checkpoint_pN/best_....pth
```

Choose the policy-specific `checkpoint_pN` file yourself. The code does not rank W&B runs or download checkpoints. Use a checkpoint whose companion run config describes a four-mode oracle high-level run with DG conditioning (`concat`, `multiply`, or `sigmoid`) and either `FiLM` or `additive` decoder conditioning. The loader checks tensor shapes before training and reports the incompatible tensor if the architecture differs.

For a later fine-tuning configuration, add `--oracle_freeze_controller=False`. The source checkpoint is needed for the initial run only; a target run's own checkpoint takes priority when resuming. Compare episode length, flexibility, and `highrew_hit` with the chosen oracle and a fresh high-level run before drawing a performance conclusion. The local tests cover the transfer and replay behavior; a full training smoke test still needs your checkpoint and training environment.

## Direct CLI example

Pass an exact local oracle checkpoint to `train_hipposlam.py` when starting a **new** experiment:

```bash
python -m sf_working_directories.zeynep.dmlab.train_hipposlam \
  --env=ymaze_instr_hl \
  --experiment=hl_from_oracle_001 \
  --train_dir=/path/to/new/training \
  --oracle_init_checkpoint=/path/to/oracle_run/checkpoint_p2/best_000012345_67890.pth
```

Use the usual `ymaze` training options as needed. The initializer reads the source run's `config.json` (or `cfg.json`) and sets the matching controller architecture, DG modulation, decoder modulation, and oracle input scale. It computes the new two-mode recurrent-state size. The source must be a four-mode high-level oracle run whose decoder actually has four trained mode columns. Modes 0 and 1 are copied into the two-mode decoder. Unsupported architectures and tensor shapes fail with a named error.

The action controller and observation normalization are frozen by default; the high-level RNN and value head train. To fine-tune the inherited controller in a later run, add `--oracle_freeze_controller=False` when starting that run. This does not change an already saved run's weights or training progress.

At each frame, DG receives the mode stored in the incoming recurrent state, multiplied by the source oracle's input scale. The decoder receives the mode selected on the current frame. A mode selected at an outcome event reaches DG on the next frame. At episode reset, the incoming mode is zero until the first choice. Learner replay uses the actor's saved incoming recurrent state for DG and its recorded choice for the decoder, including at rollout boundaries.

Use a new experiment name for initialization. Resuming an existing target experiment loads its own checkpoint and optimizer; the oracle path is only used on a fresh run. Without `--oracle_init_checkpoint`, the existing high-level training configuration is unchanged.
