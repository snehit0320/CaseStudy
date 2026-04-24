# The Self-Pruning Neural Network (Case Study)

This project implements a feed-forward image classifier with a built-in self-pruning mechanism for CIFAR-10.
Instead of pruning after training, each weight is paired with a learnable gate so the model learns which connections to suppress during optimization.

## What is implemented

- Custom `PrunableLinear(in_features, out_features)` layer
  - Learnable `weight`, `bias`, and `gate_scores` (same shape as `weight`)
  - Gates are computed as `sigmoid(gate_scores)` in `[0, 1]`
  - Effective weights: `pruned_weights = weight * gates`
- Neural network built entirely with prunable layers
- Custom loss:
  - `TotalLoss = CrossEntropyLoss + lambda * SparsityLoss`
  - `SparsityLoss` is the L1 norm over all gate values
- Training and evaluation on CIFAR-10
- Sparsity report:
  - Percentage of gates below threshold (`gate < 1e-2`)
- Lambda sweep with 3 values to show sparsity vs accuracy trade-off

## Requirements

- Python 3.10+
- PyTorch
- torchvision

Install dependencies:

```bash
pip install torch torchvision
```

## Run

```bash
python self_pruning_task.py
```

The script prints live logs for each epoch and each lambda configuration.

## Sample Console Output

```text
Starting self-pruning training on cpu
This run compares 3 lambda values: low, medium, high.
Files already downloaded and verified
Files already downloaded and verified

=== Lambda: 1e-06 ===
Epoch 01 | total_loss=3.1272 | ce_loss=1.6340 | sparsity_loss=1493249.2802 | train_acc=42.09% | test_acc=45.66%
Epoch 02 | total_loss=2.8920 | ce_loss=1.4188 | sparsity_loss=1473194.4825 | train_acc=50.08% | test_acc=49.80%
...
```

## Final Results Table (Fill with your complete run)

After the script completes all lambda runs, record the final summary here:

| Lambda | Sparsity (%) | Test Accuracy (%) |
|--------|---------------|-------------------|
| 1e-6   |               |                   |
| 1e-5   |               |                   |
| 1e-4   |               |                   |

## Interpretation Guide

- Lower `lambda`: weaker sparsity pressure, usually better accuracy, lower pruning.
- Higher `lambda`: stronger sparsity pressure, usually more pruning, possible accuracy drop.
- Best trade-off depends on deployment constraints (memory/latency vs accuracy).

## Files

- `self_pruning_task.py`: full implementation (layer, model, training loop, evaluation, lambda comparison)
- `.gitignore`: ignores dataset and cache artifacts
