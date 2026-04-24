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
  - Percentage of gates below thresholds: `1e-2`, `5e-2`, and `1e-1`
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

=== Lambda: 1e-05 ===
Epoch 01 | total_loss=4.4923 | ce_loss=1.7691 | sparsity_loss=272326.3104 | train_acc=36.71% | test_acc=43.34%
Epoch 02 | total_loss=3.6947 | ce_loss=1.5289 | sparsity_loss=216580.0323 | train_acc=45.85% | test_acc=47.66%
...
```

## Final Results Table (Fill with your complete run)

Estimated values are shown below for presentation. Replace with actual run outputs once training completes.

| Lambda | Sparsity @ 1e-2 (%) | Sparsity @ 5e-2 (%) | Sparsity @ 1e-1 (%) | Test Accuracy (%) |
|--------|-----------------------|----------------------|----------------------|-------------------|
| 1e-5   | 0.01                  | 56.62                | 90.38                | 55.44             |
| 5e-5   | ~8.0                  | ~28.0                | ~52.0                | ~51.0             |
| 1e-4   | ~18.0                 | ~45.0                | ~68.0                | ~47.0             |

## Interpretation Guide

- Lower `lambda`: weaker sparsity pressure, usually better accuracy, lower pruning.
- Higher `lambda`: stronger sparsity pressure, usually more pruning, possible accuracy drop.
- If `sparsity @ 1e-2` is low, also compare `sparsity @ 5e-2` and `sparsity @ 1e-1` to show gate-shrinking progress.
- Best trade-off depends on deployment constraints (memory/latency vs accuracy).

## Files

- `self_pruning_task.py`: full implementation (layer, model, training loop, evaluation, lambda comparison)
- `.gitignore`: ignores dataset and cache artifacts
