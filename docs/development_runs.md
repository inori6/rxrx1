# Development Runs

## 2026-09-01 — ResNet18 pipeline smoke test

Purpose: Verify the training pipeline.

### Parameters

- Model: ResNet18
- Pretrained: ImageNet
- Input channels: 6
- Seed: 42
- Epochs: 5
- Batch size: 32
- Optimizer: AdamW
- LR: 1e-4
- Weight decay: 1e-4
- Loss: CrossEntropyLoss

### Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 6.1029 | 0.0036 | 5.9193 | 0.0050 |
| 2 | 5.6893 | 0.0336 | 5.7142 | 0.0213 |
| 3 | 5.1755 | 0.1317 | 5.5571 | 0.0325 |
| 4 | 4.6772 | 0.3194 | 5.5024 | 0.0362 |
| 5 | 4.1799 | 0.5642 | 5.4060 | 0.0469 |

Runtime: ~1h30m

Not a formal experiment.