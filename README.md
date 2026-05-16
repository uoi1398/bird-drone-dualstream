# Bird-Drone Dual-Stream Classification

This project implements a preliminary proof-of-concept experiment for bird-drone video clip classification. The goal is to test whether optical-flow-based motion cues can compensate for degraded RGB appearance cues.

## Task

Binary video clip classification:

- Bird
- Drone

The experiment uses visible-light videos from DroneDetectionThesis / Drone-detection-dataset. The original dataset does not provide an official binary classification split, so we construct a fixed train / validation / test split.

## Dataset Split

Total: 165 video clips

| Split | Bird | Drone | Total |
|---|---:|---:|---:|
| Train | 36 | 79 | 115 |
| Validation | 7 | 18 | 25 |
| Test | 8 | 17 | 25 |

Random seed: 42.

## Models

1. RGB-only: ResNet-18 + temporal average pooling
2. Flow-only: ResNet-18 + temporal average pooling
3. RGB+Flow late fusion

Fusion:

P_fusion = alpha * P_RGB + (1 - alpha) * P_Flow

Candidate alpha values: 0.1, 0.2, 0.3, 0.4, 0.5.

## Evaluation Conditions

1. Clean
2. RGB-degraded / Motion-preserved
3. All-degraded

## Main Results

| Condition | Selected alpha | RGB Acc | Flow Acc | Fusion Acc | RGB Macro-F1 | Flow Macro-F1 | Fusion Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| RGB-degraded / Motion-preserved | 0.4 | 36.0% | 100.0% | 100.0% | 30.6% | 100.0% | 100.0% |
| All-degraded | 0.4 | 36.0% | 72.0% | 76.0% | 30.6% | 66.7% | 75.6% |

## How to Run

```bash
pip install -r requirements.txt

python src/train.py --config configs/rgb.yaml
python src/train.py --config configs/flow.yaml
python src/evaluate.py --config configs/fusion.yaml