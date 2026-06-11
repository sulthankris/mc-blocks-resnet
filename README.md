# Minecraft Block Classification with ResNet-18

Image classification project for recognizing Minecraft block types from screenshots of individual blocks. The training pipeline uses PyTorch and the public MiDaS-60 dataset.

The model predicts one of 60 block classes, such as `oak_planks`, `diamond_ore`, `furnace`, `jungle_planks`, or `water`.

## Features

- Deterministic train/validation/test split with saved manifests.
- Exact duplicate detection by SHA-256 before splitting.
- ResNet-18 transfer learning with a replaced classification head.
- Small CNN baseline for comparison.
- Saved training history, environment metadata, resolved config, and checkpoints.
- Evaluation with top-1 accuracy, top-5 accuracy, macro-F1, weighted-F1, per-class metrics, confusion matrix, and per-image predictions.
- Matplotlib figures for distribution, training curves, confusion patterns, and prediction examples.

## Dataset

This project uses MiDaS, a Minecraft block image dataset for non-natural image classification benchmarks.

Official sources:

- RAIL Lab publication page: <https://www.raillab.org/publication/torpey-2024-midas/>
- Paper PDF: <https://www.raillab.org/publication/torpey-2024-midas/torpey-2024-midas.pdf>
- GitHub repository: <https://github.com/MinecraftDataset/MiDaS>
- OSF dataset page: <https://osf.io/whgy6/>

MiDaS-60 is described by the official sources as having 60 classes and 36,000 images. In the local run used during development, 4 exact duplicate images were removed before splitting, leaving 35,996 images.

Local split from the current manifest:

| Split | Images |
|---|---:|
| Train | 25,196 |
| Validation | 5,397 |
| Test | 5,403 |

Raw data, processed data, checkpoints, and generated reports are ignored by git.

## Model Architecture

### ResNet-18

The main model uses `torchvision.models.resnet18`.

Architecture flow:

```text
RGB image
  -> resize/crop/normalize
  -> ResNet-18 convolution stem
  -> residual blocks
  -> global average pooling
  -> linear classifier with 60 outputs
  -> logits for block classes
```

The original ImageNet classifier is replaced with:

```python
nn.Linear(resnet.fc.in_features, num_classes)
```

The model returns raw logits. Training uses `CrossEntropyLoss`, so no softmax layer is added to the model. Probabilities are only computed during evaluation and prediction display.

Why ResNet-18 is used:

- It is small enough to train quickly on a consumer GPU.
- Residual connections make it stable for deeper feature extraction than a plain CNN.
- ImageNet pretrained filters are useful for edges, colors, local textures, and repeated block patterns.
- The final classifier can be replaced cleanly for the 60 MiDaS classes.

### Small CNN Baseline

The baseline model is a compact CNN trained from scratch:

```text
Conv2d -> BatchNorm -> ReLU -> MaxPool
Conv2d -> BatchNorm -> ReLU -> MaxPool
Conv2d -> BatchNorm -> ReLU -> MaxPool
AdaptiveAvgPool2d
Linear classifier
```

This baseline is useful for checking whether transfer learning gives an improvement over a model with no pretrained features.

## Project Structure

```text
configs/              training and evaluation defaults
scripts/              command-line entry points
src/mcblockclf/       Python package code
tests/                unit tests
data/raw/             extracted MiDaS dataset, ignored by git
data/processed/       prepared ImageFolder dataset, ignored by git
data/manifests/       split manifest and class mappings
runs/                 checkpoints and training logs
reports/metrics/      evaluation outputs
reports/figures/      generated plots
```

## Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended
- NVIDIA GPU with CUDA recommended for ResNet-18 training

The code also runs on CPU, but full ResNet-18 training will be much slower.

## Setup on Windows

PowerShell:

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
pip install -r requirements-dev.txt
```

If PowerShell blocks virtual environment activation, run this in the same terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

CMD alternative:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Download MiDaS

The downloader checks whether MiDaS images already exist under `data/raw`. If no dataset is found, it prints manual OSF download instructions.

```powershell
python scripts/download_midas.py --variant small --out data/raw
```

If manual download is needed:

1. Open <https://osf.io/whgy6/>.
2. Download `MiDaS-60_small` or `MiDaS-60_large`.
3. Extract the dataset folder into `data/raw/`.
4. Run `scripts/download_midas.py` again to write `data/raw/SOURCE.txt`.

Expected example layout:

```text
data/raw/MiDaS-60_small/train/...
data/raw/MiDaS-60_small/test/...
```

## Prepare Dataset

```powershell
python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed --manifest-dir data/manifests --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

Outputs:

- `data/manifests/class_to_idx.json`
- `data/manifests/idx_to_class.json`
- `data/manifests/split_manifest.csv`
- `data/manifests/dataset_summary.json`
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

The split is done per class so each class is represented in train, validation, and test sets. The manifest stores relative path, class name, class index, split, and SHA-256 hash.

## Smoke Test

Use a small subset to verify that the pipeline works.

```powershell
python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed_smoke --manifest-dir data/manifests_smoke --seed 42 --max-per-class 5
```

```powershell
python scripts/train.py --config configs/smoke_test.yaml --processed-dir data/processed_smoke --manifest-dir data/manifests_smoke --epochs 1 --batch-size 8 --image-size 128 --model small_cnn
```

```powershell
python scripts/evaluate.py --checkpoint runs/<smoke_run_id>/best_model.pt --processed-dir data/processed_smoke --manifest-dir data/manifests_smoke --out-dir reports/metrics_smoke
```

## Train ResNet-18

```powershell
python scripts/train.py --config configs/default.yaml --model resnet18 --pretrained true --epochs 20 --batch-size 64 --lr 0.0003 --seed 42 --amp false
```

Notes:

- `--amp false` is the safer default on the tested Windows CUDA setup.
- `best_model.pt` is selected by validation macro-F1.
- `last_model.pt` stores the last completed checkpoint.
- If training is interrupted with `Ctrl+C`, the script attempts to save `last_model.pt`; any existing `best_model.pt` remains usable.

Each run directory contains:

```text
config_resolved.yaml
environment.json
history.csv
metrics_val.json
best_model.pt
last_model.pt
logs.txt
```

## Evaluate

```powershell
python scripts/evaluate.py --checkpoint runs/<run_id>/best_model.pt --processed-dir data/processed --manifest-dir data/manifests --out-dir reports/metrics
```

Outputs:

- `reports/metrics/test_metrics.json`
- `reports/metrics/classification_report.csv`
- `reports/metrics/confusion_matrix.csv`
- `reports/metrics/predictions_test.csv`

`predictions_test.csv` contains one row per test image, including the true label, predicted label, confidence, correctness flag, and top-5 labels.

Example correct prediction from a local run:

```text
path,true_label,pred_label,confidence,correct,top5_labels
test/acacia_log/00001_11f5382e34a6.png,acacia_log,acacia_log,0.999990,True,acacia_log;chest;torch;oak_log;water
```

Example wrong prediction:

```text
path,true_label,pred_label,confidence,correct,top5_labels
test/jungle_planks/00042_f27119d43baa.png,jungle_planks,oak_planks,0.987255,False,oak_planks;jungle_planks;spruce_planks;birch_planks;oak_log
```

## Generate Figures

```powershell
python scripts/make_figures.py --run-dir runs/<run_id> --metrics-dir reports/metrics --manifest-dir data/manifests --out-dir reports/figures
```

```powershell
python scripts/predict_examples.py --checkpoint runs/<run_id>/best_model.pt --processed-dir data/processed --manifest-dir data/manifests --out-dir reports/figures --num-samples 12 --seed 42
```

Generated figures include training curves, dataset distribution, confusion matrix summary, top misclassification pairs, prediction examples, and worst classes by F1-score.

## Local Result

Latest evaluated run:

```text
runs/20260611_113209_resnet18/best_model.pt
```

This checkpoint was selected from epoch 6 by validation macro-F1.

| Metric | Value |
|---|---:|
| Top-1 accuracy | 98.80% |
| Top-5 accuracy | 99.96% |
| Macro-F1 | 0.9879 |
| Weighted-F1 | 0.9879 |
| Test loss | 0.0383 |

Main remaining confusions in that run include `jungle_planks -> oak_planks`, `dark_oak_log -> oak_log`, and `spruce_planks -> birch_planks`.

## Run Tests

```powershell
python -m pytest
```

The test suite checks manifest columns, deterministic class indices, split coverage on a mock dataset, model output shapes, and metric output shapes.

## Limitations

- The model classifies cropped block images; it does not use game context.
- Similar wood, plank, stone, and ore textures remain the hardest cases.
- Reported metrics depend on the exact dataset files, split seed, and environment.
- ImageNet pretraining helps with low-level visual features, but the source domain is still different from Minecraft imagery.

## References

- Torpey, D. et al. (2024). *MiDaS: A Large-Scale Minecraft Dataset for Non-Natural Image Benchmarking*.
- PyTorch: <https://pytorch.org/>
- PyTorch reproducibility notes: <https://docs.pytorch.org/docs/stable/notes/randomness.html>
- Torchvision models: <https://docs.pytorch.org/vision/stable/models.html>
- scikit-learn model evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>
