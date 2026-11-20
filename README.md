# Distracted Driver Detection using PyTorch and Transfer Learning

Classifies what a driver is doing from a single in-cabin camera photo (10 classes) using a baseline CNN and an ImageNet-pretrained **ResNet18** fine-tuned on the Kaggle *State Farm Distracted Driver Detection* dataset.

**Author:** Arpit Raj
**Supervisor:** Dr. Ankush Jain
**Institution:** Netaji Subhas University of Technology (NSUT), Delhi

> A companion paper (`Distracted_Driver_Detection_IEEE_Paper.pdf`) describes the methodology, results and systems analysis.

---

## Highlights

- Supervised image classification with 10 driver-behaviour classes
- Baseline residual CNN vs. **ResNet18 transfer learning** (final model)
- Best validation accuracy **99.11%** (epoch 5); **97.68%** after the first epoch
- Trains in about **19 minutes** for 6 epochs on an Intel Core i5, 16 GB RAM, GTX 1650
- Kaggle-style output: per-class probabilities in `final_sub.csv`

## Classes

| ID | Behaviour | ID | Behaviour |
|----|-----------|----|-----------|
| c0 | Safe driving | c5 | Operating the radio |
| c1 | Texting, right hand | c6 | Drinking |
| c2 | Phoning, right hand | c7 | Reaching behind |
| c3 | Texting, left hand | c8 | Hair and make-up |
| c4 | Phoning, left hand | c9 | Talking to passenger |

## Dataset

[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (Kaggle).

- ~22,400 labelled training images, ~79,700 unlabelled test images
- Colour images, 640 x 480 (tensor shape `3 x 480 x 640`)
- Classes are close to balanced
- `driver_imgs_list.csv` contains a `subject` (driver ID) column

The dataset is **not included** in this repository. Download it from Kaggle (a Kaggle account and acceptance of the competition rules are required) and place it so that the layout is:

```
data/
└── imgs/
    ├── train/
    │   ├── c0/ ... c9/
    └── test/
```

## Pipeline

1. Load and explore data (class counts, sample images)
2. Preprocess: resize to 150 x 150, augment (rotation up to 10 degrees, shift up to 10%, zoom up to 1.2x), normalise with ImageNet mean/std
3. Split 80/20 into training (~17,900) and validation (~4,500) sets, batch size 32
4. Train a baseline CNN
5. Fine-tune ResNet18 (1000-way head replaced with a 10-way layer; Adam; cross-entropy; 6 epochs)
6. Evaluate and plot loss/accuracy curves
7. Predict on the test images and write `final_sub.csv` (`img, c0 ... c9` probabilities)

## Results (ResNet18, validation split)

| Epoch | Train loss | Train acc (%) | Val loss | Val acc (%) |
|------:|-----------:|--------------:|---------:|------------:|
| 1 | 1.5751 | 84.84 | 0.1036 | 97.68 |
| 2 | 0.0607 | 98.47 | 0.0513 | 98.39 |
| 3 | 0.0261 | 99.21 | 0.1037 | 96.86 |
| 4 | 0.0300 | 99.10 | 0.0525 | 98.53 |
| 5 | 0.0165 | 99.41 | **0.0409** | **99.11** |
| 6 | 0.0231 | 99.33 | 0.0722 | 97.99 |

Training throughput, derived from wall-clock time, is roughly 94 images/s. Inference latency and GPU utilisation were **not measured**; the paper gives analytical projections only.

## Setup

Requirements: Python 3.9+, a CUDA-capable GPU is recommended.

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate

# Install PyTorch with CUDA from https://pytorch.org (match your CUDA version), then:
pip install torchvision pandas numpy matplotlib pillow tqdm jupyter
```

> If you see `Torch not compiled with CUDA enabled`, a CPU-only PyTorch build is installed. Reinstall PyTorch from the CUDA index inside the same virtual environment.

## Usage

```bash
jupyter notebook
```

Open the project notebook, set `train_path` / `test_path` to your local dataset folders, and run all cells. Output: trained model and `final_sub.csv`.

## Known Issues and How They Were Fixed

### Resolved during development

| # | Issue | Cause | Fix |
|---|-------|-------|-----|
| 1 | Out-of-memory crash on 480 x 640 images | Images and model too large for memory | Resized every image to 150 x 150 and used a smaller CNN |
| 2 | Notebook paths failed outside Kaggle | `/kaggle/input/...` exists only on Kaggle servers | Downloaded the dataset and switched to local paths (`data/imgs/train`) |
| 3 | `KeyError: 'train\c0'` on Windows | `glob` returns backslash paths, so splitting on `/` gave a wrong class name | Used `os.path.basename(...)` instead of `split("/")` |
| 4 | `Torch not compiled with CUDA enabled` | CPU-only PyTorch was installed, so the GPU was unused | Reinstalled PyTorch from the CUDA index (cu126) in the same virtual environment |
| 5 | ResNet18 stuck at ~11% accuracy (chance level, loss ~2.5) | The optimiser from the earlier CNN was reused, so ResNet weights were never updated | Created a new optimiser from the ResNet's own parameters; epoch 1 then reached 97.68% |

> **Lesson:** if accuracy equals chance and the loss does not change, check first that the optimiser is attached to the right model.

### Identified issues and planned fixes

These were found during review of the final notebook; the fix is known but has not yet been applied.

| Issue | Why it matters | Planned fix |
|-------|----------------|-------------|
| Same drivers in training and validation (~26 drivers, random split) | Validation accuracy is probably optimistic | Split by the `subject` column in `driver_imgs_list.csv` |
| Horizontal flip augmentation | Swaps left/right meaning (c1/c3, c2/c4) while keeping the old label | Remove `RandomHorizontalFlip` |
| Shared dataset object after `random_split` | The validation transform can overwrite the training transform, disabling augmentation | Create two `ImageFolder` objects (one per transform) and split the indices |
| Last epoch kept instead of the best | Epoch 6 (97.99%) is saved instead of epoch 5 (99.11%) | Save a checkpoint at the highest validation accuracy |
| No per-class metrics | Hard to see which behaviours are confused | Add a confusion matrix and per-class precision/recall |

## Future work

- Driver-wise split, confusion matrix, per-class metrics
- Larger input (224 x 224), learning-rate schedule, other backbones (ResNet50, EfficientNet)
- Grad-CAM explanations
- Real-time deployment on a camera stream with benchmarked latency and alerting

## Acknowledgements

State Farm and Kaggle for the dataset; PyTorch/torchvision for the framework and pretrained ResNet18 weights.
