# Distracted Driver Detection using PyTorch and Resnet

Classifies what a driver is doing from a single in-cabin camera photo (10 classes) using a baseline CNN and an ImageNet-pretrained **ResNet18** fine-tuned on the Kaggle *State Farm Distracted Driver Detection* dataset.

**Author:** Arpit Raj
**Supervisor:** Dr. Ankush Jain
**Institution:** Netaji Subhas University of Technology (NSUT), Delhi

---

## Highlights

- Supervised image classification with 10 driver-behaviour classes
- Baseline residual CNN vs. **ResNet18 transfer learning** (final model)
- Best validation accuracy **96.11%** (epoch 5); **97.68%** after the first epoch
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


- ~22,400 labelled training images, ~79,700 unlabelled test images
- Colour images, 640 x 480 (tensor shape `3 x 480 x 640`)
- Classes are close to balanced
- `driver_imgs_list.csv` contains a `subject` (driver ID) column

The dataset is **not included** in this repository because its of 4GB but the structure is as follows:

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


## Setup

Requirements: Python 3.9+, a CUDA-capable GPU to run locally .

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate

# Install PyTorch with CUDA from https://pytorch.org (match your CUDA version), then:
pip install torchvision pandas numpy matplotlib pillow tqdm jupyter
```

## Usage

```bash
jupyter notebook
```
Open the project notebook, set `train_path` / `test_path` to your local dataset folders, and run all cells. Output: trained model and `final_sub.csv`.

## Future work

- Driver-wise split, confusion matrix, per-class metrics
mera stream with benchmarked latency and alerting


