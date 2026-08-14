# Setup & Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA 11.8+ (optional, for GPU support)

## Installation Steps

### 1. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n multitask python=3.10
conda activate multitask
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install with the setup.py:
```bash
pip install -e .
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "from src.model import MultiTaskResNet18; print('Model import successful')"
```

## Project Structure

```
Joint-Classification-and-Segmentation/
├── README.md                 # Project description
├── setup.py                  # Package setup
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore file
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py             # MultiTaskResNet18 architecture
│   ├── data.py              # Dataset and data loading utilities
│   ├── loss.py              # Loss functions (CE + Dice)
│   └── utils.py             # Evaluation metrics and utilities
│
├── train.py                 # Main training script
├── inference.py             # Inference script
│
├── data/                    # Dataset directory (create & populate)
│   ├── train/
│   │   ├── images/
│   │   ├── labels_cls/
│   │   └── labels_seg/
│   ├── val/
│   └── test/
│
├── models/                  # Saved checkpoints
├── logs/                    # TensorBoard logs
└── configs/                 # Configuration files (optional)
```

## Dataset Preparation

Create the following directory structure in `data/`:

```
data/
├── train/
│   ├── images/              # Training images (*.jpg, *.png)
│   ├── labels_cls/          # Classification labels (*.txt files with class index)
│   └── labels_seg/          # Segmentation masks (same naming as images)
├── val/
│   ├── images/
│   ├── labels_cls/
│   └── labels_seg/
└── test/
    ├── images/
    ├── labels_cls/
    └── labels_seg/
```

### Example Label Format

**Classification labels** (labels_cls/image_001.txt):
```
5
```

**Segmentation masks** (labels_seg/image_001.png):
- PNG image where pixel values are class indices

## Training

### Basic Training

```bash
python train.py \
    --data-dir ./data \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --num-classes-cls 10 \
    --num-classes-seg 10
```

### Advanced Training with Custom Loss Weights

```bash
python train.py \
    --data-dir ./data \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --alpha 1.0 \
    --beta 1.0 \
    --log-dir ./logs \
    --checkpoint-dir ./models \
    --save-freq 10
```

### Resume Training from Checkpoint

```bash
python train.py \
    --data-dir ./data \
    --resume ./models/best_model.pth \
    --epochs 50
```

## Inference

### Run on Single Image

```bash
python inference.py \
    --model ./models/best_model.pth \
    --image path/to/test/image.jpg \
    --num-classes-cls 10 \
    --num-classes-seg 10
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch-size` | 32 | Training batch size |
| `epochs` | 50 | Number of training epochs |
| `lr` | 1e-4 | Learning rate |
| `alpha` | 1.0 | Weight for classification loss (CE) |
| `beta` | 1.0 | Weight for segmentation loss (Dice) |
| `weight-decay` | 1e-5 | L2 regularization |
| `num-workers` | 4 | Data loading workers |
| `num-classes-cls` | 10 | Number of classification classes |
| `num-classes-seg` | 10 | Number of segmentation classes |

## Expected Results

Based on the README:
- **Classification Accuracy**: ~88.1%
- **Segmentation mIoU**: ~0.63

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch-size 16` or `8`
- Reduce image size in `src/data.py`

### Slow Training
- Increase num workers: `--num-workers 8`
- Use GPU: Ensure CUDA is installed and PyTorch uses it
- Check with: `python -c "import torch; print(torch.cuda.is_available())"`

### Data Loading Issues
- Verify dataset structure matches expected format
- Check image file extensions (supported: .jpg, .png, .jpeg)
- Ensure label files exist for all images

## Model Architecture

### Backbone
- ResNet18 with pretrained ImageNet weights
- Output: 512 channels at 1/32 spatial resolution

### Classification Head
- Global Average Pooling
- Fully Connected: 512 → 256 → num_classes
- Dropout (0.5)

### Segmentation Head
- 4x Transposed Convolution blocks
- Progressive upsampling with feature refinement
- Output: num_classes channels at original resolution

### Loss Function
$$L_{\text{total}} = \alpha L_{\text{CE}} + \beta L_{\text{Dice}}$$

Where:
- $L_{\text{CE}}$ = Cross-Entropy Loss (classification)
- $L_{\text{Dice}}$ = Dice Loss (segmentation)

## License

MIT License - Feel free to use for research and commercial projects.

## Citation

If you use this code, please cite:

```bibtex
@misc{multitask-resnet18,
  title={MultiTask-ResNet18-Vision: Joint Classification and Segmentation},
  year={2024}
}
```
