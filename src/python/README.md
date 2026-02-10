# Emotion Model Training and Evaluation

This document describes the training pipeline, datasets, and evaluation used for the emotion classification model deployed with ncnn.

The goal is to provide reproducible benchmarks and justify architectural and training decisions.

---

## Environment

- Python 3.12.11
- PyTorch 2.8.0

### Dependencies
```
matplotlib==3.10.6
numpy==2.3.3
Pillow==11.3.0
scikit_learn==1.7.2
seaborn==0.13.2
thop==0.1.1.post2209072238
torchvision==0.23.0+cu126
tqdm==4.67.1
```

---

## Model Architecture

- Base model: MobileNetV2 (ImageNet-V2 pretrained)
- Input: grayscale, 64x64
- Conv1 adapted for single-channel input
- FLOPs: ~26M
- Parameters: ~2.23M
- Trainable parameters: ~2.23M

The model was selected to balance accuracy and CPU inference speed.

---

## Training Strategy

Two-stage full-network training was used.

### Stage 1
- Learning rate: 1e-3
- All layers trainable
- Train until validation loss plateaus
- Best checkpoint saved

### Stage 2
- Load best Stage 1 checkpoint
- Learning rate: 1e-5
- Train until early stopping

Traditional freeze-then-unfreeze fine-tuning underperformed on the combined dataset.

---

## Datasets

Training uses a combination of FER2013 and RAF-DB to improve robustness to domain shift.

### RAF-DB (approx. 35k images)
- angry: 4550
- disgust: 4857
- fear: 5103
- happy: 5023
- neutral: 4902
- sad: 4994
- surprise: 5231

### FER2013 (approx. 34k images)
- angry: 4529
- disgust: 483
- fear: 4724
- happy: 8733
- neutral: 6053
- sad: 5581
- surprise: 3833

---

## Data Splits

Combined dataset split:
- 70 percent train
- 15 percent validation
- 15 percent test

### Class Balancing

- Weighted random sampler
- Class weights in CrossEntropyLoss

---

## Preprocessing and Augmentation

- Grayscale conversion
- Resize to 64x64
- Random horizontal flip (p=0.3)
- Color jitter (brightness=0.2, contrast=0.3)
- Random erasing (p=0.3)
- Normalize pixel range to [-1, 1]

---

## Optimization

- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Weight decay: 1e-2
- Gradient clipping: norm 1.0
- Scheduler: ReduceLROnPlateau
- Early stopping patience: 8 epochs

---

## Ablation Results

| Approach | Validation Accuracy |
|----------|---------------------|
| Freeze then unfreeze | ~73.8% |
| EfficientNet-lite0 | ~77.4% |
| Two-stage full training | ~79.3% |

---

## Evaluation Results

### Overall Performance

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| MobileNetV2 | 79.3% | 0.793 |

### Per-Class Performance

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Angry | 0.744 | 0.778 | 0.761 |
| Disgust | 0.908 | 0.943 | 0.923 |
| Fear | 0.777 | 0.737 | 0.756 |
| Happy | 0.888 | 0.852 | 0.870 |
| Neutral | 0.706 | 0.742 | 0.723 |
| Sad | 0.704 | 0.683 | 0.694 |
| Surprise | 0.860 | 0.878 | 0.869 |

---

## Training Curves

![Training and Validation Accuracy](../../assets/acc_curves.png)  
![Training and Validation Loss](../../assets/loss_curves.png)

The learning rate reduction at epoch ~15 corresponds to the ReduceLROnPlateau scheduler.  
Final model selected based on best validation performance, not final epoch.

---

## Notes

- Combined-dataset training improves robustness to lighting, pose, and expression ambiguity
- The resulting model is optimized for fast CPU inference when exported to ncnn
