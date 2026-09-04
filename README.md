# RePlastics

Train a ResNeXt-50 classifier to tell **metal** from **plastic**. The backbone stays frozen; only the final layer is trained.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib
```

CUDA, Apple Silicon (MPS), or CPU is chosen automatically.

## Data layout

Images live in leak-free split folders. Train, validation, and test never share files.

```
data/
  train/metal/
  train/plastic/
  validation/metal/
  validation/plastic/
  test/metal/
  test/plastic/
```

Current counts:

| Split | Metal | Plastic |
|---|---|---|
| train | 1072 | 2808 |
| validation | 227 | 601 |
| test | 227 | 601 |

Drop a photo straight into the matching class folder if you already know which split it belongs to.

## Add a batch of new images

1. Put unsplitted class folders next to the splits:

   ```
   data/metal/
   data/plastic/
   ```

2. Run:

   ```bash
   python split_data.py
   ```

That script splits each class **70% train / 15% validation / 15% test** with a fixed seed, copies the files into the split folders, then deletes the unsplitted source folders. Filenames must not collide with images already in a split.

Do not put the same image in more than one split.

## Train

```bash
python train.py
```

This starts from ImageNet weights, trains for 16 epochs, picks the best epoch by validation accuracy, then reports held-out test accuracy. The checkpoint is saved to `resnext50_metal_plastic.pt`.

Continue training from that checkpoint:

```bash
python train.py --resume
```

Resume only replaces the saved file if validation accuracy improves. During training, that best checkpoint is written to disk after every improvement, so you can stop with Ctrl+C without losing it.

## Predict

```bash
python predict.py
```

Opens a window with 9 random test images. Each photo is labeled with the predicted class and confidence.
