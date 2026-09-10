# RePlastics

Train a ResNeXt-50 classifier to tell **metal** from **plastic**. Start with a frozen backbone and train only the final layer, then optionally unfreeze later residual stages.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

Current counts (original internet-style set):

| Split | Metal | Plastic |
|---|---|---|
| train | 1072 | 2808 |
| validation | 227 | 601 |
| test | 227 | 601 |

Drop a photo straight into the matching class folder if you already know which split it belongs to.

## Real photos (the ones you will actually classify)

Do **not** delete `data/` and train only on a couple hundred in-the-wild photos from scratch. Keep the original set, keep the best checkpoint, and fine-tune on a **separate** tree so the photos you score stay out of training.

1. Put unsplitted class folders in a new root (this does not touch `data/`):

   ```
   data_real/metal/
   data_real/plastic/
   ```

2. Split them 70% train / 15% validation / 15% test:

   ```bash
   python split_data.py --dest data_real
   ```

The real **test** split is the number you should report. Never copy those files into train.

Fine-tune the current best model on that tree:

```bash
python train.py --resume --data-dir data_real --unfreeze fc --epochs 8
```

If validation accuracy stalls, unfreeze the last residual stage (`layer4`) at a lower learning rate:

```bash
python train.py --resume --data-dir data_real --unfreeze layer4 --lr 0.001 --lr-backbone 0.0001 --epochs 8
```

## Add a batch of new images to the original set

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

This starts from ImageNet weights, freezes the backbone, trains the `fc` head, picks the best epoch by validation accuracy, then reports held-out test accuracy. The checkpoint is saved to `models/resnext50_metal_plastic.pt`.

Continue training from that checkpoint:

```bash
python train.py --resume
```

Resume only replaces the saved file if validation accuracy improves. During training, that best checkpoint is written to disk after every improvement, so you can stop with Ctrl+C without losing it.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | Dataset root with `train` / `validation` / `test` |
| `--model` | `models/resnext50_metal_plastic.pt` | Checkpoint to load and save |
| `--epochs` | `2` | Training epochs |
| `--lr` | `0.001` | Learning rate for `fc` |
| `--lr-backbone` | `--lr / 10` | Learning rate for unfrozen backbone stages |
| `--unfreeze` | `fc` | `fc`, `layer4`, `layer3`, or `all` |

### Progressive unfreeze

ResNeXt-50 is stem → `layer1` → `layer2` → `layer3` → **`layer4`** → `fc`. Train from the top down, dropping the learning rate as you go:

1. `--unfreeze fc` — backbone frozen, train the 2-class head (always do this first).
2. `--unfreeze layer4` — also train the last residual stage; keep `layer1`–`layer3` frozen.
3. `--unfreeze layer3` — optional next step. Skip `all` unless you have a lot of data.

Do not train the full backbone first and then freeze it. That wrecks the ImageNet features.

## Train on Colab GPU (no notebook copy-paste)

The official [Colab CLI](https://github.com/googlecolab/google-colab-cli) can provision a runtime, upload this repo, run `train.py`, and download the checkpoint. Default GPU is **T4**. Change it later with `--gpu L4` / `--gpu A100` or `COLAB_GPU`.

```bash
chmod +x colab_train.sh
./colab_train.sh --drive --resume --unfreeze fc --epochs 8
```

That mounts Google Drive, copies `train/NonPlastic` and `train/Plastic` onto the VM disk as **metal** / **plastic**, and keeps Drive `val/` as validation while holding out 15% of train as test. Drive originals are not moved. The copy is the slow step; after that the T4 reads local files. Override the folder with `--drive-dir`.

Fine-tune on the **same VM** without copying again:

```bash
./colab_train.sh --keep --drive --resume --unfreeze fc --epochs 8
./colab_train.sh --keep --drive --resume --unfreeze layer4 --lr 0.001 --lr-backbone 0.0001 --epochs 8
```

`--keep` leaves the T4 up so `/content/data_real` survives. Changing `--unfreeze` or the learning rate still runs a short validation pass first (now with batch logs). That pass is seconds on local disk, not minutes on Drive. If you stop the session, the VM disk is wiped and the next `--drive` run copies photos once more.

To upload a local `data_real` tree instead:

```bash
./colab_train.sh --resume --unfreeze fc --data-dir data_real --epochs 8
```

The first run prints a Google sign-in URL. Open it, approve access, then paste the **authorization code** (not the URL) back into the terminal. Drive training then opens a second Google consent page for Drive. Finish that page, then press Enter. If that handshake still fails, a Colab tab opens: run `drive.mount('/content/drive')` in a cell and wait until the helper sees My Drive. A failed Python step now stops the helper instead of printing "Training finished".

If login fails with `Scope has changed`, re-run the command. The helper treats a reduced Google grant as a warning instead of crashing. You should not need to paste a code again once `~/.config/colab-cli/token.json` exists.

With `--drive`, the helper zips only the training scripts, mounts Drive, copies photos onto the VM disk, and (if `--resume`) copies the checkpoint from Drive. Otherwise it also zips the local dataset directory. It installs `requirements-colab.txt` on the VM (PyTorch is already on Colab), then pulls back `models/resnext50_metal_plastic.pt` and the confusion-matrix artifacts. The VM is stopped when the script exits unless you pass `--keep`.

## Predict

```bash
python predict.py
python predict.py --data-dir data_real
```

Opens a window with 9 random test images. Each photo is labeled with the predicted class and confidence. The checkpoint is loaded from `models/resnext50_metal_plastic.pt`, or from `resnext50_metal_plastic.pt` in the current directory if that is the file you already have.
