# Jetson

The belt loads a TensorRT engine built on the Jetson. The file you copy there is ONNX, exported on the training machine from `models/resnext50_metal_plastic.pt`.

## Export

```bash
pip install onnx onnxruntime
python export_onnx.py --check --data-dir data_belt_stage1
```

That writes two files in `models/`:

- `resnext50_metal_plastic.onnx` — ResNeXt-50, one image in, two logits out
- `resnext50_metal_plastic.onnx.json` — class index, mean, std, temperature, threshold, and the checkpoint name

`--check` scores two metal and two plastic test JPEGs with PyTorch and with ONNX Runtime, both through `inference_transform`. It stops if any logit differs by more than 1e-4. Pass `--images` to score specific photos instead.

`models/` is gitignored, same as the `.pt` files.

The graph is `model.forward` at opset 17, batch 1, input `image` with shape `[1, 3, 224, 224]`. The output is `logits` with shape `[1, 2]`. Export runs on CPU with the legacy exporter (`dynamo=False`), which is the graph JetPack TensorRT accepts.

## What each frame must do

`preprocess.py` and `decide.py` stay outside the graph. A threshold change does not need a new export.

1. Open the JPEG as RGB, including the phone EXIF rotation (`load_rgb`).
2. Bilinear resize to 224×224. Pillow matches training. OpenCV `INTER_LINEAR` lands on different pixels. If the belt app uses OpenCV, re-score frames whose winning score sits near 0.85 before trusting the hold.
3. Divide by 255, subtract mean `(0.485, 0.456, 0.406)`, divide by std `(0.229, 0.224, 0.225)`.
4. Send NCHW `float32`, shape `[1, 3, 224, 224]`, input name `image`. The channel order is RGB.
5. Read output `logits`, shape `[1, 2]`. The JSON `class_to_idx` is the order. For this checkpoint, index 0 is metal and index 1 is plastic.
6. Softmax of `logits / temperature`. The JSON temperature is `1.0118387937545776`.
7. When the winning score is below the JSON threshold `0.85`, the label is `uncertain`. Otherwise it is the winning class. The measured fit was 0.97. The live cutoff is 0.85 so a showcase sorts the confident calls.

## Build the engine on the Jetson

Copy the `.onnx` and the `.json`. Build the engine on the Jetson that will run it. An engine is tied to that GPU and that JetPack.

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=resnext50_metal_plastic.onnx \
  --saveEngine=/opt/replastics/resnext50_metal_plastic.engine \
  --fp16
```

The camera process opens `/opt/replastics/resnext50_metal_plastic.engine` once at start and keeps the context. Each frame is the seven steps above.

FP16 is the runtime. This network at 224 is small on an Orin. INT8 needs a calibration set.

Rebuild the engine after a JetPack upgrade. A new training run needs a new export, a new JSON, and a new engine built on the device. The Jetson loads that engine.
