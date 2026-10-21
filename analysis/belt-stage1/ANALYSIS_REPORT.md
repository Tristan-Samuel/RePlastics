# Model inspection report

Generated: 2026-10-16 12:56:49

## Configuration

- Model: `models/resnext50_metal_plastic_belt_stage1_layer4.pt`
- Images: data_belt_stage1/test (test split)
- Device: mps
- Steps: cam, filters, maps, pca, stats

How to read this: feature maps show what fired; Grad-CAM shows which of those firings a class score used; PCA is a 2-D cartoon of the 2048-D pooled vector.

## Scoring

- Labeled images: 300
- Correct: 299 (99.7%)
- Mistakes: 1

By resin:

- OTHERS: 0/26 mistakes (0.0%)
- HDPE: 0/23 mistakes (0.0%)
- PET: 0/61 mistakes (0.0%)
- PP: 0/6 mistakes (0.0%)
- unknown: 1/184 mistakes (0.5%)

- `data_belt_stage1/test/metal/capture_01896_20260929_141207_925981.jpg`: true metal → plastic (69.2%)

## Feature maps

Bright cells are filters that fired on this 224×224 input. Each grid shows the 16 strongest channels, not filters 0–15. layer1 is edges and color; layer4 is a 7×7 abstract map.
- metal capture_01895_20260929_141206_407448.jpg layer1: 256 filters, mean=1.1303, sparsity=48.8%
- metal capture_01895_20260929_141206_407448.jpg layer2: 512 filters, mean=0.7292, sparsity=58.6%
- metal capture_01895_20260929_141206_407448.jpg layer3: 1024 filters, mean=0.4997, sparsity=62.8%
- metal capture_01895_20260929_141206_407448.jpg layer4: 2048 filters, mean=0.1139, sparsity=92.8%
- metal capture_00168_20260917_123444_682681_OTHERS.jpg layer1: 256 filters, mean=1.1832, sparsity=48.7%
- metal capture_00168_20260917_123444_682681_OTHERS.jpg layer2: 512 filters, mean=0.7445, sparsity=58.9%
- metal capture_00168_20260917_123444_682681_OTHERS.jpg layer3: 1024 filters, mean=0.4970, sparsity=62.9%
- metal capture_00168_20260917_123444_682681_OTHERS.jpg layer4: 2048 filters, mean=0.1048, sparsity=93.8%
- plastic HDPE_capture_00014_20251112_120713_685907.jpg layer1: 256 filters, mean=1.0791, sparsity=46.9%
- plastic HDPE_capture_00014_20251112_120713_685907.jpg layer2: 512 filters, mean=0.6475, sparsity=58.4%
- plastic HDPE_capture_00014_20251112_120713_685907.jpg layer3: 1024 filters, mean=0.4589, sparsity=62.6%
- plastic HDPE_capture_00014_20251112_120713_685907.jpg layer4: 2048 filters, mean=0.1775, sparsity=88.7%
- plastic capture_00215_20260914_102957_949156_PET.jpg layer1: 256 filters, mean=1.1479, sparsity=48.4%
- plastic capture_00215_20260914_102957_949156_PET.jpg layer2: 512 filters, mean=0.7172, sparsity=59.5%
- plastic capture_00215_20260914_102957_949156_PET.jpg layer3: 1024 filters, mean=0.4833, sparsity=62.2%
- plastic capture_00215_20260914_102957_949156_PET.jpg layer4: 2048 filters, mean=0.0865, sparsity=95.3%

## Filter activation

Column 0 is the 224×224 tensor the network scored. The next four panels are the layer4 filters with the highest mean activation on that photo.
- metal capture_01895_20260929_141206_407448.jpg: top filters [47, 1120, 441, 1888]
- metal capture_00168_20260917_123444_682681_OTHERS.jpg: top filters [2041, 1980, 398, 652]
- plastic HDPE_capture_00014_20251112_120713_685907.jpg: top filters [747, 168, 1246, 711]
- plastic capture_00215_20260914_102957_949156_PET.jpg: top filters [1190, 1755, 1255, 367]
Positive bars fire more on metal photos; negative bars fire more on plastic. Head-only training still uses these ImageNet channels.
- filter 1203: Δ=-1.1089 (metal 0.0107, plastic 1.1196)
- filter 441: Δ=+1.0751 (metal 1.2594, plastic 0.1843)
- filter 1888: Δ=+1.0593 (metal 1.1929, plastic 0.1336)
- filter 1886: Δ=-0.9875 (metal 0.0314, plastic 1.0189)
- filter 168: Δ=-0.9748 (metal 0.0331, plastic 1.0079)
- filter 1120: Δ=+0.8597 (metal 0.8891, plastic 0.0294)
- filter 18: Δ=-0.8577 (metal 0.0431, plastic 0.9008)
- filter 787: Δ=-0.7538 (metal 0.1337, plastic 0.8876)
- filter 1481: Δ=+0.7476 (metal 1.0952, plastic 0.3477)
- filter 736: Δ=+0.7273 (metal 0.9675, plastic 0.2402)

## Layer activation statistics

Mean activation and sparsity (% of zeros) averaged over a sample of photos per class. Sparsity usually rises in deeper ReLU stages. If metal and plastic look the same through layer4, the frozen backbone is not separating them and the fc head is doing the work.
**metal** (16 photos):
- layer1: mean=1.1371, std=1.8790, sparsity=48.3%
- layer2: mean=0.7223, std=1.3468, sparsity=58.4%
- layer3: mean=0.4942, std=1.0153, sparsity=62.8%
- layer4: mean=0.1084, std=0.6370, sparsity=93.3%
**plastic** (16 photos):
- layer1: mean=1.1666, std=1.9100, sparsity=47.7%
- layer2: mean=0.7129, std=1.3485, sparsity=58.7%
- layer3: mean=0.4857, std=0.9993, sparsity=62.6%
- layer4: mean=0.1348, std=0.6886, sparsity=91.8%

## Grad-CAM

Grad-CAM is not a feature map. It weights layer4 by how much each channel contributed to one class score, then upsamples the 7×7 result to 224×224. Overlay is on the exact tensor the model scored. On a mistake the middle panel is “why it predicted that” and the next panel is the true class.
Labeled accuracy on this set: 299/300 (99.7%). 1 mistakes have dual CAMs.
- data_belt_stage1/test/metal/capture_01896_20260929_141207_925981.jpg: true metal → plastic (69.2%)

## Feature space PCA

Each point is one photo after global average pooling (2048 numbers). Separate blobs mean the vector already clusters by class. A mixed cloud means overlap; black X marks are mistakes. Two PCs are a cartoon of 2048-D space — read the variance bars.
- samples: 300
- feature dim: 2048
- PC1 12.1%, PC2 4.8%, first two 16.8%
t-SNE used PCA-10 then 2-D embedding (perplexity=30). Local neighborhoods, not global axes.

## Figures

- `maps_metal_can-2026-09-29_layer1.png`
- `maps_metal_can-2026-09-29_layer2.png`
- `maps_metal_can-2026-09-29_layer3.png`
- `maps_metal_can-2026-09-29_layer4.png`
- `maps_metal_OTHERS_foil-tray_layer1.png`
- `maps_metal_OTHERS_foil-tray_layer2.png`
- `maps_metal_OTHERS_foil-tray_layer3.png`
- `maps_metal_OTHERS_foil-tray_layer4.png`
- `maps_plastic_HDPE_bottle_layer1.png`
- `maps_plastic_HDPE_bottle_layer2.png`
- `maps_plastic_HDPE_bottle_layer3.png`
- `maps_plastic_HDPE_bottle_layer4.png`
- `maps_plastic_PET_belt-frame_layer1.png`
- `maps_plastic_PET_belt-frame_layer2.png`
- `maps_plastic_PET_belt-frame_layer3.png`
- `maps_plastic_PET_belt-frame_layer4.png`
- `filters_side-by-side.png`
- `filters_class-ranking.png`
- `stats_layer-activations.png`
- `cam_CORRECT_metal_OTHERS_metal_foil-tray.png`
- `cam_CORRECT_metal_metal_can-2026-09-28.png`
- `cam_CORRECT_metal_metal_can-2026-09-29.png`
- `cam_WRONG_metal_plastic_crushed-can.png`
- `pca_feature-space.png`
- `tsne_feature-space.png`
