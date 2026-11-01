# Experiment evidence and interpretation

All original notebook outputs are preserved. Absence of an error output does not prove every cell completed. Inspect execution counts, saved tables and terminal logs together.

| Study | Evidence | Interpretation |
|---|---|---|
| Controlled RF-DETR experiments | `notebooks/06-Roadscan-AllExperiments.ipynb`, `results/RoadScan_20Epoch_Experiments/` | Separate held-out India protocol: train 5,026; validation 644; internal test 810; India 1,520; Saudi 100. These are not the deployed checkpoint's original partitions. |
| E1 image-presence baseline | Same controlled notebook | RDD F1: ResNet18 60.81%, RF-DETR 82.30%; Saudi 53.15% and 74.46%. Image-presence F1 is not detection mAP. |
| E2 resolution | Controlled 512 vs 640 runs | RDD mAP50 about 58.2% to 59.5%; Saudi about 45.5% to 44.9%. CPU about 0.395 to 0.630 s. Higher resolution did not consistently improve both domains. |
| E3 horizontal flip | Controlled augmentation run | Saudi mAP50 about 45.5% to 49.2% within this experiment. |
| E4 held-out country | Controlled no-augmentation 512 model | RDD 58.22%, India 9.28%, Saudi 45.54% mAP50. Domain shift is substantial. |
| Historical YOLO version search | `RoadScan_YOLO_VersionCompare (3).ipynb` | Configured 10-epoch exploration, 1,600 training / 800 validation / 800 internal test. YOLO26n validation mAP50 37.20%, mAP50:95 15.20%. Different protocol from final deployment. |
| YOLO26n extra fine-tuning | `RoadScan_YOLO26n_FineTuning (1) (1).ipynb` | 20 additional epochs. RDD mAP50 47.95 to 50.00%; Saudi 15.01 to 16.32%. Saudi precision fell 39.91 to 20.85%, so gains were not uniformly beneficial. Both optimization and augmentation changed. Saudi set: 78 images. |
| SAHI | `RoadScan_YOLO26n_SAHI_Results.ipynb` | Existing detector; no new training. Standard vs SAHI-512: RDD mAP50 48.96 to 39.08%; Saudi 23.38 to 23.41%. CPU 199.97 to 1,404.53 ms on the timing sample. Saudi set: 77 images. Standard inference retained. |
| Latest original-split YOLO benchmark | `yolo-comparison (1).ipynb` | Saved 640-pixel YOLO26n/s/m/l, YOLO11s/m and YOLOv8s archive runs completed 20 epochs. Notebook also records YOLOv8m-640 completion. YOLO26s-960 completed 6 epochs and part of epoch 7 before GPU/runtime interruption. YOLO26m-960 completion is not established. Not every evaluation cell is populated. |

The sibling `Previous_Model_Experiments` directory retains additional attempts and original training exports. An earlier shortened 10-epoch search is a planned exploratory schedule; it should not be described as every model failing to reach 20 epochs. GPU/runtime limits explain the unfinished longer experiments as reported by the team, not a proven model failure.

RF-DETR was retained after considering the available evaluation and application results. The historical YOLO numbers do not constitute a controlled head-to-head comparison with the deployed RF checkpoint.
