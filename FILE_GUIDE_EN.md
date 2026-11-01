# RoadScan file guide

This guide explains every delivery file in the main submission folder and the separate previous-model experiment folder. Original code, notebooks, weights and saved results were copied without changing their bytes. This guide and the packaging utilities are additional documentation; no technical report is included.

## Start here

| Your task | Open this file or folder |
|---|---|
| Publish the final project | `RoadScan_Public_Repo/` in VS Code; follow `docs/PUBLISH_FROM_VSCODE.md`. |
| Run the application | `RoadScan_Public_Repo/notebooks/application/app.py`, using the command in the repository README. |
| Inspect the final model | `RoadScan_Public_Repo/models/deployment/rf_small.pth` and `configs/detector_selection.json`. |
| Read recorded deployment scores | `RoadScan_Public_Repo/results/detection_metrics.csv`. |
| Review the main RF experiments | `RoadScan_Public_Repo/notebooks/07-Roadscan-AllExperiments.ipynb`. |
| Review YOLO, fine-tuning and SAHI | `Previous_Model_Experiments/`; start with its README and notebook inventory. |
| Obtain the exact dataset | Local `.local_artifacts/RoadScan_Dataset.zip`, or the published dataset Release after the maintainer uploads it. |

## How to interpret the files

- `.ipynb`: notebook containing code, explanations and any saved outputs. Execution counts and outputs do not prove that every stage finished.
- `.py`: executable Python source; scripts are run from the terminal, while modules are imported by other code.
- `.csv`: tables such as metrics, per-class results, timing samples or split assignments.
- `.json` / `.jsonl`: configuration, provenance, manifests or structured verification records.
- `.pth`, `.pt`, `.ckpt`, `.safetensors`: model/checkpoint binaries; their roles depend on the run and metadata.
- `.png` / `.jpg`: plots or saved qualitative examples, not standalone aggregate performance measurements.
- `.md`: documentation readable directly in VS Code Markdown Preview.
- `.zip`: preserved archive. The archive is listed as one delivery file; its internal members are not separately listed here.

The deployment results and controlled experiment results use different checkpoints/protocols. The run folders `rf_small_512_noaug`, `rf_small_640_noaug` and `rf_small_512_flip` mean 512 without the tested flip augmentation, 640 without it, and 512 with it, respectively. Do not mix their thresholds or weights with the deployed model.

Legacy similarity code is retained for provenance. E5 was canceled by the instructor, and the Streamlit application is a detection/priority/reporting workflow. Its presence does not imply an evaluated duplicate-detection feature.

## Files by directory

The following catalog covers **301 delivery files**. Each directory heading is relative to `RoadScan_Delivery_20261101/`. Every file has its own row, including repeated result filenames in different runs.

### Delivery root

| File | Purpose |
|---|---|
| `COPY_PROVENANCE_LOCAL.json` | Local copy ledger: original source locations, destination paths, file sizes and SHA-256 hashes. Keep outside the public repository because it includes local computer paths. |
| `PACKAGING_SUMMARY.json` | Summary of the packaging operation, including verified copies, notebook count and credential-pattern scan findings. |
| `ابدئي_هنا.md` | Arabic quick-start instructions explaining the two delivery folders and how to publish the repository and dataset. |

### `Previous_Model_Experiments`

| File | Purpose |
|---|---|
| `.gitattributes` | Configures Git LFS for model binaries (and ZIP archives in the experiment folder). Also preserves notebook files without text line-ending conversion. |
| `.gitignore` | Excludes temporary files, environments, caches and local runtime data from Git. In the main repository it also excludes .local_artifacts and downloaded data. |
| `DUPLICATES_OMITTED.json` | Lists exact duplicate earlier notebooks omitted during packaging. Distinct notebook versions were retained. |
| `NOTEBOOK_INVENTORY.json` | Notebook inventory with cell counts, executed-cell counts, saved-output counts and recorded error outputs. A nonempty output count does not prove all cells completed. |
| `ORIGINAL_FILES_SHA256.json` | Checksums and sizes of original copied files. Use this ledger to confirm that notebook outputs, code and recorded results have not changed. |
| `README.md` | Overview of previous-model experiment groups, preserved evidence, incomplete runs and optional separate publication. |

### `Previous_Model_Experiments/01_YOLO_Version_Search`

| File | Purpose |
|---|---|
| `RoadScan_YOLO_VersionCompare (3).ipynb` | Historical executed YOLO version/size comparison on an earlier experimental protocol. Saved state: 15 executed cells, 15 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `Previous_Model_Experiments/02_YOLO26n_FineTuning`

| File | Purpose |
|---|---|
| `RoadScan_YOLO26n_FineTuning (1) (1).ipynb` | Executed YOLO26n additional fine-tuning study, preserving before/after evaluation outputs and training logs. Saved state: 18 executed cells, 17 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `Previous_Model_Experiments/03_SAHI`

| File | Purpose |
|---|---|
| `RoadScan_YOLO26n_SAHI_Results.ipynb` | Executed SAHI tiled-inference study using an existing YOLO26n checkpoint. Contains no new model training. Saved state: 11 executed cells, 8 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `Previous_Model_Experiments/04_Latest_YOLO_Benchmark/Additional_Run_Exports`

| File | Purpose |
|---|---|
| `last (1).pt` | Original last-saved YOLO checkpoint paired with run_identity (1).json; the paired identity records yolo26n_640. This is a last checkpoint, not necessarily validation-best. |
| `last (2).pt` | Original last-saved YOLO checkpoint paired with run_identity (2).json; the paired identity records yolo26s_640. This is a last checkpoint, not necessarily validation-best. |
| `last.pt` | Original last-saved YOLO checkpoint paired with run_identity.json; the paired identity records yolo26m_640. This is a last checkpoint, not necessarily validation-best. |
| `results (1).csv` | Original per-epoch training/validation metrics paired by suffix with yolo26n_640. Keep the matching run_identity (1).json alongside it. |
| `results (2).csv` | Original per-epoch training/validation metrics paired by suffix with yolo26s_640. Keep the matching run_identity (2).json alongside it. |
| `results.csv` | Original per-epoch training/validation metrics paired by suffix with yolo26m_640. Keep the matching run_identity.json alongside it. |
| `run_identity (1).json` | Original identity record for the suffix-matched checkpoint and results CSV: yolo26n_640, configured for 20 epochs. The configuration alone does not prove completion. |
| `run_identity (2).json` | Original identity record for the suffix-matched checkpoint and results CSV: yolo26s_640, configured for 20 epochs. The configuration alone does not prove completion. |
| `run_identity.json` | Original identity record for the suffix-matched checkpoint and results CSV: yolo26m_640, configured for 20 epochs. The configuration alone does not prove completion. |

### `Previous_Model_Experiments/04_Latest_YOLO_Benchmark`

| File | Purpose |
|---|---|
| `RoadScan_Benchmark_Resume_Latest.zip` | Unmodified saved YOLO benchmark archive containing model weights, training logs, plots and protocol identity. Uses Git LFS if this archive folder is published separately. |
| `yolo-comparison (1).ipynb` | Latest saved original-split YOLO benchmarking notebook. Preserves completed and interrupted stages; not all evaluation cells have outputs. Saved state: 10 executed cells, 9 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `Previous_Model_Experiments/05_Unexecuted_Configurations`

| File | Purpose |
|---|---|
| `RoadScan_Professional.ipynb` | Unexecuted proposed YOLO training configuration. Code/configuration only, not measured evidence. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `RoadScan_Professional_NewDataset (1).ipynb` | Unexecuted newer-dataset YOLO training/evaluation configuration. Not a completed experiment. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `Previous_Model_Experiments/06_Earlier_Attempts`

| File | Purpose |
|---|---|
| `RoadScan_KaikePing_YOLO11s.ipynb` | Earlier YOLO11s experiment adapting the KaikePing/RoadDamageYOLO approach to the project dataset. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `RoadScan_Simple_Compare.ipynb` | Earlier simplified comparison of multiple YOLO detectors with validation-based selection. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `RoadScan_YOLO11.ipynb` | Earlier YOLO11 experiment notebook; retain its original code and saved execution state. Saved state: 14 executed cells, 13 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `RoadScan_YOLO26_Improved_Claude.ipynb` | Earlier YOLO26 modification/experiment notebook; its historical filename is retained. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `RoadScan_YOLO_VersionCompare.ipynb` | Earlier version of the YOLO architecture/configuration comparison. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `RoadScan_YOLOv8_Improved.ipynb` | Earlier YOLOv8 experiment notebook with modified training/evaluation settings. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `roadscan_best.pt` | Historical YOLO checkpoint from earlier attempts. Not the RF-DETR model used by the final application. |
| `roadscan_yolov8 copy (1).ipynb` | Earlier YOLOv8 training/evaluation variant. Numbered copy names are preserved because the contents are distinct, not renamed as a new completed study. Saved state: 14 executed cells, 13 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `roadscan_yolov8 copy 2.ipynb` | Earlier YOLOv8 training/evaluation variant. Numbered copy names are preserved because the contents are distinct, not renamed as a new completed study. Saved state: 14 executed cells, 13 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `roadscan_yolov8 copy.ipynb` | Earlier YOLOv8 training/evaluation variant. Numbered copy names are preserved because the contents are distinct, not renamed as a new completed study. Saved state: 13 executed cells, 13 cells with outputs, 1 recorded error outputs. These counts do not certify completion. |
| `roadscan_yolov8_copy.ipynb` | Earlier YOLOv8 training/evaluation variant. Numbered copy names are preserved because the contents are distinct, not renamed as a new completed study. Saved state: 16 executed cells, 15 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo`

| File | Purpose |
|---|---|
| `.gitattributes` | Configures Git LFS for model binaries (and ZIP archives in the experiment folder). Also preserves notebook files without text line-ending conversion. |
| `.gitignore` | Excludes temporary files, environments, caches and local runtime data from Git. In the main repository it also excludes .local_artifacts and downloaded data. |
| `ORIGINAL_FILES_SHA256.json` | Checksums and sizes of original copied files. Use this ledger to confirm that notebook outputs, code and recorded results have not changed. |
| `README.md` | Main repository landing page: project overview, architecture, installation, application commands, recorded results, data/evaluation instructions and limitations. |
| `THIRD_PARTY_CREDITS.md` | Attribution and upstream links for RF-DETR, DINOv2, RDD2022 and SaudiRD2. Third-party assets retain their own terms. |
| `pyproject.toml` | Python package/build metadata. Enables installing the src/roadscan package with pip install -e . |
| `requirements.txt` | Original Python dependency requirements. Install these before running the application and original tests. |

### `RoadScan_Public_Repo/.local_artifacts`

| File | Purpose |
|---|---|
| `RoadScan_Dataset.zip` | Exact curated dataset archive, staged locally for publication as a Release asset. Contains images, labels, metadata, manifests and checksums. Excluded from ordinary Git commits. |

### `RoadScan_Public_Repo/.local_artifacts/experiment_checkpoints/results/RoadScan_20Epoch_Experiments/India_6f44f8600748/rf_small_512_flip`

| File | Purpose |
|---|---|
| `checkpoint_14.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_19.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_4.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_9.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_best_regular.pth` | Validation-selected regular/non-EMA model checkpoint for this controlled training run. Not automatically the deployed application model. |
| `checkpoint_best_total.pth` | Checkpoint saved as best-total by the original trainer. Interpret it using the run’s metadata; do not assume it is identical to the final-epoch or deployed checkpoint. |
| `final_epoch20.ckpt` | Final checkpoint recorded after 20 completed epochs for this controlled run. Kept locally; it is distinct from the deployed EMA checkpoint. |
| `last.ckpt` | Most recently saved training checkpoint for this run, retained locally for recovery/continuation. |

### `RoadScan_Public_Repo/.local_artifacts/experiment_checkpoints/results/RoadScan_20Epoch_Experiments/India_6f44f8600748/rf_small_512_noaug`

| File | Purpose |
|---|---|
| `checkpoint_14.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_19.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_4.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_9.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_best_regular.pth` | Validation-selected regular/non-EMA model checkpoint for this controlled training run. Not automatically the deployed application model. |
| `checkpoint_best_total.pth` | Checkpoint saved as best-total by the original trainer. Interpret it using the run’s metadata; do not assume it is identical to the final-epoch or deployed checkpoint. |
| `final_epoch20.ckpt` | Final checkpoint recorded after 20 completed epochs for this controlled run. Kept locally; it is distinct from the deployed EMA checkpoint. |
| `last.ckpt` | Most recently saved training checkpoint for this run, retained locally for recovery/continuation. |

### `RoadScan_Public_Repo/.local_artifacts/experiment_checkpoints/results/RoadScan_20Epoch_Experiments/India_6f44f8600748/rf_small_640_noaug`

| File | Purpose |
|---|---|
| `checkpoint_14.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_19.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_4.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_9.ckpt` | Intermediate training checkpoint saved under the original trainer’s epoch numbering. Retained locally for recovery; the filename alone is not a completed-run claim. |
| `checkpoint_best_regular.pth` | Validation-selected regular/non-EMA model checkpoint for this controlled training run. Not automatically the deployed application model. |
| `checkpoint_best_total.pth` | Checkpoint saved as best-total by the original trainer. Interpret it using the run’s metadata; do not assume it is identical to the final-epoch or deployed checkpoint. |
| `final_epoch20.ckpt` | Final checkpoint recorded after 20 completed epochs for this controlled run. Kept locally; it is distinct from the deployed EMA checkpoint. |
| `last.ckpt` | Most recently saved training checkpoint for this run, retained locally for recovery/continuation. |

### `RoadScan_Public_Repo/configs`

| File | Purpose |
|---|---|
| `detector_selection.json` | Recorded validation-selected detector settings: class order, 512 input resolution, 0.33 confidence thresholds and the expected deployment-weight SHA-256. |
| `project.json` | Original project path and runtime configuration. Some dataset paths are historical and may need changing for a different machine; the original file is preserved. |

### `RoadScan_Public_Repo/data`

| File | Purpose |
|---|---|
| `dataset_release.json` | Expected ZIP name, SHA-256, size and GitHub Release tag for distributing the exact curated dataset. |

### `RoadScan_Public_Repo/data/metadata`

| File | Purpose |
|---|---|
| `DATA_CARD.md` | Dataset documentation describing the source, class definitions, splits and limitations. |
| `data.yaml` | Original YOLO-format dataset configuration for RDD partitions and class names. Historical absolute paths, if present, are retained. |
| `dataset_version.json` | Exact curated dataset definition: 8,000 RDD images plus 100 Saudi images, split counts, annotation counts, sources and limitations. |
| `integrity_sha256.json` | Expected SHA-256 values for the dataset image and label files. Used to detect missing, modified or substituted data. |
| `manifest.jsonl` | One JSON record per dataset image linking its relative path, paired label, split and source/domain metadata. |
| `saudi.yaml` | Original YOLO-format configuration for the separate Saudi evaluation partition. |

### `RoadScan_Public_Repo/docs`

| File | Purpose |
|---|---|
| `EXPERIMENTS.md` | Interpretation of the measured studies, including E1-E4, YOLO fine-tuning, SAHI, GPU-limited runs and differences between evaluation protocols. |
| `NOTEBOOK_INVENTORY.json` | Notebook inventory with cell counts, executed-cell counts, saved-output counts and recorded error outputs. A nonempty output count does not prove all cells completed. |
| `NOTEBOOK_RUN_RESTORATION.json` | Documents the correction that moved short entry-point notebooks into reference_templates and restored original executed notebooks to the main notebook directory, with source paths and SHA-256 records. |
| `ORIGINAL_README.md` | Unchanged copy of the original project README. The repository landing-page README is additional documentation. |
| `PACKAGING_VERIFICATION.json` | Verification record: copy checksums, 10 original unit tests, dataset integrity checks, Git LFS setup and limits of what was executed. |
| `PUBLISH_FROM_VSCODE.md` | Step-by-step instructions for publishing the prepared repository from VS Code and uploading the dataset as a GitHub Release asset. |

### `RoadScan_Public_Repo/models/deployment`

| File | Purpose |
|---|---|
| `deployment_config.json` | Model inference contract, class names, thresholds and priority policy. Also retains configuration for the optional legacy similarity service. |
| `retrieval_provenance.json` | Provenance/fingerprint information for the retained DINOv2 retrieval encoder. It is not evidence of evaluated duplicate-detection accuracy. |
| `rf_small.pth` | Trained RF-DETR Small checkpoint used by the final application. This is the deployment model; Git LFS stores it because of its size. |

### `RoadScan_Public_Repo/models/deployment/dinov2_encoder`

| File | Purpose |
|---|---|
| `config.json` | Architecture/configuration of the saved DINOv2 encoder in this directory. |
| `model.safetensors` | Saved weights of the legacy DINOv2 visual encoder. Retained for provenance and optional similarity code; the delivered Streamlit detection workflow does not require similarity search. |
| `preprocessor_config.json` | Image preprocessing settings paired with the saved DINOv2 encoder. |

### `RoadScan_Public_Repo/models/training`

| File | Purpose |
|---|---|
| `metrics.csv` | Original per-epoch training and validation metrics for this source run. |
| `roadscan_training_identity.json` | Recorded identity/provenance of the original training run, used to relate saved weights and settings to their source run. |
| `training_config.json` | Saved training configuration and hyperparameters. It documents the source run rather than launching training itself. |

### `RoadScan_Public_Repo/notebooks`

| File | Purpose |
|---|---|
| `01_Preprocessing.ipynb` | Original executed data-preparation notebook with saved structural validation tables and 8 figures, restored from the existing reviewed project copy. Saved state: 7 executed cells, 5 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `01_RoadScan_HeldOutCountry_20Epochs_WithRun.ipynb` | Original executed held-out-country notebook with saved evaluation tables and a figure. Saved state: 13 executed cells, 7 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `07-Roadscan-AllExperiments.ipynb` | Main executed RF-DETR experiment notebook, with original saved outputs for the controlled studies and held-out-country protocol. Saved state: 18 executed cells, 12 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `Preprocessing.ipynb` | Original function-definition helper directly executed by ResNet18_baseline.ipynb through %run ./Preprocessing.ipynb. No saved outputs are expected; it is not the executed 01_Preprocessing.ipynb exploration notebook. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `README.md` | Start-here index identifying the original executed notebooks, their saved output counts and historical scopes, and distinguishing reference templates from run evidence. |
| `ResNet18_baseline.ipynb` | Original historical executed ResNet18 classification notebook, preserving saved training and test tables. Its earlier dataset protocol differs from the later controlled study. Saved state: 5 executed cells, 10 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `notebook703c4b4ceb.ipynb` | Original historical RF-DETR Kaggle notebook. Saved training logs explicitly reach max_epochs=20; later cells are unexecuted, and checkpoint identity with the final deployment must not be inferred. Saved state: 6 executed cells, 6 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo/notebooks/application`

| File | Purpose |
|---|---|
| `app.py` | Original Streamlit application. Loads the deployment detector on CPU, accepts images, displays boxes and priority, and exports CSV/JSON reports. |

### `RoadScan_Public_Repo/notebooks/previous_models/01_YOLO_Version_Search`

| File | Purpose |
|---|---|
| `RoadScan_YOLO_VersionCompare (3).ipynb` | Historical executed YOLO version/size comparison on an earlier experimental protocol. Saved state: 15 executed cells, 15 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo/notebooks/previous_models/02_YOLO26n_FineTuning`

| File | Purpose |
|---|---|
| `RoadScan_YOLO26n_FineTuning (1) (1).ipynb` | Executed YOLO26n additional fine-tuning study, preserving before/after evaluation outputs and training logs. Saved state: 18 executed cells, 17 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo/notebooks/previous_models/03_SAHI`

| File | Purpose |
|---|---|
| `RoadScan_YOLO26n_SAHI_Results.ipynb` | Executed SAHI tiled-inference study using an existing YOLO26n checkpoint. Contains no new model training. Saved state: 11 executed cells, 8 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo/notebooks/previous_models/04_Latest_YOLO_Benchmark`

| File | Purpose |
|---|---|
| `yolo-comparison (1).ipynb` | Latest saved original-split YOLO benchmarking notebook. Preserves completed and interrupted stages; not all evaluation cells have outputs. Saved state: 10 executed cells, 9 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo/notebooks/reference_templates`

| File | Purpose |
|---|---|
| `01_Preprocessing.ipynb` | Original notebook entry point for dataset preparation/validation. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `01_Preprocessing_Workspace_Snapshot.ipynb` | Pre-existing delivery workspace variation preserved separately because its bytes differed from the original packaging hash. Not discarded or silently substituted for the source original. Saved state: 2 executed cells, 1 cells with outputs, 1 recorded error outputs. These counts do not certify completion. |
| `02_Classification.ipynb` | Original image-level multi-label classification baseline notebook. Classification presence metrics differ from bounding-box AP. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `03_Detection.ipynb` | Original notebook entry point for RF-DETR detection/evaluation. This is a short project notebook, not the complete controlled-experiment evidence. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `04_Priority.ipynb` | Original notebook for applying and examining the rule-based repair-priority policy. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `05_Similarity.ipynb` | Retained legacy similarity notebook. The instructor canceled E5, so this file is not a claim of validated duplicate detection. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |
| `06_Application.ipynb` | Original notebook entry point for application integration. Saved state: 0 executed cells, 0 cells with outputs, 0 recorded error outputs. These counts do not certify completion. |

### `RoadScan_Public_Repo/results/RoadScan_20Epoch_Experiments/India_6f44f8600748`

| File | Purpose |
|---|---|
| `data_protocol.json` | Controlled-experiment split protocol, held-out country, seed, duplicate-screening settings and limitations. Kept separate from the deployed checkpoint protocol. |
| `environment.json` | Recorded software/runtime environment information for the controlled experiment. |
| `excluded.csv` | Recorded list of images excluded or quarantined by this experiment’s data-screening process; an empty table can indicate no recorded exclusions. |
| `near_duplicate_pairs.csv` | Candidate visually similar image pairs from dataset screening. These are not independently labelled E5 physical-defect pairs. |
| `split_manifest.csv` | Image-level assignment to the controlled experiment’s train, validation, internal test, held-out-country and Saudi partitions. |
| `split_summary.csv` | Counts and summary statistics for the controlled experiment’s partitions. |

### `RoadScan_Public_Repo/results/RoadScan_20Epoch_Experiments/India_6f44f8600748/rf_small_512_flip`

| File | Purpose |
|---|---|
| `cpu_latency.json` | Summary of the CPU timing measurements for this run and its recorded timing scope. |
| `cpu_latency_samples.csv` | Individual CPU timing samples underlying the run’s latency summary. |
| `detection_metrics.csv` | Recorded detection metrics for this directory’s model and protocol. In the results root, these are the deployed RF checkpoint’s RDD and Saudi aggregate results. |
| `frozen_selection.json` | Saved operating-point configuration for this specific experiment. Do not substitute it for the deployment configuration of another checkpoint. |
| `held_country_test_confusion.csv` | Saved bounding-box detection confusion matrix for the held-out-country test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `held_country_test_confusion.png` | Saved bounding-box detection confusion matrix for the held-out-country test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `held_country_test_example_1.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_2.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_3.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_4.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_5.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_6.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_per_class.csv` | Per-class detection metrics for the held-out-country test set; used to inspect class-specific performance hidden by aggregate scores. |
| `metrics.csv` | Original per-epoch training and validation metrics for this source run. |
| `run_identity.json` | Original training-run identity and configuration fingerprint for this controlled RF-DETR experiment. Used to check compatible settings when restoring or continuing a checkpoint. |
| `saudi_test_confusion.csv` | Saved bounding-box detection confusion matrix for the Saudi external test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `saudi_test_confusion.png` | Saved bounding-box detection confusion matrix for the Saudi external test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `saudi_test_example_1.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_2.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_3.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_4.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_5.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_6.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_per_class.csv` | Per-class detection metrics for the Saudi external test set; used to inspect class-specific performance hidden by aggregate scores. |
| `test_confusion.csv` | Saved bounding-box detection confusion matrix for the internal RDD test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `test_confusion.png` | Saved bounding-box detection confusion matrix for the internal RDD test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `test_example_1.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_2.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_3.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_4.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_5.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_6.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_per_class.csv` | Per-class detection metrics for the internal RDD test set; used to inspect class-specific performance hidden by aggregate scores. |
| `training_complete.json` | Recorded completed-epoch count and selected final-checkpoint identity for this run. It distinguishes final non-EMA weights from validation-best weights. |
| `training_config.json` | Saved training configuration and hyperparameters. It documents the source run rather than launching training itself. |
| `val_confusion.csv` | Saved bounding-box detection confusion matrix for the validation set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `val_confusion.png` | Saved bounding-box detection confusion matrix for the validation set, as a plot. Interpret using this run’s threshold and matching rules. |
| `val_example_1.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_2.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_3.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_4.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_5.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_6.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_per_class.csv` | Per-class detection metrics for the validation set; used to inspect class-specific performance hidden by aggregate scores. |
| `validation_F1_vs_confidence.csv` | Saved validation macro-F1 versus confidence-threshold table of operating points. Used to examine the threshold trade-off within this run. |
| `validation_F1_vs_confidence.png` | Saved validation macro-F1 versus confidence-threshold chart. Used to examine the threshold trade-off within this run. |

### `RoadScan_Public_Repo/results/RoadScan_20Epoch_Experiments/India_6f44f8600748/rf_small_512_noaug`

| File | Purpose |
|---|---|
| `cpu_latency.json` | Summary of the CPU timing measurements for this run and its recorded timing scope. |
| `cpu_latency_samples.csv` | Individual CPU timing samples underlying the run’s latency summary. |
| `detection_metrics.csv` | Recorded detection metrics for this directory’s model and protocol. In the results root, these are the deployed RF checkpoint’s RDD and Saudi aggregate results. |
| `frozen_selection.json` | Saved operating-point configuration for this specific experiment. Do not substitute it for the deployment configuration of another checkpoint. |
| `held_country_test_confusion.csv` | Saved bounding-box detection confusion matrix for the held-out-country test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `held_country_test_confusion.png` | Saved bounding-box detection confusion matrix for the held-out-country test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `held_country_test_example_1.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_2.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_3.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_4.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_5.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_6.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_per_class.csv` | Per-class detection metrics for the held-out-country test set; used to inspect class-specific performance hidden by aggregate scores. |
| `metrics.csv` | Original per-epoch training and validation metrics for this source run. |
| `run_identity.json` | Original training-run identity and configuration fingerprint for this controlled RF-DETR experiment. Used to check compatible settings when restoring or continuing a checkpoint. |
| `saudi_test_confusion.csv` | Saved bounding-box detection confusion matrix for the Saudi external test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `saudi_test_confusion.png` | Saved bounding-box detection confusion matrix for the Saudi external test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `saudi_test_example_1.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_2.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_3.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_4.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_5.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_6.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_per_class.csv` | Per-class detection metrics for the Saudi external test set; used to inspect class-specific performance hidden by aggregate scores. |
| `test_confusion.csv` | Saved bounding-box detection confusion matrix for the internal RDD test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `test_confusion.png` | Saved bounding-box detection confusion matrix for the internal RDD test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `test_example_1.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_2.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_3.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_4.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_5.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_6.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_per_class.csv` | Per-class detection metrics for the internal RDD test set; used to inspect class-specific performance hidden by aggregate scores. |
| `training_complete.json` | Recorded completed-epoch count and selected final-checkpoint identity for this run. It distinguishes final non-EMA weights from validation-best weights. |
| `training_config.json` | Saved training configuration and hyperparameters. It documents the source run rather than launching training itself. |
| `val_confusion.csv` | Saved bounding-box detection confusion matrix for the validation set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `val_confusion.png` | Saved bounding-box detection confusion matrix for the validation set, as a plot. Interpret using this run’s threshold and matching rules. |
| `val_example_1.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_2.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_3.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_4.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_5.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_6.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_per_class.csv` | Per-class detection metrics for the validation set; used to inspect class-specific performance hidden by aggregate scores. |
| `validation_F1_vs_confidence.csv` | Saved validation macro-F1 versus confidence-threshold table of operating points. Used to examine the threshold trade-off within this run. |
| `validation_F1_vs_confidence.png` | Saved validation macro-F1 versus confidence-threshold chart. Used to examine the threshold trade-off within this run. |

### `RoadScan_Public_Repo/results/RoadScan_20Epoch_Experiments/India_6f44f8600748/rf_small_640_noaug`

| File | Purpose |
|---|---|
| `cpu_latency.json` | Summary of the CPU timing measurements for this run and its recorded timing scope. |
| `cpu_latency_samples.csv` | Individual CPU timing samples underlying the run’s latency summary. |
| `detection_metrics.csv` | Recorded detection metrics for this directory’s model and protocol. In the results root, these are the deployed RF checkpoint’s RDD and Saudi aggregate results. |
| `frozen_selection.json` | Saved operating-point configuration for this specific experiment. Do not substitute it for the deployment configuration of another checkpoint. |
| `held_country_test_confusion.csv` | Saved bounding-box detection confusion matrix for the held-out-country test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `held_country_test_confusion.png` | Saved bounding-box detection confusion matrix for the held-out-country test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `held_country_test_example_1.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_2.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_3.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_4.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_5.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_example_6.jpg` | Saved visual detection example from the held-out-country test set for this specific run. Not a separate training image or a new evaluation metric. |
| `held_country_test_per_class.csv` | Per-class detection metrics for the held-out-country test set; used to inspect class-specific performance hidden by aggregate scores. |
| `metrics.csv` | Original per-epoch training and validation metrics for this source run. |
| `run_identity.json` | Original training-run identity and configuration fingerprint for this controlled RF-DETR experiment. Used to check compatible settings when restoring or continuing a checkpoint. |
| `saudi_test_confusion.csv` | Saved bounding-box detection confusion matrix for the Saudi external test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `saudi_test_confusion.png` | Saved bounding-box detection confusion matrix for the Saudi external test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `saudi_test_example_1.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_2.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_3.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_4.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_5.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_example_6.jpg` | Saved visual detection example from the Saudi external test set for this specific run. Not a separate training image or a new evaluation metric. |
| `saudi_test_per_class.csv` | Per-class detection metrics for the Saudi external test set; used to inspect class-specific performance hidden by aggregate scores. |
| `test_confusion.csv` | Saved bounding-box detection confusion matrix for the internal RDD test set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `test_confusion.png` | Saved bounding-box detection confusion matrix for the internal RDD test set, as a plot. Interpret using this run’s threshold and matching rules. |
| `test_example_1.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_2.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_3.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_4.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_5.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_example_6.jpg` | Saved visual detection example from the internal RDD test set for this specific run. Not a separate training image or a new evaluation metric. |
| `test_per_class.csv` | Per-class detection metrics for the internal RDD test set; used to inspect class-specific performance hidden by aggregate scores. |
| `training_complete.json` | Recorded completed-epoch count and selected final-checkpoint identity for this run. It distinguishes final non-EMA weights from validation-best weights. |
| `training_config.json` | Saved training configuration and hyperparameters. It documents the source run rather than launching training itself. |
| `val_confusion.csv` | Saved bounding-box detection confusion matrix for the validation set, as numeric counts. Interpret using this run’s threshold and matching rules. |
| `val_confusion.png` | Saved bounding-box detection confusion matrix for the validation set, as a plot. Interpret using this run’s threshold and matching rules. |
| `val_example_1.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_2.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_3.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_4.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_5.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_example_6.jpg` | Saved visual detection example from the validation set for this specific run. Not a separate training image or a new evaluation metric. |
| `val_per_class.csv` | Per-class detection metrics for the validation set; used to inspect class-specific performance hidden by aggregate scores. |
| `validation_F1_vs_confidence.csv` | Saved validation macro-F1 versus confidence-threshold table of operating points. Used to examine the threshold trade-off within this run. |
| `validation_F1_vs_confidence.png` | Saved validation macro-F1 versus confidence-threshold chart. Used to examine the threshold trade-off within this run. |

### `RoadScan_Public_Repo/results`

| File | Purpose |
|---|---|
| `data_summary.csv` | Recorded dataset split/image-count summary for the final model evaluation. |
| `detection_metrics.csv` | Recorded detection metrics for this directory’s model and protocol. In the results root, these are the deployed RF checkpoint’s RDD and Saudi aggregate results. |
| `model_card.json` | Recorded model provenance, evaluation scope, selected configuration and interpretation limitations. |
| `priority_band_distribution.csv` | Distribution of images among priority bands under the recorded scoring policy. |
| `priority_sensitivity.csv` | Sensitivity of the priority ranking and bands to policy-weight perturbations. This measures rule stability, not expert-certified urgency. |
| `priority_summary.csv` | Priority agreement/error statistics comparing predictions with the same priority rule applied to annotation boxes. |
| `validation_metrics.csv` | Recorded validation metrics associated with detector/threshold selection. |
| `verification.json` | Original saved verification information. Distinct from the newly added packaging verification record. |
| `yolo_model_search.csv` | Saved summary of historical YOLO model-search results. |

### `RoadScan_Public_Repo/results/deployment_training_backup/rf_small_training`

| File | Purpose |
|---|---|
| `completed.json` | Original trainer completion record from the supplied RF training backup, including completed-epoch count and checkpoint path. |
| `metrics.csv` | Original per-epoch training and validation metrics for this source run. |
| `roadscan_training_identity.json` | Recorded identity/provenance of the original training run, used to relate saved weights and settings to their source run. |
| `runtime_current.json` | Original runtime/version metadata saved with the RF training backup. |
| `training_config.json` | Saved training configuration and hyperparameters. It documents the source run rather than launching training itself. |

### `RoadScan_Public_Repo/results/figures`

| File | Purpose |
|---|---|
| `detection_example_01.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_02.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_03.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_04.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_05.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_06.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_07.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_example_08.png` | Saved visual example of road-damage detection from the original project results. Illustrative evidence, not an aggregate accuracy metric. |
| `detection_results_01.png` | Saved chart summarizing detection results from the original project. |
| `priority_01.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_02.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_03.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_04.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_05.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_06.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_07.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_08.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |
| `priority_09.png` | Saved visualization illustrating the original priority-scoring results. The numbered filename is preserved. |

### `RoadScan_Public_Repo/scripts`

| File | Purpose |
|---|---|
| `download_dataset.py` | Added dataset utility: downloads the actual repository Release asset or imports the supplied ZIP, locates the true dataset root and checks metadata and file hashes. |
| `evaluate.py` | Added command-line wrapper that evaluates the frozen deployment checkpoint on test/Saudi splits. Writes to a new output directory without retuning thresholds or replacing original results. |
| `predict_image.py` | Added command-line entry point for predicting one road image and exporting detections and priority information as JSON. |
| `verify_artifacts.py` | Added checksum verification utility. Checks the deployed RF checkpoint and, optionally, all original files listed in the repository checksum ledger. |

### `RoadScan_Public_Repo/src/roadscan`

| File | Purpose |
|---|---|
| `__init__.py` | Marks roadscan as a Python package. |
| `api.py` | Original API adapter exposing service operations through FastAPI. Optional service interface; separate from the Streamlit dashboard. |
| `duplicate_evaluation.py` | Legacy evaluation utilities for labelled same-defect/different-defect image pairs and embedding methods. E5 was canceled; no project E5 performance is claimed. |
| `evaluation.py` | Original detection-evaluation functions: box matching, per-class AP, precision/recall/F1, prediction caching and validation threshold selection. The added evaluate.py wrapper uses the saved thresholds. |
| `predict.py` | Original RF-DETR loading and prediction implementation. Checks class mappings, cleans/clips predicted boxes and applies the saved confidence policy. |
| `preprocessing.py` | Original dataset validation and label-reading utilities, including metadata, split, checksum and bounding-box checks. |
| `priority.py` | Original rule-based priority scoring based on damage type, relative box area and count. Confidence is a separate reliability/review signal. |
| `service.py` | Original persistent report/similarity service, including report lifecycle, location filtering and human duplicate review. Retained legacy functionality, separate from the final dashboard workflow. |
| `settings.py` | Loads project.json and resolves project-relative configuration paths. |

### `RoadScan_Public_Repo/tests`

| File | Purpose |
|---|---|
| `test_service.py` | Original unit tests for priority, persistence, review decisions and pair-evaluation behavior. Synthetic fixtures test software behavior, not model accuracy. |

## Generated internal directories

`.git/` is Git’s internal database, local configuration and LFS hooks. It is generated tooling state, not a project deliverable to upload manually. Git manages it automatically.
`__pycache__/` contains regenerated Python bytecode. `.local_artifacts/test_tmp/` contains disposable local test/cache files. Both are excluded from the file catalog and Git publishing.
The entire `.local_artifacts/` directory stays local. The catalog lists its preserved dataset and checkpoint assets, but these are not normal Git-tracked files. Publish the curated dataset through a Release and keep the supplied model Git LFS settings.

## Preservation and verification

The local source ledger documents copied originals and preserved workspace files, with SHA-256 hashes. The executed notebooks are directly in notebooks/; short modular templates are separated under notebooks/reference_templates/. The 10 original unit tests passed. Dataset verification checked 8,100 images and 16,200 image/label hashes. Training and full detector inference were not rerun during packaging. No GitHub push or Release publication was performed.
This guide describes the files as supplied. It does not turn historical notebook plans, partially executed stages or synthetic software fixtures into completed model experiments.
