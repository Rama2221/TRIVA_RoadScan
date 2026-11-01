# Notebooks

This index matches the notebooks currently present in this folder. No notebook cells, source code, execution counts, or saved outputs were changed while packaging the model weights.

## Main notebooks

| Notebook | Scope and saved state |
|---|---|
| `01_Preprocessing.ipynb` | Dataset exploration and validation, with saved tables/figures. |
| `02_ResNet18_Baseline.ipynb` | Image-level multi-label ResNet18 baseline. Saved outputs are included. The standalone weights artifact is `models/baselines/resnet18_baseline_weights.pt`; it records epoch 20 and is weights-only, not an optimizer/resume checkpoint. |
| `03_Detection_RFDETR.ipynb` | RF-DETR evaluation and priority scoring. Priority logic comes from `src/roadscan/priority.py`. Default `MODE="load"` evaluates saved weights and does not train. |
| `04_Similarity.ipynb` | Optional similarity/duplicate-method exploration; currently a short unexecuted template. It is not required by the detector or application. |
| `05_Application.ipynb` | Short unexecuted application entry point. The working Streamlit app is `notebooks/application/app.py`. |
| `06-Roadscan-AllExperiments.ipynb` | Controlled model experiments with saved outputs. Its held-out-country split is distinct from the deployment evaluation. |

## Archived experiments and templates

`previous_models/` contains historical YOLO and held-out-country notebooks. They preserve their original outputs and may contain machine-specific paths; treat them as experiment records and update paths before attempting to rerun them.

`reference_templates/` contains short, mostly unexecuted notebook templates. `01_Preprocessing_Workspace_Snapshot.ipynb` preserves an older workspace snapshot whose saved output includes a `FileNotFoundError` from a local dataset path. It is retained as a record, not as the current preprocessing workflow.

The deployment checkpoint, the RF-DETR 20-epoch training backup, and the ResNet18 baseline weights are distinct artifacts. See the root `README.md` for their paths and use. `docs/NOTEBOOK_RUN_RESTORATION.json` is a historical packaging record and may mention earlier filenames and locations.
