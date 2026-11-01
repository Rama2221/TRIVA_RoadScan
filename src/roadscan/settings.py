"""Portable paths and model settings shared across the RoadScan stages."""
import json
from pathlib import Path

PATH_KEYS={'dataset_source','dataset_root','output_dir','work_dir','rf_run_source',
           'rf_backup','rf_weights','frozen_selection','deployment_package','state_dir','duplicate_pairs'}

def load_settings(root):
    """Resolve relative configuration paths against the project directory."""
    root=Path(root).resolve()
    config=json.loads((root/'configs/project.json').read_text(encoding='utf-8'))
    for key in PATH_KEYS:
        value=config.get(key,'')
        if value:
            path=Path(value).expanduser()
            config[key]=path if path.is_absolute() else (root/path).resolve()
        else:
            config[key]=''
    return config
