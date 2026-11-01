"""Evaluate the saved deployment policy without retuning it or changing original results."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/evaluation')
    parser.add_argument('--splits', nargs='+', choices=['test','saudi_test'], default=['test','saudi_test'])
    args = parser.parse_args()
    output = args.output.resolve()
    if output == (ROOT/'results').resolve() or output in [(ROOT/'results').resolve().parent, ROOT]:
        raise SystemExit('Choose a new evaluation directory, not the original results root.')
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory is not empty. Choose a fresh path.')
    from download_dataset import find_root
    data = find_root(args.data)
    from verify_artifacts import verify_model
    verify_model()
    import numpy as np
    import pandas as pd
    from roadscan.predict import load_deployment
    from roadscan.evaluation import collect_records, ap_scores, operating, serialize_records, CODES
    detector, config = load_deployment(ROOT/'models/deployment', device='cpu')
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        records = collect_records(detector, data, split)
        ap = ap_scores(records)
        point = operating(records, np.asarray(config['thresholds']))
        for c, code in enumerate(CODES):
            rows.append(dict(split=split, code=code, AP50=float(ap[c,0]), AP50_95=float(ap[c].mean()),
                             **{key:float(point[key][c]) for key in ['P','R','F1','TP','FP','FN']}))
        rows.append(dict(split=split,code='MACRO',images=len(records),AP50=float(ap[:,0].mean()),
                         AP50_95=float(ap.mean()),P=float(point['P'].mean()),R=float(point['R'].mean()),F1=float(point['F1'].mean())))
        serialize_records(output/f'{split}_predictions.json', records)
    pd.DataFrame(rows).to_csv(output/'detection_metrics.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(data_root=str(data),thresholds=config['thresholds'],
        threshold_selection='Frozen deployment configuration; no retuning',resolution=config['rf_resolution']),indent=2))
    print(f'Saved new evaluation to {output}')

if __name__ == '__main__':
    main()
