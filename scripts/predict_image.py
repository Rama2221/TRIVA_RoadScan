"""Predict one image with the original deployed model and saved thresholds."""
import argparse
import json
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'runtime/prediction.json')
    args = parser.parse_args()
    from roadscan.predict import load_deployment
    from roadscan.priority import priority_score
    from verify_artifacts import verify_model
    verify_model()
    detector, config = load_deployment(ROOT / 'models/deployment', device='cpu')
    started = time.perf_counter()
    predictions = detector.predict(args.image, config)
    elapsed = time.perf_counter() - started
    with Image.open(args.image) as image:
        priority = priority_score(predictions, image.width, image.height, config['priority'])
    result = dict(image_name=args.image.name, detections=predictions.tolist(),
                  columns=['x1','y1','x2','y2','confidence','class_id'],
                  priority=priority, prediction_filter_seconds=elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=lambda v:v.item()), encoding='utf-8')
    print(f'Saved {len(predictions)} detections to {args.output}')

if __name__ == '__main__':
    main()
