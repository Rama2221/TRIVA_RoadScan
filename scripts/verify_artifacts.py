"""Verify original-file checksums without importing the ML stack."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_model():
    selected = json.loads((ROOT/'configs/detector_selection.json').read_text(encoding='utf-8'))
    weights = ROOT/'models/deployment/rf_small.pth'
    if not weights.exists():
        raise SystemExit('RF weights are missing. Run git lfs pull.')
    if digest(weights) != selected['weights_sha256']:
        raise SystemExit('RF weight mismatch or unresolved Git LFS pointer. Run git lfs pull.')
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-originals', action='store_true')
    args = parser.parse_args()
    verify_model()
    count = 1
    if args.all_originals:
        for entry in json.loads((ROOT/'ORIGINAL_FILES_SHA256.json').read_text()):
            path = ROOT/entry['path']
            if not path.is_file() or digest(path) != entry['sha256']:
                raise SystemExit('Original file mismatch: '+entry['path'])
            count += 1
    print(f'Artifact verification passed ({count} checks).')

if __name__ == '__main__':
    main()
