"""Import the exact local dataset ZIP, or download the published GitHub Release asset."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def find_root(folder):
    folder=folder.resolve()
    candidates=[p.parent for p in folder.rglob('dataset_version.json')
                if (p.parent/'images').is_dir() and (p.parent/'labels').is_dir()]
    if len(candidates)!=1:raise ValueError(f'Expected one dataset root, found {len(candidates)} under {folder}')
    return candidates[0]

def verify_dataset(folder):
    metadata=json.loads((folder/'dataset_version.json').read_text(encoding='utf-8-sig'))
    expected=json.loads((ROOT/'data/metadata/dataset_version.json').read_text(encoding='utf-8-sig'))
    if metadata!=expected:raise ValueError('Dataset metadata differs from the submitted definition.')
    integrity=json.loads((folder/'integrity_sha256.json').read_text(encoding='utf-8-sig'))
    for rel,expected_hash in integrity.items():
        p=(folder/rel).resolve()
        if not p.is_relative_to(folder.resolve()):raise ValueError('Unsafe integrity path')
        if not p.is_file() or digest(p)!=expected_hash:raise ValueError('Missing or modified dataset file: '+rel)
    print(f'Dataset verified: {metadata["total_images"]} images, {len(integrity)} checksums.')

def release_url(metadata):
    proc=subprocess.run(['git','remote','get-url','origin'],cwd=ROOT,capture_output=True,text=True)
    match=re.fullmatch(r'(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?',proc.stdout.strip())
    if proc.returncode or not match:
        raise ValueError('No GitHub origin found. Publish the repository and dataset Release first, or use --archive / --url.')
    return f'https://github.com/{match.group(1)}/releases/download/{metadata["release_tag"]}/{metadata["asset_name"]}'

def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--archive',type=Path)
    source.add_argument('--url')
    parser.add_argument('--output',type=Path,default=ROOT/'data/extracted')
    parser.add_argument('--verify-only',type=Path)
    args=parser.parse_args()
    if args.verify_only:
        verify_dataset(find_root(args.verify_only));return
    metadata=json.loads((ROOT/'data/dataset_release.json').read_text())
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Output is not empty; refusing to overwrite.')
    with tempfile.TemporaryDirectory(prefix='roadscan_dataset_') as tmp:
        archive=args.archive
        if archive is None:
            url=args.url or release_url(metadata)
            if not url.startswith('https://'):raise ValueError('HTTPS download URL required.')
            archive=Path(tmp)/metadata['asset_name']
            print('Downloading exact curated dataset from '+url)
            with urllib.request.urlopen(url) as response, archive.open('wb') as target:
                shutil.copyfileobj(response,target)
        if digest(archive)!=metadata['sha256']:raise ValueError('Dataset ZIP SHA-256 mismatch.')
        output.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                rel=PurePosixPath(info.filename.replace('\\','/'))
                if rel.is_absolute() or '..' in rel.parts or ':' in info.filename:raise ValueError('Unsafe ZIP path')
                target=(output/str(rel)).resolve()
                if not target.is_relative_to(output):raise ValueError('Unsafe ZIP path')
            z.extractall(output)
    folder=find_root(output)
    verify_dataset(folder)
    print('DATA_ROOT: '+str(folder))

if __name__=='__main__':
    main()
