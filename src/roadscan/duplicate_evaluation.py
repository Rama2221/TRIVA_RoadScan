"""E5: run only with independently labelled real damage identities.

CSV columns: image_a,image_b,label,split,group_a,group_b.
split is validation or test. group IDs identify the physical defect, not its class.
Use different photographs for positives, not copies/augmentations of the same photo.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .service import normalize_vectors


def load_pairs(csv_path):
    path = Path(csv_path)
    table = pd.read_csv(path, dtype={'group_a': str, 'group_b': str})
    required = ['image_a', 'image_b', 'label', 'split', 'group_a', 'group_b']
    if not set(required) <= set(table):
        raise ValueError('Missing columns: '+', '.join(sorted(set(required)-set(table))))
    if table.empty:
        raise ValueError('The evaluation template contains no labelled pairs.')
    if table[required].isna().any().any():
        raise ValueError('Evaluation pairs must not contain missing fields.')
    if not set(table['label']) <= {0, 1}:
        raise ValueError('label must be 0 or 1.')
    if set(table['split']) != {'validation', 'test'}:
        raise ValueError('Both validation and test splits are required.')
    identities, digest_groups = {}, {}
    for side in ['a', 'b']:
        paths = []
        for _, row in table.iterrows():
            p = Path(row[f'image_{side}'])
            p = p if p.is_absolute() else path.parent/p
            p = p.resolve(strict=True)
            key, group = str(p), str(row[f'group_{side}']).strip()
            if not group:
                raise ValueError('Physical damage group IDs must not be empty.')
            identity = (group, row['split'])
            if key in identities and identities[key] != identity:
                raise ValueError('A photo has inconsistent damage identity or split.')
            identities[key] = identity
            with Image.open(p) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            if digest in digest_groups and digest_groups[digest] != identity:
                raise ValueError('Identical image content crosses damage identities or splits.')
            digest_groups[digest] = identity
            paths.append(key)
        table[f'image_{side}'] = paths
    for _, row in table.iterrows():
        if row.image_a == row.image_b:
            raise ValueError('Self-pairs do not evaluate different views of a damage.')
        if int(row.label) != int(row.group_a == row.group_b):
            raise ValueError('Pair label conflicts with physical defect group IDs.')
    val = table[table.split == 'validation']
    test = table[table.split == 'test']
    vg = set(val.group_a)|set(val.group_b)
    tg = set(test.group_a)|set(test.group_b)
    if vg & tg:
        raise ValueError('Physical damage groups must be disjoint between validation and test.')
    for split, part in table.groupby('split'):
        if set(part.label) != {0, 1}:
            raise ValueError(f'{split} needs positive and negative pairs.')
    table = table.drop_duplicates(['image_a', 'image_b', 'split']).reset_index(drop=True)
    return table


def _pil(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def extract_methods(paths, dino_encoder, device='cpu'):
    import imagehash
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    methods = {}
    hashes = np.vstack([np.asarray(imagehash.phash(_pil(p), hash_size=8).hash).reshape(-1) for p in paths])
    methods['phash'] = hashes
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    chunks = []
    for start in range(0, len(paths), 16):
        inputs = torch.stack([weights.transforms()(_pil(p)) for p in paths[start:start+16]]).to(device)
        with torch.inference_mode():
            chunks.append(model(inputs).float().cpu().numpy())
    methods['resnet18'] = normalize_vectors(np.vstack(chunks))
    chunks = [dino_encoder.encode([_pil(p) for p in paths[start:start+16]]) for start in range(0, len(paths), 16)]
    methods['dinov2'] = normalize_vectors(np.vstack(chunks), dino_encoder.dimension)
    return methods


def similarity(method, features, a, b):
    if method == 'phash':
        return 1.0 - np.mean(features[a] != features[b], axis=-1)
    return np.clip(np.sum(features[a] * features[b], axis=-1), -1, 1)


def evaluate_pairs(csv_path, output_dir, dino_encoder, device='cpu'):
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(csv_path)
    paths = sorted(set(pairs.image_a)|set(pairs.image_b))
    lookup = {p: i for i, p in enumerate(paths)}
    methods = extract_methods(paths, dino_encoder, device)
    ia, ib = pairs.image_a.map(lookup).to_numpy(), pairs.image_b.map(lookup).to_numpy()
    metrics, calibration = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, features in methods.items():
        scores = similarity(name, features, ia, ib)
        pairs[name+'_similarity'] = scores
        mask = (pairs.split == 'validation').to_numpy()
        # Exhaustive observed score boundaries; select using validation only.
        candidates = sorted(set(scores[mask].tolist()+[float(np.nextafter(scores[mask].max(), np.inf))]))
        best = None
        for threshold in candidates:
            p, r, f1, _ = precision_recall_fscore_support(
                pairs.loc[mask, 'label'], scores[mask] >= threshold, average='binary', zero_division=0)
            candidate = (float(f1), float(p), float(threshold))
            if best is None or candidate > best:
                best = candidate
        threshold = best[2]
        # Keep a JSON-safe threshold in [-1,1]. A threshold above 1 means no accepted pair.
        threshold = min(1.0, threshold)
        calibration[name] = dict(threshold=threshold, selected_on='validation', objective='F1; ties prefer precision, then higher threshold')
        for ax, split in zip(axes, ['validation', 'test']):
            m = (pairs.split == split).to_numpy()
            y, z = pairs.loc[m, 'label'].to_numpy(), scores[m]
            p, r, f1, _ = precision_recall_fscore_support(y, z >= threshold, average='binary', zero_division=0)
            fpr, tpr, roc_thresholds = roc_curve(y, z)
            pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=roc_thresholds)).to_csv(output/f'{name}_{split}_roc.csv', index=False)
            ax.plot(fpr, tpr, label=name)
            # Retrieval: query=a, gallery=all b photos in this split. Physical IDs supply relevance.
            part = pairs.loc[m]
            gallery = sorted(set(part.image_b))
            group_map = dict(zip(part.image_b, part.group_b))
            recall1, recall5, eligible = [], [], 0
            for query, group in part[['image_a', 'group_a']].drop_duplicates().itertuples(index=False, name=None):
                pool = [p for p in gallery if p != query]
                positives = np.array([group_map[p] == group for p in pool])
                if not positives.any():
                    continue
                eligible += 1
                a = np.full(len(pool), lookup[query], dtype=int)
                b = np.array([lookup[p] for p in pool])
                ranked = np.argsort(-similarity(name, features, a, b), kind='stable')
                recall1.append(float(positives[ranked[:1]].sum()/positives.sum()))
                recall5.append(float(positives[ranked[:5]].sum()/positives.sum()))
            metrics.append(dict(method=name, split=split, pairs=int(m.sum()), threshold=threshold,
                                precision=float(p), recall=float(r), F1=float(f1), ROC_AUC=float(roc_auc_score(y,z)),
                                recall_at_1=float(np.mean(recall1)) if eligible else None,
                                recall_at_5=float(np.mean(recall5)) if eligible else None,
                                eligible_queries=eligible, gallery_images=len(gallery)))
    for ax, split in zip(axes, ['validation', 'test']):
        ax.plot([0,1],[0,1],'k--',alpha=.3)
        ax.set(title=split, xlabel='False positive rate', ylabel='True positive rate')
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/'roc.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(output/'embedding_comparison.csv', index=False)
    pairs.to_csv(output/'pair_scores.csv', index=False)
    digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    result = dict(status='validation_calibrated', manifest_sha256=digest, methods=calibration,
                  retrieval_protocol='query=image_a; gallery=all unique image_b in same split; exclude self; relevant=same physical defect ID; macro fraction of relevant images retrieved at K',
                  scope='Visual crop retrieval only; GPS filtering is tested separately. No claim about unseen locations.')
    (output/'calibration.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return pd.DataFrame(metrics), result
