"""Persistent application service. Load once per worker; reuse for every upload.

The detector and embedding encoder are injected in tests. No training runs here.
Duplicate matches are suggestions requiring a human decision, never automatic merges.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(folder):
    digest = hashlib.sha256()
    for path in sorted(Path(folder).rglob('*')):
        if path.is_file():
            digest.update((path.relative_to(folder).as_posix()+file_sha256(path)).encode())
    return digest.hexdigest()


def validate_gps(latitude, longitude):
    if latitude is None and longitude is None:
        return None, None
    if latitude is None or longitude is None:
        raise ValueError('Supply both latitude and longitude, or neither.')
    latitude, longitude = float(latitude), float(longitude)
    if not (math.isfinite(latitude) and math.isfinite(longitude)
            and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('Invalid GPS coordinates.')
    return latitude, longitude


def haversine_m(lat1, lon1, lat2, lon2):
    a, b = math.radians(lat1), math.radians(lat2)
    dlat, dlon = b-a, math.radians(lon2-lon1)
    x = math.sin(dlat/2)**2 + math.cos(a)*math.cos(b)*math.sin(dlon/2)**2
    return 2 * 6371000.0 * math.asin(math.sqrt(min(1., max(0., x))))


def normalize_vectors(values, dimension=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or (dimension is not None and values.shape[1] != dimension):
        raise ValueError('Embedding shape does not match the configured encoder.')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite embedding.')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('Zero embedding is invalid for cosine similarity.')
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class DINOEncoder:
    """A separate frozen DINOv2 encoder for retrieval, not RF-DETR's decoder."""
    def __init__(self, model_path, device='cpu', local_files_only=True, batch_size=16):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_path), use_fast=False, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=local_files_only).to(device).eval()
        self.model.requires_grad_(False)
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, images):
        chunks = []
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(images=images[start:start+self.batch_size],
                                    return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                vector = self.model(**inputs).last_hidden_state[:, 0, :]
            chunks.append(vector.float().cpu().numpy())
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)
        return normalize_vectors(np.vstack(chunks), self.dimension)


def damage_crops(image, boxes):
    crops = []
    for x1, y1, x2, y2, confidence, class_id in boxes:
        box = (max(0, math.floor(x1)), max(0, math.floor(y1)),
               min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Detector returned an invalid crop.')
        crops.append(image.crop(box))
    return crops


class RoadScanService:
    def __init__(self, package, state_dir, device='cpu', detector=None, encoder=None,
                 search_backend='faiss', priority_fn=None):
        self.package, self.state_dir = Path(package), Path(state_dir)
        self.config = json.loads((self.package/'deployment_config.json').read_text(encoding='utf-8'))
        if self.config.get('app_schema_version') != 1:
            raise ValueError('An app-ready deployment package with schema version 1 is required.')
        self.dup = self.config['duplicate_detection']
        if self.dup['embedding_dimension'] != 384:
            raise ValueError('This package expects DINOv2 Small, dimension 384.')
        if self.config['priority']['weights']['confidence'] != 0:
            raise ValueError('This service expects confidence to be reported separately.')
        if self.dup['threshold_status'] not in {'provisional', 'validation_calibrated'}:
            raise ValueError('Unknown duplicate threshold status.')
        if not (0 < self.dup['gps_radius_m'] and -1 <= self.dup['similarity_threshold'] <= 1):
            raise ValueError('Invalid duplicate search configuration.')
        if search_backend not in {'faiss', 'numpy'}:
            raise ValueError('search_backend must be faiss or numpy.')
        self.search_backend = search_backend
        if detector is None:
            if file_sha256(self.package/'rf_small.pth') != self.config['protocol']['rf_hash']:
                raise ValueError('RF weights do not match the saved validation selection.')
        if encoder is None:
            if directory_fingerprint(self.package/'dinov2_encoder') != self.dup['embedding_fingerprint']:
                raise ValueError('Saved DINOv2 encoder/processor fingerprint does not match the configuration.')
        if detector is None or priority_fn is None:
            from .predict import load_deployment, priority_score
            if detector is None:
                detector, _ = load_deployment(self.package, device=device)
            if priority_fn is None:
                priority_fn = priority_score
        self.detector, self.priority_fn = detector, priority_fn
        self.encoder = encoder or DINOEncoder(self.package/'dinov2_encoder', device=device)
        self._inference_lock = threading.Lock()
        if self.encoder.dimension != self.dup['embedding_dimension']:
            raise ValueError('Encoder dimension mismatch.')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir/'images').mkdir(exist_ok=True)
        self.db_path = self.state_dir/'reports.sqlite3'
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        # Version the full deployment contract to prevent mixing vectors or policies.
        signature = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        with self._connect() as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript('''
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reports(
                    report_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, image TEXT NOT NULL,
                    latitude REAL, longitude REAL, priority REAL NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open','pending_duplicate','duplicate','resolved')),
                    suggested_report_id TEXT REFERENCES reports(report_id),
                    canonical_report_id TEXT REFERENCES reports(report_id),
                    result_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS damages(
                    report_id TEXT NOT NULL REFERENCES reports(report_id), damage_id INTEGER NOT NULL,
                    class_id INTEGER NOT NULL, embedding BLOB NOT NULL,
                    PRIMARY KEY(report_id,damage_id));
                CREATE TABLE IF NOT EXISTS audit(
                    event_id INTEGER PRIMARY KEY, report_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, action TEXT NOT NULL);
            ''')
            row = con.execute("SELECT value FROM metadata WHERE key='deployment_signature'").fetchone()
            if row and row['value'] != signature:
                raise ValueError('Deployment changed. Use a new state directory or explicitly migrate/re-embed existing reports.')
            con.execute("INSERT OR IGNORE INTO metadata VALUES('deployment_signature',?)", (signature,))

    def _search(self, con, vectors, latitude, longitude):
        if latitude is None or not len(vectors):
            return None
        # Only existing, open, canonical reports participate. Pending suggestions are not originals.
        rows = con.execute('''SELECT d.report_id,d.damage_id,d.class_id,d.embedding,r.latitude,r.longitude
                              FROM damages d JOIN reports r USING(report_id)
                              WHERE r.status='open' AND r.canonical_report_id IS NULL
                              AND r.latitude IS NOT NULL AND r.longitude IS NOT NULL''').fetchall()
        candidates = []
        for row in rows:
            distance = haversine_m(latitude, longitude, row['latitude'], row['longitude'])
            if distance <= self.dup['gps_radius_m']:
                candidates.append((row, distance))
        if not candidates:
            return None
        gallery = normalize_vectors(np.vstack([
            np.frombuffer(row['embedding'], dtype=np.float32) for row, _ in candidates
        ]), self.encoder.dimension)
        if self.search_backend == 'faiss':
            import faiss
            index = faiss.IndexFlatIP(self.encoder.dimension)
            index.add(gallery)
            scores, indices = index.search(vectors, 1)
            qi = int(scores[:, 0].argmax())
            gi, similarity = int(indices[qi, 0]), float(scores[qi, 0])
        else:
            matrix = vectors @ gallery.T
            qi, gi = map(int, np.unravel_index(matrix.argmax(), matrix.shape))
            similarity = float(matrix[qi, gi])
        row, distance = candidates[gi]
        return dict(report_id=row['report_id'], damage_id=int(row['damage_id']),
                    query_damage_id=qi, distance_m=float(distance),
                    cosine_similarity=float(np.clip(similarity, -1, 1)))

    def submit_report(self, image_path, latitude=None, longitude=None, report_id=None):
        started = time.perf_counter()
        latitude, longitude = validate_gps(latitude, longitude)
        report_id = report_id or uuid.uuid4().hex
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', report_id):
            raise ValueError('Invalid report ID.')
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        # Run expensive inference before the write transaction. Search and insert remain atomic.
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        payload = stream.getvalue()
        content_hash = hashlib.sha256(payload).hexdigest()
        with self._connect() as con:
            prior = con.execute('SELECT result_json FROM reports WHERE report_id=?', (report_id,)).fetchone()
        if prior:
            result = json.loads(prior['result_json'])
            if (result['image_sha256'], result['latitude'], result['longitude']) != (content_hash, latitude, longitude):
                raise ValueError('Report ID already belongs to a different request.')
            return self.get_report(report_id)
        destination = self.state_dir/'images'/f'{report_id}.png'
        # A temporary normalized input avoids keeping a file for a failed transaction.
        import tempfile
        # A worker reuses its models, serializing inference for predictable GPU memory use.
        with self._inference_lock:
            with tempfile.TemporaryDirectory(dir=self.state_dir) as temporary:
                normalized = Path(temporary)/'input.png'
                normalized.write_bytes(payload)
                inference_started = time.perf_counter()
                boxes = np.asarray(self.detector.predict(normalized, self.config), dtype=float).reshape(-1, 6)
                detector_seconds = time.perf_counter()-inference_started
            if not np.isfinite(boxes).all():
                raise ValueError('Non-finite predictions.')
            priority = self.priority_fn(boxes, image.width, image.height, self.config['priority'])
            embedding_started = time.perf_counter()
            vectors = normalize_vectors(self.encoder.encode(damage_crops(image, boxes)), self.encoder.dimension)
            embedding_seconds = time.perf_counter()-embedding_started
        if len(vectors) != len(boxes):
            raise ValueError('One embedding per detected damage is required.')
        with self._connect() as con:
            con.execute('BEGIN IMMEDIATE')
            # Multiple workers may finish the same request at the same time.
            existing = con.execute('SELECT result_json FROM reports WHERE report_id=?', (report_id,)).fetchone()
            if existing:
                result = json.loads(existing['result_json'])
                if (result['image_sha256'], result['latitude'], result['longitude']) != (content_hash, latitude, longitude):
                    raise ValueError('Report ID conflict.')
                return self.get_report(report_id)
            match = self._search(con, vectors, latitude, longitude)
            likely = bool(match and match['cosine_similarity'] >= self.dup['similarity_threshold'])
            decision = ('No Damage Detected' if not len(boxes) else
                        'Location Required' if latitude is None else
                        'Likely Duplicate' if likely else 'New Report')
            status = 'pending_duplicate' if likely else 'open'
            reasons = [x for x in priority.get('review_reason', '').split(';') if x and x != 'no_detections']
            if not len(boxes): reasons.append('no_damage_detected')
            if latitude is None: reasons.append('missing_gps_duplicate_check_skipped')
            if likely: reasons.append('duplicate_suggestion_requires_human_confirmation')
            result = dict(report_id=report_id, created_at=utc_now(), image=str(destination.resolve()),
                          image_sha256=content_hash, width=image.width, height=image.height,
                          latitude=latitude, longitude=longitude, boxes=boxes.tolist(),
                          priority=priority, decision=decision, status=status, match=match,
                          duplicate_threshold=self.dup['similarity_threshold'],
                          threshold_status=self.dup['threshold_status'],
                          needs_review=bool(reasons), review_reasons=reasons,
                          timing_seconds=dict(detector=detector_seconds, embedding=embedding_seconds,
                              processing_before_save=time.perf_counter()-started))
            destination.write_bytes(payload)
            con.execute('INSERT INTO reports VALUES(?,?,?,?,?,?,?,?,?,?)', (
                report_id, result['created_at'], str(destination.resolve()), latitude, longitude,
                priority['score'], status, match['report_id'] if likely else None, None,
                json.dumps(result, allow_nan=False)))
            con.executemany('INSERT INTO damages VALUES(?,?,?,?)', [
                (report_id, i, int(boxes[i, 5]), vectors[i].tobytes()) for i in range(len(vectors))])
            con.execute('INSERT INTO audit(report_id,created_at,action) VALUES(?,?,?)',
                        (report_id, utc_now(), 'submitted'))
        return result

    def get_report(self, report_id):
        with self._connect() as con:
            row = con.execute('SELECT * FROM reports WHERE report_id=?', (report_id,)).fetchone()
        if row is None:
            raise KeyError(report_id)
        result = json.loads(row['result_json'])
        result.update(status=row['status'], canonical_report_id=row['canonical_report_id'])
        if row['status'] in {'duplicate', 'resolved'}:
            result['decision'] = 'Confirmed Duplicate' if row['status'] == 'duplicate' else 'Resolved'
        elif row['status'] == 'open' and result['decision'] == 'Likely Duplicate':
            result['decision'] = 'New Report'
        result['duplicate_review_pending'] = row['status'] == 'pending_duplicate'
        # Review reasons are recorded at submission and retained as an audit of model uncertainty.
        return result

    def list_reports(self, include_closed=False):
        sql = 'SELECT report_id FROM reports'
        if not include_closed:
            sql += " WHERE status IN ('open','pending_duplicate')"
        sql += ' ORDER BY priority DESC,created_at,report_id'
        with self._connect() as con:
            ids = [row['report_id'] for row in con.execute(sql)]
        return [self.get_report(report_id) for report_id in ids]

    def review_duplicate(self, report_id, confirm):
        if not isinstance(confirm, bool):
            raise ValueError('confirm must be a boolean.')
        with self._connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT * FROM reports WHERE report_id=?', (report_id,)).fetchone()
            if row is None or row['status'] != 'pending_duplicate':
                raise ValueError('Only pending duplicate suggestions can be reviewed.')
            canonical = row['suggested_report_id'] if confirm else None
            if confirm:
                original = con.execute('SELECT status FROM reports WHERE report_id=?', (canonical,)).fetchone()
                if original is None or original['status'] != 'open':
                    raise ValueError('The suggested original is no longer open; review the report again.')
            con.execute('UPDATE reports SET status=?,canonical_report_id=? WHERE report_id=?',
                        ('duplicate' if confirm else 'open', canonical, report_id))
            con.execute('INSERT INTO audit(report_id,created_at,action) VALUES(?,?,?)',
                        (report_id, utc_now(), 'confirm_duplicate' if confirm else 'reject_duplicate'))
        return self.get_report(report_id)

    def resolve_report(self, report_id):
        with self._connect() as con:
            con.execute('BEGIN IMMEDIATE')
            changed = con.execute("UPDATE reports SET status='resolved' WHERE report_id=? AND status='open'",
                                  (report_id,)).rowcount
            if not changed:
                raise ValueError('Only an open original report can be resolved.')
            # Pending suggestions pointing to a resolved original become open for manual review.
            con.execute("UPDATE reports SET status='open',suggested_report_id=NULL WHERE status='pending_duplicate' AND suggested_report_id=?", (report_id,))
            con.execute('INSERT INTO audit(report_id,created_at,action) VALUES(?,?,?)',
                        (report_id, utc_now(), 'resolve_report'))
        return self.get_report(report_id)

    def export_queue(self, path):
        Path(path).write_text(json.dumps(self.list_reports(), indent=2, allow_nan=False), encoding='utf-8')

    def export_faiss_snapshot(self, folder):
        """Optional derived snapshot. SQLite remains authoritative for status and persistence."""
        import faiss
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            rows = con.execute("SELECT d.* FROM damages d JOIN reports r USING(report_id) WHERE r.status='open'").fetchall()
        index = faiss.IndexFlatIP(self.encoder.dimension)
        if rows:
            index.add(np.ascontiguousarray(np.vstack([np.frombuffer(r['embedding'], dtype=np.float32) for r in rows])))
        faiss.write_index(index, str(folder/'open_reports.faiss'))
        metadata = [dict(report_id=r['report_id'], damage_id=r['damage_id']) for r in rows]
        (folder/'index_rows.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return len(metadata)
