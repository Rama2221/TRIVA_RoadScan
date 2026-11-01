
import json
import numpy as np

# Priority = how urgent the DAMAGE is. It does not depend on model confidence by default
# (confidence is reported next to it as a reliability indicator). All numbers below are EXPERT
# ASSUMPTIONS, not learned from data: review them with a road-maintenance specialist.
PRIORITY_CONFIG = dict(
    type_weight={'0': 0.40, '1': 0.50, '2': 0.80, '3': 1.00},          # D00, D10, D20, D40 severity
    size_ref_area={'0': 0.05, '1': 0.05, '2': 0.25, '3': 0.03},        # box area / image area that counts as "full size", per class
    severity_weights=dict(type=0.40, size=0.60),
    weights=dict(severity=0.80, count=0.20, confidence=0.0),           # must sum to 1; confidence weight 0 = report only
    count_ref=5,                                                       # 5+ detections = full count score (log-scaled)
    review_conf=0.50,
    bands=[[25, 'Low'], [50, 'Medium'], [75, 'High'], [101, 'Critical']])

def detection_severity(pred, width, height, cfg=None):
    """Per-detection severity 0-1 = 0.40 x type weight + 0.60 x relative size, so type and size belong to the SAME damage.
    pred rows: x1,y1,x2,y2,confidence,class_id. The box area is only an approximate size proxy
    (camera angle/distance matter); it is not the real damaged area or depth."""
    cfg = cfg or PRIORITY_CONFIG
    pred = np.asarray(pred, dtype=float).reshape(-1, 6)
    if len(pred) == 0:
        return np.empty(0), np.empty(0)
    area = np.maximum(pred[:, 2] - pred[:, 0], 0) * np.maximum(pred[:, 3] - pred[:, 1], 0) / float(width * height)
    ref = np.array([cfg['size_ref_area'][str(int(c))] for c in pred[:, 5]])
    size = np.minimum(1.0, np.sqrt(area / ref))
    type_w = np.array([cfg['type_weight'][str(int(c))] for c in pred[:, 5]])
    # the type and size still refer to the same detection.
    sw = cfg['severity_weights']
    return sw['type'] * type_w + sw['size'] * size, size

def priority_score(pred, width, height, cfg=None):
    """Image-level score 0-100 from detections already filtered by the frozen confidence thresholds."""
    cfg = cfg or PRIORITY_CONFIG
    pred = np.asarray(pred, dtype=float).reshape(-1, 6)
    if len(pred) == 0:
        return dict(score=0.0, band='None', n=0, severity=0.0, count=0.0, mean_conf=0.0,
                    worst_class=-1, worst_size=0.0, worst_conf=None,
                    needs_review=False, review_reason='no_detections')
    sev, size = detection_severity(pred, width, height, cfg)
    worst = int(sev.argmax())                                   # the single most serious damage
    count = float(min(1.0, np.log1p(len(pred)) / np.log1p(cfg['count_ref'])))
    mean_conf = float(pred[:, 4].mean())
    # include all tied worst-severity detections so a weak tie cannot be hidden.
    worst_conf = float(pred[np.isclose(sev, sev[worst], rtol=0, atol=1e-12), 4].min())
    reasons = []
    if mean_conf < cfg['review_conf']: reasons.append('low_mean_confidence')
    if worst_conf < cfg['review_conf']: reasons.append('low_worst_damage_confidence')
    w = cfg['weights']
    score = 100.0 * (w['severity'] * float(sev[worst]) + w['count'] * count + w['confidence'] * mean_conf)
    score = float(np.clip(score, 0, 100))
    band = next(name for limit, name in cfg['bands'] if score < limit)
    return dict(score=score, band=band, n=int(len(pred)), severity=float(sev[worst]), count=count,
                mean_conf=mean_conf, worst_class=int(pred[worst, 5]), worst_size=float(size[worst]),
                worst_conf=worst_conf, needs_review=bool(reasons),
                review_reason=';'.join(reasons))
