"""Validation selection and bounding-box evaluation for the trained detector."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .predict import RoadScanRF
from .preprocessing import CLASS_NAMES,list_images,read_boxes,RDD_NAMES

NC=4
MAX_DET=300
IOU_LEVELS=np.linspace(.5,.95,10)
THRESHOLD_GRID=np.round(np.arange(.05,.81,.01),2)
CODES=['D00','D10','D20','D40']
def box_iou(a,b):
    a=np.asarray(a).reshape(-1,4); b=np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1)
    bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def match(truth,pred,levels=IOU_LEVELS):
    flags=np.zeros((len(pred),len(levels)),dtype=bool)
    overlaps=box_iou(pred[:,:4],truth[:,1:])
    for j,level in enumerate(levels):
        used=set()
        for i in np.argsort(-pred[:,4],kind='stable'):
            candidates=[g for g in range(len(truth)) if g not in used
                and int(truth[g,0])==int(pred[i,5]) and overlaps[i,g]>=level]
            if candidates:
                g=max(candidates,key=lambda g:overlaps[i,g]); used.add(g); flags[i,j]=True
    return flags

def packed(records):
    if isinstance(records, tuple): return records
    pred=np.concatenate([r['pred'] for r in records])
    target=np.concatenate([r['truth'][:,0] for r in records]).astype(int)
    for r in records:
        if 'matches' not in r: r['matches']=match(r['truth'],r['pred'])
    tp=np.concatenate([r['matches'] for r in records])
    return pred,target,tp

def operating(records,thresholds):
    pred,target,tp=packed(records)
    cls=pred[:,5].astype(int); thresholds=np.broadcast_to(thresholds,(NC,))
    keep=pred[:,4]>=thresholds[cls]
    support=np.bincount(target,minlength=NC)
    detected=np.bincount(cls[keep],minlength=NC)
    true=np.bincount(cls[keep & tp[:,0]],minlength=NC)
    p=true/np.maximum(detected,1); r=true/np.maximum(support,1)
    f=2*p*r/np.maximum(p+r,1e-12)
    return dict(P=p,R=r,F1=f,TP=true,FP=detected-true,FN=support-true,support=support)

def ap_scores(records):
    images=[]; annotations=[]; detections=[]
    for image_id,rec in enumerate(records,1):
        images.append(dict(id=image_id,width=rec['width'],height=rec['height']))
        for c,x1,y1,x2,y2 in rec['truth']:
            annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=int(c),
                bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],
                area=float((x2-x1)*(y2-y1)),iscrowd=0))
        for x1,y1,x2,y2,score,c in rec['pred']:
            detections.append(dict(image_id=image_id,category_id=int(c),score=float(score),
                bbox=[float(x1),float(y1),float(max(0,x2-x1)),float(max(0,y2-y1))]))
    ap=np.zeros((NC,10))
    if not detections: return ap
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=dict(info={},images=images,annotations=annotations,
            categories=[dict(id=i,name=name) for i,name in enumerate(CLASS_NAMES)])
        gt.createIndex(); dt=gt.loadRes(detections)
        evaluator=COCOeval(gt,dt,'bbox')
        evaluator.params.catIds=list(range(NC)); evaluator.params.maxDets=[1,10,MAX_DET]
        evaluator.evaluate(); evaluator.accumulate()
    # precision axes: IoU, recall, class, area, max-detections.
    precision=evaluator.eval['precision'][:,:,:,0,-1]
    for c in range(NC):
        for j in range(10):
            values=precision[j,:,c]; values=values[values>=0]
            ap[c,j]=float(values.mean()) if len(values) else 0.0
    return ap

def tune(records):
    # Search global and class thresholds without re-running inference.
    stats=packed(records)
    metrics=[operating(stats,float(t)) for t in THRESHOLD_GRID]
    scores=np.array([m['F1'] for m in metrics])
    common=float(THRESHOLD_GRID[scores.mean(1).argmax()])
    per_class=THRESHOLD_GRID[scores.argmax(0)].copy()
    support=metrics[0]['support']
    per_class[support<20]=common
    common_f1=float(operating(stats,common)['F1'].mean())
    per_f1=float(operating(stats,per_class)['F1'].mean())
    # Require one percentage point of macro F1 gain to add class-specific thresholds.
    use_per=per_f1>=common_f1+0.01
    thresholds=per_class if use_per else np.full(NC,common)
    return dict(thresholds=thresholds.tolist(),per_class=bool(use_per),
                val_macro_F1=float(operating(stats,thresholds)['F1'].mean()))

def file_hash(path):
    """Identify model and dataset bytes used by an evaluation."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def collect_records(detector,data_root,split):
    """Load paired labels and run the shared detector for one split."""
    data_root=Path(data_root);records=[]
    for path in list_images(data_root,split):
        label=data_root/'labels'/split/path.relative_to(data_root/'images'/split).with_suffix('.txt')
        with Image.open(path) as image:width,height=image.size
        truth=np.asarray(read_boxes(label,(width,height),RDD_NAMES),dtype=float).reshape(-1,5)
        records.append(dict(image=path.relative_to(data_root).as_posix(),width=width,height=height,
            truth=truth,pred=detector.raw(path)))
    if not records:raise ValueError(f'No paired images in {split}')
    return records

def serialize_records(path,records):
    """Retain raw predictions for priority analysis and visual review."""
    values=[dict(r,truth=r['truth'].tolist(),pred=r['pred'].tolist()) for r in records]
    Path(path).write_text(json.dumps(values,allow_nan=False),encoding='utf-8')

def load_records(path):
    """Restore numerical predictions without repeating model inference."""
    values=json.loads(Path(path).read_text(encoding='utf-8'))
    for r in values:
        r['truth']=np.asarray(r['truth'],float).reshape(-1,5)
        r['pred']=np.asarray(r['pred'],float).reshape(-1,6)
    return values

def evaluate_detector(settings):
    """Select confidence on validation and evaluate each test domain separately."""
    root=Path(settings['dataset_root']);out=Path(settings['output_dir']);out.mkdir(parents=True,exist_ok=True)
    detector=RoadScanRF(settings['rf_weights'],CLASS_NAMES,device=settings['device'],resolution=512)
    validation=collect_records(detector,root,'val')
    selected=tune(validation)
    digest=file_hash(settings['rf_weights'])
    frozen=dict(selected,weights_sha256=digest,class_names=CLASS_NAMES,rf_resolution=512,
        confidence_floor=.001,max_det=MAX_DET,selected_on='validation')
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    grid=[dict(threshold=float(t),macro_F1=float(operating(validation,float(t))['F1'].mean())) for t in THRESHOLD_GRID]
    pd.DataFrame(grid).to_csv(out/'validation_confidence_curve.csv',index=False)
    rows=[]
    for split in ['test','saudi_test']:
        records=collect_records(detector,root,split)
        ap=ap_scores(records);point=operating(records,np.asarray(selected['thresholds']))
        for c,code in enumerate(CODES):
            rows.append(dict(split=split,code=code,AP50=float(ap[c,0]),AP50_95=float(ap[c].mean()),
                **{k:float(point[k][c]) for k in ['P','R','F1','TP','FP','FN']}))
        rows.append(dict(split=split,code='MACRO',images=len(records),AP50=float(ap[:,0].mean()),AP50_95=float(ap.mean()),
            P=float(point['P'].mean()),R=float(point['R'].mean()),F1=float(point['F1'].mean())))
        serialize_records(out/f'{split}_predictions.json',records)
    frame=pd.DataFrame(rows);frame.to_csv(out/'detection_metrics.csv',index=False)
    return frame
