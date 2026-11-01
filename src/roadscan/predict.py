"""Damage detection using the fine-tuned RF-DETR checkpoint."""
import json

from pathlib import Path

import numpy as np

from PIL import Image

from rfdetr import RFDETRSmall

def clean_predictions(pred,width,height,max_det=300):
    """Validate model outputs and clip valid boxes to the image boundary."""

    pred=np.asarray(pred,dtype=float).reshape(-1,6).copy()

    if not np.isfinite(pred).all(): raise ValueError('Non-finite detector output.')

    if len(pred):

        if not ((pred[:,5]==pred[:,5].astype(int))&(pred[:,5]>=0)&(pred[:,5]<4)).all():

            raise ValueError('Unexpected class IDs; expected 0..3.')

        if not ((pred[:,4]>=0)&(pred[:,4]<=1)).all(): raise ValueError('Invalid confidence values.')

        pred[:,[0,2]]=np.clip(pred[:,[0,2]],0,width)

        pred[:,[1,3]]=np.clip(pred[:,[1,3]],0,height)

        pred=pred[(pred[:,2]>pred[:,0])&(pred[:,3]>pred[:,1])]

    return pred[np.argsort(-pred[:,4],kind='stable')[:max_det]]


class RoadScanRF:
    """Load the detector once and apply the saved confidence policy."""

    def __init__(self,weights,classes,device='cpu',resolution=512,conf_floor=0.001,max_det=300):

        self.classes=list(classes); self.floor=conf_floor; self.max_det=max_det

        self.model=RFDETRSmall.from_checkpoint(str(weights),device=device,resolution=resolution,trust_checkpoint=True)

        if list(self.model.class_names)!=self.classes: raise ValueError(f'RF class mapping mismatch: {self.model.class_names}')

    def raw(self,image):
        """Retain low-confidence boxes for ranking-based detection evaluation."""

        with Image.open(image) as im: rgb=im.convert('RGB')

        detections=self.model.predict(rgb,threshold=self.floor)

        if not len(detections.xyxy): return np.empty((0,6),dtype=float)

        ids=np.asarray(detections.class_id)

        names=np.asarray(detections.data.get('class_name',['']*len(ids)),dtype=object)

        valid=(ids>=0)&(ids<len(self.classes))

        background=(ids==len(self.classes))&(names=='__background__')

        unexpected=~(valid|background)

        if unexpected.any(): raise ValueError(f'Unexpected RF classes: {ids[unexpected].tolist()}, names: {names[unexpected].tolist()}')

        pred=np.column_stack([detections.xyxy,detections.confidence,ids])[valid]

        return clean_predictions(pred,rgb.width,rgb.height,self.max_det)

    def predict(self,image,config):
        """Filter detections using the recorded per-class operating thresholds."""

        pred=self.raw(image);thresholds=np.asarray(config['thresholds'])

        return pred[pred[:,4]>=thresholds[pred[:,5].astype(int)]]


def load_deployment(folder,device='cpu'):
    """Read the deployment contract and construct the shared detector."""

    folder=Path(folder);config=json.loads((folder/'deployment_config.json').read_text())

    detector=RoadScanRF(folder/'rf_small.pth',config['class_names'],device=device,

        resolution=config['rf_resolution'],conf_floor=config['confidence_floor'],max_det=config['max_det'])

    return detector,config


from .priority import PRIORITY_CONFIG, detection_severity, priority_score
