import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from roadscan.service import RoadScanService, haversine_m, normalize_vectors
from roadscan.duplicate_evaluation import load_pairs, evaluate_pairs
from roadscan.priority import PRIORITY_CONFIG, priority_score
scope={'PRIORITY_CONFIG': PRIORITY_CONFIG, 'priority_score': priority_score}
CONFIG=dict(app_schema_version=1,priority=scope['PRIORITY_CONFIG'],
            duplicate_detection=dict(embedding_dimension=384,threshold_status='provisional',similarity_threshold=.8,gps_radius_m=100))

class FakeDetector:
    def __init__(self): self.empty=False
    def predict(self,path,config):
        return np.empty((0,6)) if self.empty else np.array([[0,0,20,20,.9,3]])

class FakeEncoder:
    dimension=384
    def encode(self,images):
        v=np.zeros((len(images),384),dtype=np.float32)
        v[:,0]=1
        return v

class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'package').mkdir()
        (self.root/'package'/'deployment_config.json').write_text(json.dumps(CONFIG))
        self.image=self.root/'input.png'
        Image.new('RGB',(40,40),'gray').save(self.image)
        self.detector=FakeDetector()
        self.encoder=FakeEncoder()
        self.service=self.open()
    def tearDown(self): self.temp.cleanup()
    def open(self):
        return RoadScanService(self.root/'package',self.root/'state',detector=self.detector,
            encoder=self.encoder,search_backend='numpy',priority_fn=scope['priority_score'])
    def submit(self,id,lat=24.,lon=46.):
        return self.service.submit_report(self.image,lat,lon,id)
    def test_query_before_insert_and_human_review(self):
        first=self.submit('first')
        self.assertEqual(first['decision'],'New Report')
        self.assertIsNone(first['match'])
        second=self.submit('second')
        self.assertEqual(second['decision'],'Likely Duplicate')
        self.assertEqual(second['match']['report_id'],'first')
        self.assertEqual(len(self.service.list_reports()),2)
        self.assertEqual(self.service.review_duplicate('second',True)['status'],'duplicate')
        self.assertEqual(len(self.service.list_reports()),1)
        third=self.submit('third')
        self.assertEqual(third['match']['report_id'],'first')
    def test_reject_and_resolve_lifecycle(self):
        self.submit('first');self.submit('second')
        self.service.review_duplicate('second',False)
        self.assertEqual(self.service.get_report('second')['decision'],'New Report')
        self.service.resolve_report('first')
        self.assertEqual(self.submit('third')['match']['report_id'],'second')
        self.service.resolve_report('second')
        self.assertEqual(self.service.get_report('third')['status'],'open')
        self.service.resolve_report('third')
        self.assertIsNone(self.submit('fourth')['match'])
    def test_restart_persistence_idempotency(self):
        self.submit('first')
        restored=self.open()
        self.assertEqual(len(restored.list_reports()),1)
        result=restored.submit_report(self.image,24.,46.,'first')
        self.assertEqual(result['report_id'],'first')
        self.assertEqual(len(restored.list_reports()),1)
        with self.assertRaises(ValueError): restored.submit_report(self.image,25.,46.,'first')
    def test_missing_gps_and_empty_reports_retained(self):
        self.assertEqual(self.submit('missing',None,None)['decision'],'Location Required')
        self.detector.empty=True
        result=self.submit('empty')
        self.assertEqual(result['decision'],'No Damage Detected')
        self.assertEqual(result['priority']['score'],0)
        self.assertTrue(result['needs_review'])
        self.assertEqual(len(self.service.list_reports()),2)
    def test_distance_and_gps_rejection(self):
        self.submit('first')
        self.assertIsNone(self.submit('far',25.,46.)['match'])
        for lat,lon in [(91,46),(24,181),(24,None),(float('nan'),46)]:
            with self.assertRaises(ValueError): self.submit('bad',lat,lon)
        self.assertAlmostEqual(haversine_m(24,46,24,46),0)
    def test_policy_change_rejects_old_state(self):
        new=copy.deepcopy(CONFIG);new['duplicate_detection']['similarity_threshold']=.9
        (self.root/'package'/'deployment_config.json').write_text(json.dumps(new))
        with self.assertRaises(ValueError): self.open()
    def test_invalid_vectors_rejected(self):
        with self.assertRaises(ValueError): normalize_vectors(np.zeros((1,384)),384)
        with self.assertRaises(ValueError): normalize_vectors(np.ones((1,20)),384)
    def test_priority_confidence_and_class_caps(self):
        f=scope['priority_score']
        weak=np.array([[0,0,20,20,.4,3],[0,0,5,5,.99,0],[0,0,5,5,.99,0]])
        self.assertTrue(f(weak,100,100)['needs_review'])
        confident=weak.copy();confident[:,4]=.99
        self.assertEqual(f(weak,100,100)['score'],f(confident,100,100)['score'])
        for c in range(4):
            self.assertEqual(f(np.tile([0,0,100,100,.9,c],(5,1)),100,100)['band'],'Critical')
    def test_pair_manifest_requires_disjoint_real_identities(self):
        paths=[]
        for i in range(8):
            p=self.root/f'{i}.png';Image.new('RGB',(10,10),(i*20,20,10)).save(p);paths.append(str(p))
        rows=[(paths[0],paths[1],1,'validation','v1','v1'),
              (paths[0],paths[2],0,'validation','v1','v2'),
              (paths[3],paths[4],1,'test','t1','t1'),
              (paths[3],paths[5],0,'test','t1','t2')]
        path=self.root/'pairs.csv'
        table=pd.DataFrame(rows,columns=['image_a','image_b','label','split','group_a','group_b'])
        table.to_csv(path,index=False)
        self.assertEqual(len(load_pairs(path)),4)
        table.loc[2,'group_a']='v1';table.loc[2,'group_b']='v1';table.loc[3,'group_a']='v1'
        table.to_csv(path,index=False)
        with self.assertRaises(ValueError): load_pairs(path)
    def test_evaluation_protocol_with_fixture_features(self):
        # Synthetic fixtures test metric code only. They are never a project evaluation result.
        rows=[];group_by_path={}
        for split in ['validation','test']:
            photos=[]
            for i in range(3):
                path=self.root/f'{split}_{i}.png'
                Image.new('RGB',(10,10),(i*50,0 if split=='validation' else 100,20)).save(path)
                photos.append(str(path))
                group_by_path[str(path)]=0 if i<2 else 1
            rows.extend([(photos[0],photos[1],1,split,split+'1',split+'1'),
                         (photos[0],photos[2],0,split,split+'1',split+'2')])
        path=self.root/'pairs.csv'
        pd.DataFrame(rows,columns=['image_a','image_b','label','split','group_a','group_b']).to_csv(path,index=False)
        def fixture(paths,*args):
            values=np.array([[1.,0.] if group_by_path[p]==0 else [0.,1.] for p in paths])
            hashes=np.array([[False]*64 if group_by_path[p]==0 else [True]*64 for p in paths])
            return dict(phash=hashes,resnet18=values,dinov2=values)
        with patch('roadscan.duplicate_evaluation.extract_methods',fixture):
            metrics,calibration=evaluate_pairs(path,self.root/'metrics',self.encoder)
        self.assertEqual(len(metrics),6)
        self.assertTrue((metrics.F1==1).all())
        self.assertTrue((metrics.recall_at_1==1).all())
        self.assertEqual(calibration['status'],'validation_calibrated')

if __name__=='__main__': unittest.main(verbosity=2)
