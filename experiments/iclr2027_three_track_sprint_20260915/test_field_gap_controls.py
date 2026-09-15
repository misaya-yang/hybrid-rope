"""Regression checks for evidence qualification and clustered QA inference."""
import unittest
from copy import deepcopy
import numpy as np
from .e1_experimental_audit import qualify_runtime
from experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report import bootstrap

class ControlsTest(unittest.TestCase):
    def fixture(self):
        contract={'batch_size':1,'prefill_chunk_size':8192,'row_ids':list(range(390))}
        raw=[{'eval_id':i,'prompt_sha256':str(i),'input_tokens':8192,'length_cap':8192,'task':'task','references':['a']}for i in range(390)]
        return contract,raw

    def test_missing_metadata_cannot_pass(self):
        c,r=self.fixture()
        self.assertEqual(qualify_runtime(c,c,r,r)['status'],'QUALIFIED_ONLY')

    def test_cross_batch_and_reordering_remain_qualified(self):
        c,r=self.fixture();other=deepcopy(c);other['batch_size']=2;other['row_ids'].reverse()
        self.assertEqual(qualify_runtime(c,other,r,list(reversed(r)))['status'],'QUALIFIED_ONLY')

    def test_prompt_drift_fails(self):
        c,r=self.fixture();other=deepcopy(r);other[3]['prompt_sha256']='changed'
        with self.assertRaises(ValueError):qualify_runtime(c,c,r,other)

    def test_duplicate_raw_fails(self):
        c,r=self.fixture()
        with self.assertRaises(ValueError):qualify_runtime(c,c,r,r[:-1]+[r[0]])

    def test_backend_mismatch_fails(self):
        c,r=self.fixture();a=deepcopy(c);b=deepcopy(c)
        a['backend']='one';b['backend']='two'
        self.assertEqual(qualify_runtime(a,b,r,r)['status'],'FAIL')

    def test_family_interval_and_unequal_clusters(self):
        # One three-question document and one single-question document per task.
        tasks=('hotpotqa','2wikimqa','qasper','narrativeqa','multifieldqa_en')
        panel={};candidate={};baseline={}
        for task in tasks:
            for i in range(4):
                key=f'{task}_{i}';panel[key]={'task':task,'document_cluster_id':f'{task}_{int(i==3)}'}
                candidate[key]={'whole_response_f1':float(i<3)};baseline[key]={'whole_response_f1':0.0}
        result=bootstrap(panel,candidate,baseline,list(panel),draws=5000,family_size=2)
        lo,hi=result['ci95'];flo,fhi=result['familywise_ci95_bonferroni']
        self.assertLessEqual(flo,lo);self.assertGreaterEqual(fhi,hi)
        self.assertGreater(result['bootstrap_mean'],.5)
        self.assertEqual(result['comparison_family_size'],2)

if __name__=='__main__':unittest.main()
