"""CPU orchestration/identifiability boundaries; fixtures are not model evidence."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from scripts.experiments import matched_transfer_round as M
from scripts.analysis.review_native_constrained_transfer import (
    decide, decide_v4, family_comparisons, native_transitions)
from scripts.analysis.audit_generation_transitions import join_annotations


class MatchedRoundTests(unittest.TestCase):
    def fixture(self):
        task=[];q=[]
        for i in range(128):
            for w in (0,1):
                gold=[10+2*i+w,0]
                q.append({'semantic_id':str(i),'world':w,'target_ids':gold,
                          'compact_prompt_ids':[1,2,3],'gold_margins':[.5,2.]})
                for cap in (2048,8192,16384):
                    task.append({'semantic_id':str(i),'world':w,'split':'train',
                                 'length_cap':cap,'target_ids':gold,'prompt_ids':[1,2,3] if cap==2048 else [4]*20})
        native=[{'id':f'{g}:{i}','group':g,'split':'train'} for g in M.GROUPS for i in range(128)]
        return task,{'rows':q},native

    def test_compact_targets_and_replay_are_matched_to_independent_schedule(self):
        task,q,native=self.fixture()
        result=M.exposure_contract(task,q,native)
        self.assertEqual(result['views'],768)
        self.assertEqual(result['answer_EOS_labels'],1536)
        self.assertLess(result['actual_compact_input_tokens'],result['actual_long_input_tokens'])
        # Independent reconstruction of archived per-stratum consumption order.
        rng=random.Random(42); pools={g:[r['id'] for r in native if r['group']==g] for g in M.GROUPS}
        for v in pools.values():rng.shuffle(v)
        rstage=[pools[g][i] for step in range(32) for g in M.GROUPS for i in (2*step,2*step+1)]
        cursor=dict.fromkeys(M.GROUPS,64);tstage=[]
        for step in range(96):
            for gi in (step%4,(step+1)%4):
                g=M.GROUPS[gi];tstage.append(pools[g][cursor[g]]);cursor[g]+=1
        self.assertEqual(result['native_exposure_sha256'],M.digest(rstage+tstage))
        self.assertEqual(len(set(tstage)),192)
        self.assertTrue(set(rstage).isdisjoint(tstage))
        self.assertEqual(result,M.exposure_contract(task,q,native))

    def test_target_or_duplicate_cell_is_rejected(self):
        task,q,native=self.fixture()
        changed=copy.deepcopy(task);changed[1]['target_ids']=[123,0]
        with self.assertRaisesRegex(ValueError,'target'):M.exposure_contract(changed,q,native)
        changed=copy.deepcopy(task);changed[3]=copy.deepcopy(changed[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):M.exposure_contract(changed,q,native)

    def test_commands_cannot_resume_change_recipe_or_open_test(self):
        c={key:'/fixture/'+key for key in ('output_root','checkpoint','checkpoint_contract','tasks',
          'native_pool','engine_root','teacher_cache','controls','native_baseline','task_baseline','baseline_eval_engine')}
        c['python']=sys.executable
        controls={'arms':{arm:{'path':arm+'.npy','rotary_amplitude':1.12} for arm in ('Z','Y')}}
        for case in M.CASES:
            stages=M.commands(c,case,controls); train=stages[0]['argv']
            self.assertEqual(train[train.index('--stop-after-step')+1],'128')
            self.assertEqual('--compact-only' in train,case=='N_compact')
            self.assertEqual('--table' in train,case!='N_compact')
            for stage in stages:
                argv=stage['argv']
                self.assertNotIn('--resume',argv);self.assertNotIn('--prefix-lm',argv)
                self.assertNotIn('test',argv);self.assertIn('--checkpoint',argv)
                if '--split' in argv:self.assertEqual(argv[argv.index('--split')+1],'validation')
            self.assertEqual([s['name'] for s in stages if s['name']=='native32'],[] if case=='N_compact' else ['native32'])

    def test_run_without_authorization_does_not_read_plan_or_spawn(self):
        with self.assertRaisesRegex(ValueError,'authorized'):
            M.execute(Path('/nonexistent/plan'),['N_compact'],False)

    def test_subprocess_timeout_releases_owned_process(self):
        with tempfile.TemporaryDirectory() as d:
            stage={'name':'fixture','seconds':-119.95,'argv':[sys.executable,'-c','import time;time.sleep(10)']}
            with self.assertRaises(subprocess.TimeoutExpired):
                M.call_stage(stage,d,Path(d)/'log')
            self.assertTrue((Path(d)/'log').exists())

    def test_v4_family_local_gating_preserves_legacy_verdict(self):
        cells={'single_evidence':{'native_compact_groups':26,'compact':1.,'near':.5,'far':.7},
               'double_evidence':{'native_compact_groups':4,'compact':1.,'near':.5,'far':0.}}
        self.assertEqual(decide(True,True,cells,128)[0],'UNRESOLVED_CONTROLS')
        self.assertEqual(decide_v4(True,True,cells,128)[0],'VALIDATION_FEASIBLE_CONFIRMATION_PENDING')
        self.assertEqual(decide_v4(False,True,cells,128)[0],'STOP_NATIVE_DAMAGE')
        self.assertEqual(decide_v4(True,False,cells,128)[0],'POINT_FEASIBLE_CONFIRMATION_PENDING')
        self.assertEqual(decide_v4(True,True,cells,32)[0],'DIAGNOSTIC_CHECKPOINT_ONLY')
        cells['single_evidence']['far']=0
        self.assertEqual(decide_v4(True,True,cells,128)[0],'STOP_ZERO_PRIMARY_GENERATION')

    def test_source_cluster_interaction_keeps_near_and_far_paired(self):
        a={};b={}
        for sid in ('a','b'):
            for layout in ('compact','near','far'):
                k=('single_evidence',sid,layout,2048 if layout=='compact' else 16384)
                a[k]=layout!='far';b[k]=True
        result=family_comparisons(a,b,{'a':'same_source','b':'same_source'},resamples=20)['single_evidence']
        self.assertEqual(result['source_clusters'],1)
        self.assertEqual(result['near_far_interaction'],1)
        self.assertEqual(result['ci95']['near_far_interaction'],[1,1])
        self.assertEqual(result['all_groups']['far']['gained'],2)

    def test_native_score_ratio_and_original_correct_retention_are_distinct(self):
        a=[];b=[]
        for task in ('instruction','reasoning','position_format'):
            for i,(x,y) in enumerate(((1,0),(1,1),(0,1))):
                r={'task':task,'asset_sha256':f'{task}{i}','row_id':str(i)}
                a.append({**r,'score_eos':x});b.append({**r,'score_eos':y})
        result=native_transitions(a,b)['position_format']
        self.assertEqual(result['score_retention'],1)
        self.assertEqual(result['original_correct_preserved'],.5)
        self.assertEqual(result['lost_ids'],['0'])

    def test_annotation_requires_all_cases_and_does_not_use_gold_substrings(self):
        case={'prompt_id':'p','output':'It is not GOLD.','accepted_answers':['GOLD']}
        original=[{**case,'case_id':str(i)} for i in range(2)]
        mapping=[{'case_id':str(i),'arm':arm,'key':['f','s',0,'far',16384],
                  'strict':False,'termination_valid':True} for i,arm in enumerate(('baseline','candidate'))]
        labels=[{**r,'semantic_correct':'incorrect','format_compliant':'no','reason':'Explicit negation.'} for r in original]
        paired=join_annotations(original,labels,mapping)
        self.assertEqual(paired[0]['baseline']['semantic_correct'],'incorrect')
        with self.assertRaisesRegex(ValueError,'incomplete'):join_annotations(original,labels[:1],mapping)
        labels[0]['semantic_correct']=None
        with self.assertRaisesRegex(ValueError,'labels'):join_annotations(original,labels,mapping)


if __name__=='__main__':unittest.main()
