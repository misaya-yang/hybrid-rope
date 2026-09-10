"""Checks against the exact downloaded Mini source, with tiny CPU attention.

Set DEEPSEEK_MINI_SOURCE to a directory containing modeling/configuration files
when the original source mirror is not in results/reference_position_20260909.
These tests make no language-model capability claim.
"""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import torch

from experiments.deepseek_mini_position.runtime import compress, forward, index_kl


def load_source():
    default = Path(__file__).resolve().parents[2]/'results/reference_position_20260909/deepseek_mini_source'
    root = Path(os.environ.get('DEEPSEEK_MINI_SOURCE', str(default)))
    if not (root/'modeling_deepseek_v4.py').exists():
        raise unittest.SkipTest('original Mini source is required for reference parity')
    name = 'mini_position_reference_fixture'
    package = types.ModuleType(name);package.__path__=[str(root)];sys.modules[name]=package
    loaded=[]
    for leaf in ('configuration_deepseek_v4','modeling_deepseek_v4'):
        spec=importlib.util.spec_from_file_location(name+'.'+leaf,root/(leaf+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        spec.loader.exec_module(module);loaded.append(module)
    return loaded


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_source, cls.source=load_source()

    def attention(self, ratio):
        config=self.config_source.DeepseekV4Config(vocab_size=32,hidden_size=16,
            num_hidden_layers=1,compress_ratios=[ratio],num_attention_heads=2,
            head_dim=8,q_lora_rank=8,o_lora_rank=8,o_groups=1,qk_rope_head_dim=4,
            index_head_dim=4,index_n_heads=2,index_topk=2,sliding_window=5)
        return self.source.DeepseekV4Attention(config,ratio)

    def inputs(self, module, length=37):
        x=torch.randn(2,length,16,requires_grad=True)
        pos=torch.arange(length)
        c,s=self.source.build_rope_cache(length,4,10000.,x.device,x.dtype)
        cc,ss=self.source.build_rope_cache(length,4,160000.,x.device,x.dtype)
        valid=torch.ones(2,length,dtype=torch.bool);valid[1,-3:]=False
        return x,pos,c,s,cc,ss,valid

    def options(self, geometry='replica', aux=False, source=False, dense=False):
        return dict(geometry=geometry,source_core=source,source_index=source,
                    indexer_aux=aux,dense_warmup=dense)

    def test_window_gather_matches_original_outputs_and_input_gradients(self):
        for ratio in (0,4,32):
            torch.manual_seed(482+ratio)
            original=self.attention(ratio);fast=copy.deepcopy(original)
            fast.position_workbench=self.options()
            data=self.inputs(original)
            x2=data[0].detach().clone().requires_grad_(True)
            expected=original(*data)
            actual=forward(fast,x2,*data[1:])
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            upstream=torch.randn_like(expected)
            (expected*upstream).sum().backward();(actual*upstream).sum().backward()
            torch.testing.assert_close(data[0].grad,x2.grad,rtol=3e-4,atol=3e-6)

    def test_auxiliary_objective_reaches_every_indexer_parameter(self):
        torch.manual_seed(821)
        module=self.attention(4);module.position_workbench=self.options('coherent',aux=True,dense=True)
        result=forward(module,*self.inputs(module))
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool(torch.isfinite(module.index_aux_loss)))
        module.index_aux_loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in module.indexer.parameters()))
        self.assertGreater(sum(p.grad.norm().item() for p in module.indexer.parameters()),0.)
        # The teacher target and inputs are detached: no auxiliary gradient
        # is silently attributed to the main compressor or query projection.
        self.assertTrue(all(p.grad is None for p in module.compressor.parameters()))
        self.assertIsNone(module.wq_a.weight.grad)

    def test_source_phase_is_finite_with_overlap_and_incomplete_blocks(self):
        for ratio in (4,32):
            torch.manual_seed(ratio)
            module=self.attention(ratio)
            module.position_workbench=self.options('coherent',aux=ratio==4,source=True,dense=True)
            for length in (3,ratio,ratio+3):
                data=self.inputs(module,length)
                result=forward(module,*data)
                loss=result.square().mean()
                if module.index_aux_loss is not None:loss=loss+module.index_aux_loss
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(bool(torch.isfinite(data[0].grad).all()))
                module.zero_grad(set_to_none=True)

    def test_source_phase_keeps_nonrotary_sum_before_normalization(self):
        comp=self.attention(32).compressor
        comp.norm=torch.nn.Identity()
        x=torch.randn(2,64,16)
        original=compress(comp,x,False,4,160000.)
        changed=compress(comp,x,True,4,160000.)
        torch.testing.assert_close(changed[...,:4],original[...,:4])
        self.assertGreater((changed[...,4:]-original[...,4:]).abs().max().item(),1e-5)


if __name__=='__main__':unittest.main()
