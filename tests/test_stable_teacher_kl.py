"""Work-machine numerical tests; skip rather than install Torch on the personal PC."""
import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.lib.rope.generation_contract import stable_teacher_kl


def test_identical_wide_bf16_logits_have_exactly_zero_first_order_gradient():
    torch.manual_seed(42)
    teacher=torch.randn(2,3,151936,dtype=torch.bfloat16).float()
    student=teacher.clone().requires_grad_(True)
    loss=stable_teacher_kl(student,teacher);loss.backward()
    assert loss.item()==0.
    assert torch.count_nonzero(student.grad).item()==0


def test_nonidentity_value_and_analytic_gradient_match_kl():
    torch.manual_seed(43)
    teacher=torch.randn(2,3,11,dtype=torch.float64)
    student=torch.randn(2,3,11,dtype=torch.float64,requires_grad=True)
    logp=teacher.log_softmax(-1);logq=student.log_softmax(-1)
    expected=(logp.exp()*(logp-logq)).sum(-1).mean()
    actual=stable_teacher_kl(student,teacher)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
    assert torch.autograd.gradcheck(lambda x:stable_teacher_kl(x,teacher),(student,),atol=1e-6,rtol=1e-5)
