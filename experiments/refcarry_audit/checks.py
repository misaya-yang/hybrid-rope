"""CPU-only operator counterexamples for the user-provided RefCarry proposal.
These checks establish algebraic facts, not neural performance or novelty.
"""
import json
from pathlib import Path
import numpy as np

def rot(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

def softmax(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def run():
    rng=np.random.default_rng(20260909);maxerr=0.
    for _ in range(100):
        mu=rng.dirichlet(np.ones(9));positions=rng.integers(0,100,size=9);omega=.137
        q=rng.normal(size=2);k=rng.normal(size=2);j=117
        moment=sum(w*rot(omega*a) for w,a in zip(mu,positions))
        direct=sum(w*(q@rot(omega*(j-a))@k) for w,a in zip(mu,positions))
        factored=(moment@q)@rot(omega*j)@k
        maxerr=max(maxerr,abs(direct-factored))
    assert maxerr<1e-12
    logits=np.array([[10.,-10.,0.],[-10.,10.,0.]])
    average_attention=softmax(logits).mean(0)
    attention_of_average=softmax(logits.mean(0))
    assert abs(attention_of_average[2]-1/3)<1e-15
    assert average_attention[2]<.00005
    # Same moments do not determine the mixture of actual reader distributions.
    exact_rotations=np.array([[[1.,0.],[0.,1.]],[[0.,-1.],[1.,0.]],
                              [[-1.,0.],[0.,-1.]],[[0.,1.],[-1.,0.]]])
    mus=[np.array([.5,0.,.5,0.]),np.array([0.,.5,0.,.5])]
    qs=np.einsum('aij,j->ai',exact_rotations,np.array([10.,0.]))
    memory=np.array([[1.,0.],[-1.,0.],[0.,0.]])
    distributions=softmax(qs@memory.T)
    moments=[np.einsum('a,aij->ij',mu,exact_rotations) for mu in mus]
    mixtures=[mu@distributions for mu in mus]
    assert np.array_equal(moments[0],moments[1])
    assert np.max(np.abs(mixtures[0]-mixtures[1]))>.3
    # A TAPE-style weighted positional matrix, with fixed key fields, realizes
    # the same score. This is an operator restriction, not a claim that default
    # TAPE per-group attention maps equal an arbitrary supplied writer map.
    mu=rng.dirichlet(np.ones(7));addresses=np.arange(7);omega=.3
    e_query=sum(w*rot(omega*a).T for w,a in zip(mu,addresses));e_key=rot(omega*13).T
    q=rng.normal(size=2);k=rng.normal(size=2)
    tape_style=q@e_query@e_key.T@k
    refcarry=(e_query.T@q)@rot(omega*13)@k
    tape_error=abs(tape_style-refcarry);assert tape_error<1e-14
    # A normalized measure on two addresses needs only its first mass.
    features=np.stack([np.array([1.,0.]),np.array([0.,1.])],axis=1)
    t=.37;mu=np.array([t,1-t]);sketch=np.array([[1.,0.]])@mu
    reconstructed=np.array([sketch[0],1-sketch[0]])
    assert np.array_equal(features@mu,reconstructed)
    assert np.linalg.matrix_rank(features)==2 and sketch.size==1
    # Convex interpolation of two opposite unit rotations can erase a query.
    gated=.5*rot(0.)+.5*rot(np.pi)
    normratio=float(np.linalg.norm([REDACTED_EMAIL]([1.,0.])))
    assert normratio<1e-15
    # Same unrotated learned query, correct origin replacement can double-count
    # a reference already computed by native hidden states.
    omega=.2;i=10;a=6;offset=-1;u=np.array([1.,0.])
    q=rot(omega*(a+offset-i))@u
    positions=np.arange(11);keys=np.array([rot(omega*j)@u for j in positions])
    native=keys@(rot(omega*i)@q);rebased=keys@(rot(omega*a)@q)
    assert int(native.argmax())==a+offset==5
    assert int(rebased.argmax())==2*a+offset-i==1
    # Arbitrary content keys do not yield a translation of the score profile.
    omega=.1;positions=np.arange(20);content_scale=np.ones(20);content_scale[7]=10
    prekeys=np.array([c*(rot(-omega*j)@u) for c,j in zip(content_scale,positions)])
    postkeys=np.array([rot(omega*j)@k for j,k in zip(positions,prekeys)])
    selected=[int((postkeys@(rot(omega*a)@u)).argmax()) for a in range(4)]
    assert selected==[7]*4
    # Standard nonlinear conditioning implements a binary address switch exactly
    # on a bounded query domain, although one affine map cannot do so.
    def gated_copy(x,z):
        return np.maximum(x+(z-1),0)-np.maximum(-x+(z-1),0)
    e=0.
    for z in (0.,1.):
        for q in rng.uniform(-1,1,size=(100,2)):
            # Gate each q coordinate first, keeping its guaranteed [-1,1] bound.
            h=gated_copy(q,z);out=q+(rot(.7)-np.eye(2))@h
            e=max(e,float(np.max(np.abs(out-((rot(.7) if z else np.eye(2))@q)))))
    assert e<1e-14
    return {
      'scope':'CPU algebra and counterexamples; no GPU, checkpoint or task result',
      'group_logit_factorization_max_abs_error':maxerr,
      'softmax_counterexample':{'mean_of_attention':average_attention.tolist(),'attention_of_mean_logits':attention_of_average.tolist(),'total_variation':float(.5*np.abs(average_attention-attention_of_average).sum())},
      'same_moments_different_reader_mixtures':{'address_probabilities':[x.tolist() for x in mus],'moments_equal':True,'reader_mixtures':[x.tolist() for x in mixtures]},
      'fixed_key_TAPE_style_operator_identity_error':float(tape_error),
      'simplex_rank_counterexample':{'rank_Phi':2,'sufficient_linear_sketch_dimension':1,'reconstructed_moment':reconstructed.tolist(),'correct_general_dimension':'rank([ones_row; Phi]) - 1, with known unit mass and an affine decoder'},
      'zero_entropy_gate_cancellation':{'reference_entropy':0.,'g':.5,'query_norm_ratio':normratio},
      'native_query_rebasing_counterexample':{'current_position':i,'gold_anchor':a,'desired_offset':offset,'native_argmax':int(native.argmax()),'gold_reference_argmax':int(rebased.argmax()),'interpretation':'Oracle rebasing failure need not mean reference information was absent'},
      'shift_reference_counterexample':{'reference_positions':[0,1,2,3],'selected_positions':selected,'slope':0.,'interpretation':'Unit positional slope requires controlled/translated key content; group law alone is insufficient'},
      'nonlinear_residual_address_switch':{'bounded_domain':'q in [-1,1]^2, two addresses encoded by z in {0,1}','max_abs_error':e,'interpretation':'One affine-map impossibility is not an impossibility for ordinary MLP blocks'}
    }
if __name__=='__main__':
    result=run();p=Path(__file__).with_name('results.json');p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
