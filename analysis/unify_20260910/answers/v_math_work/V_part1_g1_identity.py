# -*- coding: utf-8 -*-
"""V_math part1: independent recomputation of G1 identity Σ_excess(g23..39)=ln4·(m40-m23)
across ALL 38 tables; exceptions; P2 single-gap reconciliation; Stack hole position;
closed-form reconstructions ramp17/16/15 vs G1 arrays."""
import json, math
ROOT='/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope'
d=json.load(open(f'{ROOT}/analysis/unify_20260910/tables/ground_truth_tables.json'))
M=d['methods']
c=math.log(1e6)/64; ln4=math.log(4); rnat=1e6**(1/64)
print('constants: ln4=%.16f  c=%.16f  rho_nat=%.16f'%(ln4,c,rnat))

def excess_sum_nu(nu,g0=23,g1=39):
    return sum(math.log(nu[g]/nu[g+1])-c for g in range(g0,g1+1))
def excess_sum_m(m,g0=23,g1=39):
    return sum((m[g+1]-m[g])*ln4 for g in range(g0,g1+1))

print('\n== Σ_excess(g23..39) per method (from nu_j fp32 arrays AND from m_j arrays) ==')
print(f"{'method':30s} {'Σ_from_nu':>14s} {'Σ_from_m':>14s} {'m40-m23':>10s} {'stored':>14s} {'|Σ−ln4|':>10s}")
exceptions=[]
n_none=[]
for name,e in M.items():
    nu=e['nu_j']; m=e['m_j']
    if nu is None or m is None:
        n_none.append(name); continue
    s_nu=excess_sum_nu(nu); s_m=excess_sum_m(m)
    stored=e.get('sum_transition_gap_increments_gaps23_39')
    dm=e['endpoint_delta_m']; span=dm['m_40']-dm['m_23']
    flag='' if abs(s_nu-ln4)<1e-6 else ' *EXCEPTION*'
    if flag: exceptions.append((name,s_nu,s_m,span,stored))
    print(f"{name:30s} {s_nu:14.8f} {s_m:14.8f} {span:10.6f} {stored:14.8f} {abs(s_nu-ln4):10.2e}{flag}")
    # identity check: s_m should equal span*ln4 exactly (m-based)
    assert abs(s_m-span*ln4)<1e-12, (name,s_m,span*ln4)

print('\nexceptions (|Σ−ln4|>=1e-6):')
for x in exceptions: print('  ',x[0], 'Σ_nu=%.6f  span*m=%.6f  ln4·span=%.6f'%(x[1],x[2],x[3]*ln4))

print('\n== P2 (FullLagP2_Transfer3B) reconciliation ==')
e=M['FullLagP2_Transfer3B']; m=e['m_j']; nu=e['nu_j']
gaps_raw=[math.log(nu[g]/nu[g+1]) for g in range(63)]
exc=[g-c for g in gaps_raw]
order=sorted(range(23,40),key=lambda g:-exc[g])[:5]
for g in order:
    rho=math.exp(gaps_raw[g-0]*0)*1  # placeholder
rho_from_T=[e['T_j'][g+1]/e['T_j'][g] for g in range(63)]
for g in order:
    print(f'  gap g{g}: raw ln(νg/νg+1)={gaps_raw[g]:.6f}  excess={exc[g]:.6f} nats  ρ={rho_from_T[g]:.6f}  lnρ={math.log(rho_from_T[g]):.6f}')
print('  Σ_excess(23..39)=%.8f  (ln4=%.8f, diff=%.2e)  m40-m23=%.6f  m23=%.6f'%(excess_sum_nu(nu),ln4,excess_sum_nu(nu)-ln4,e['endpoint_delta_m']['m_40']-e['endpoint_delta_m']['m_23'],e['endpoint_delta_m']['m_23']))
biggest=math.log(rho_from_T[29])
print('  P2 max ρ = %.6f @g29? lnρ=%.6f  -> "1.088" is the RAW log-gap ln ρ29, its BUDGET (excess) consumption = %.6f nats = %.2f%% of ln4'%(rho_from_T[29],biggest,biggest-c,(biggest-c)/ln4*100))
print('  remaining 16 gaps supply %.6f nats; Σ_raw(23..39)=%.6f (NOT ln4!)'%(excess_sum_nu(nu)-(biggest-c),sum(gaps_raw[23:40])))
# P2 m array shape
print('  P2 m_24..m_32:',[round(m[j],5) for j in range(24,33)])
# check ρ29 = e^1.0885 = 2.9696 claim
print('  claim e^{1.0885}=2.9696 -> actual %.6f (ln=%.6f)'%(math.exp(1.0885),math.log(2.9696)))
# three P2 big holes claim 1.5215/2.9696/1.522
holes=[(g,rho_from_T[g]) for g in range(23,40) if rho_from_T[g]>1.5]
print('  holes>1.5:',[(g,round(r,4)) for g,r in holes])

print('\n== E2 / HighGapToLong / NTK violations ==')
for nm in ['E2_tail_more','HighGapToLong','NTK_static','HighGapToMid']:
    e=M[nm]; m=e['m_j']; dm=e['endpoint_delta_m']
    s=excess_sum_m(m); span=dm['m_40']-dm['m_23']
    print(f'  {nm:18s} Σ_m={s:.6f}  ln4+lnb/64={ln4+c:.6f}  span(m40-m23)={span:.6f}  m23={dm["m_23"]:.6f} m40={dm["m_40"]:.6f}  s/ln4={s/ln4:.4f}')

print('\n== Stack hole position (D1@35 vs T2/T3@38) ==')
st=M['StackFrontBack']; lbs=M['LongBridgeSlower']
ms=st['m_j']; ml=lbs['m_j']
rhoS=[st['T_j'][g+1]/st['T_j'][g] for g in range(63)]
argmax=max(range(63),key=lambda g:rhoS[g])
print('  Stack ρ35=%.4f ρ37=%.4f ρ38=%.4f ρ39=%.4f ; global argmax=g%d val=%.4f'%(rhoS[35],rhoS[37],rhoS[38],rhoS[39],argmax,rhoS[argmax]))
print('  Stack stored fields: hole_max_global=%.4f argmax=%d'%(st['hole_ratio_max_global'],st['hole_ratio_argmax_gap']))
print('  m36..39 Stack:',[round(ms[j],4) for j in range(36,40)],' LBS:',[round(ml[j],4) for j in range(36,40)])
print('  Stack m28=%.6f ρ28=%.4f ρ27=%.4f'%(ms[28],rhoS[28],rhoS[27]))
lbsrho=[lbs['T_j'][g+1]/lbs['T_j'][g] for g in range(63)]
am2=max(range(63),key=lambda g:lbsrho[g])
print('  LBS argmax=g%d val=%.4f  ρ28(LBS)=%.4f'%(am2,lbsrho[am2],lbsrho[28]))

print('\n== closed-form reconstructions ==')
def ramp(Np):
    return [min(1.0,max(0.0,(q*(q+1))/(Np*(Np+1)))) for q in [max(0,min(Np,j-23)) for j in range(64)]]
def uni(Np):
    return [min(1.0,max(0.0,(j-23)/Np)) for j in range(64)]
for Np,nm in [(17,'MrPro'),(16,'MrProN16'),(15,'MrProN15')]:
    a=ramp(Np); b=M[nm]['m_j']
    print(f'  ramp{Np} vs {nm}: max|Δm|={max(abs(x-y) for x,y in zip(a,b)):.3e}  Σ_excess(exact)={excess_sum_m(a):.10f}')
a=uni(17); b=M['MrUni']['m_j']
print('  uni17 vs MrUni: max|Δm|=%.3e'%(max(abs(x-y) for x,y in zip(a,b))))

print('\n== D4 §2 BL claim: uniform N=14 BL=Σ_24^28 m_j doc says 1.000 ==')
u14=uni(14); u17=uni(17); mp=ramp(17)
print('  BL(uni14)=%.6f  BL(uni17)=%.6f(=.8824?)  BL(MrPro)=%.6f (D4 table .2288)'%(
 sum(u14[j] for j in range(24,29)),sum(u17[j] for j in range(24,29)),sum(mp[j] for j in range(24,29))))

print('\n== gain & misc constants ==')
print('  1+0.1*ln4=%.15f (doc 1.138629436111989)'%(1+0.1*ln4))
print('  lnS/lnb=%.6f (T3 0.100343)'%(ln4/math.log(1e6)))
print('  17c+ln4=%.6f (T3 5.056039)'%(17*c+ln4))
