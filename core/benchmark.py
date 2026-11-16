"""Known-truth calibration of joint functional composition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from .joint_decomposition import BITS, COUNT, PLAYERS, decompose, influence_operator, rake_many, independent

def _run(output_dir):
    OUT = Path(output_dir)/'stats/mosaic'
    OUT.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(2026092309)
    p0=softmax(BITS@np.array([-1.8,-2.2,-1.7,-2.1])+.7*BITS[:,0]*BITS[:,2]+.4*BITS[:,1]*BITS[:,3])
    p1=softmax(np.log(p0)+BITS@np.array([.7,.3,.8,.2])+.5*BITS[:,0]*BITS[:,1]*BITS[:,2])
    derivative,cond=influence_operator(p0,p1);errors=[]
    for arm in range(2):
        for k in range(1,16):
            direction=np.eye(16)[k]-np.eye(16)[0];eps=1e-6
            plus=[p0.copy(),p1.copy()];minus=[p0.copy(),p1.copy()]
            plus[arm]+=eps*direction;minus[arm]-=eps*direction
            fd=(decompose(*plus)['phi']-decompose(*minus)['phi'])/(2*eps)
            errors.append(np.max(np.abs(fd-derivative[:,:,arm,:]@direction)))
    base=decompose(p0,p1);reverse=decompose(p1,p0)
    perm=np.array([2,0,3,1]);perm_index=(BITS[:,perm]*2**np.arange(4)).sum(1).astype(int)
    q0=np.zeros(16);q1=np.zeros(16);q0[perm_index]=p0;q1[perm_index]=p1
    reordered=decompose(q0,q1)['phi']
    tests=dict(derivative_max_error=float(max(errors)),reverse_error=float(np.max(abs(base['phi']+reverse['phi']))),
        additivity_error=float(np.max(abs(base['phi'].sum(0)-(base['value'][-1]-base['value'][0])))),
        permutation_error=float(np.max(abs(reordered[:4]-base['phi'][perm]))),
        identity_error=float(np.max(abs(decompose(p0,p0)['phi']))),max_covariance_condition=cond)
    # Three deliberately different generating processes, each with known joint truth.
    scenarios={'Marginal change':softmax(np.log(p0)+BITS@np.array([.7,.3,.8,.2])),
        'Association change':rake_many(softmax(np.log(p0)+1.4*BITS[:,0]*BITS[:,2]+.8*BITS[:,1]*BITS[:,3])[None,:],(p0@BITS)[None,:])[0][0],
        'Combined change':p1}
    rows=[];components=[]
    for name,truth1 in scenarios.items():
        truth=decompose(p0,truth1)
        target=truth['value'][-1,1]-truth['value'][0,1]
        for rep in range(200):
            r0=rng.multinomial(1200,p0)/1200;r1=rng.multinomial(400,truth1)/400
            fit=decompose(r0,r1);grad,cc=influence_operator(r0,r1)
            cov0=(np.diag(r0)-np.outer(r0,r0))/1200;cov1=(np.diag(r1)-np.outer(r1,r1))/400
            for j,player in enumerate(PLAYERS):
                g0=grad[j,1,0];g1=grad[j,1,1];se=np.sqrt(max(g0@cov0@g0+g1@cov1@g1,0))
                val=fit['phi'][j,1];true=truth['phi'][j,1]
                components.append(dict(scenario=name,replicate=rep,component=player,estimate=val,truth=true,se=se,
                    covered=bool(abs(val-true)<=1.959964*se)))
            for method,estimate in [('MOSAIC joint',fit['value'][-1,1]-fit['value'][0,1]),
                ('Independent tasks',independent(fit['p1'])[COUNT>=2].sum()-independent(fit['p0'])[COUNT>=2].sum())]:
                rows.append(dict(scenario=name,replicate=rep,method=method,truth=target,estimate=estimate,error=estimate-target))
    d=pd.DataFrame(rows);d.to_csv(OUT/'mosaic_benchmark_runs.csv',index=False)
    comp=pd.DataFrame(components);comp.to_csv(OUT/'mosaic_benchmark_components.csv',index=False)
    summary=[]
    for (scenario,method),g in d.groupby(['scenario','method']):
        summary.append(dict(scenario=scenario,method=method,replicates=len(g),bias=g.error.mean(),rmse=np.sqrt(np.mean(g.error**2)),
            bias_mcse=g.error.std(ddof=1)/np.sqrt(len(g))))
    pd.DataFrame(summary).to_csv(OUT/'mosaic_benchmark_summary.csv',index=False)
    cs=[]
    for (scenario,component),g in comp.groupby(['scenario','component']):
        err=g.estimate-g.truth;cs.append(dict(scenario=scenario,component=component,bias=err.mean(),rmse=np.sqrt(np.mean(err**2)),coverage=g.covered.mean(),
            mcse_coverage=np.sqrt(g.covered.mean()*(1-g.covered.mean())/len(g))))
    pd.DataFrame(cs).to_csv(OUT/'mosaic_component_calibration.csv',index=False)
    tests['status']='PASS' if max(tests[k] for k in ['derivative_max_error','reverse_error','additivity_error','permutation_error','identity_error'])<1e-6 else 'FAIL'
    tests.update(replicates=600,estimated_components=3000,comparison_fits=1200)
    (OUT/'mosaic_numerical_validation.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')

def run(output_dir: Path):
    """Run 200 samples for each of three generating mechanisms."""
    with threadpool_limits(limits=1):
        _run(output_dir)
    return Path(output_dir)/'stats/mosaic'
