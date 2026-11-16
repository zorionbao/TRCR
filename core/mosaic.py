"""Cross-fitted joint functional distributions and MOSAIC composition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import clinical as c
from .joint_decomposition import BITS, COUNT, PLAYERS, simplex, decompose, independent, information

def fit(cohort, d, output_dir, *, scores_path=None, score_frame=None, design_fn=None, output_prefix='mosaic_bootstrap'):
    output_dir = Path(output_dir)
    HARM = output_dir/'harmonized'
    OUT = output_dir/'stats/mosaic'
    OUT.mkdir(parents=True, exist_ok=True)
    d = d.loc[d.cohort.eq(cohort) & d.obs1].reset_index(drop=True)
    assert d.living_interview.eq(True).all() and d.n1_living_interview.eq(True).all()
    assert d.dt1.between(1.5, 3.5).all() and d.n1_wave.sub(d.wave).eq(1).all()
    a = d.a.to_numpy(int); ids = d.id.astype(str).to_numpy()
    state = (d[['n1_'+x for x in c.CORE]].to_numpy(int) * 2**np.arange(4)).sum(1)
    y = np.eye(16)[state]
    s = score_frame.copy() if score_frame is not None else pd.read_csv(scores_path or HARM/f'clinical_scores_{cohort.lower()}.csv', dtype={'id': str})
    keys = ['cohort','id','wave','a']
    check = d[keys].merge(s, on=keys, how='left', sort=False, validate='one_to_one')
    assert len(check) == len(d) and check[keys].equals(d[keys])
    assert check[['propensity','fold']].notna().all().all()
    assert check.groupby('id', sort=False).fold.nunique().eq(1).all()
    assert check.y.eq(d.n1_core4.ge(2).astype(float) if check.endpoint.eq('multiple4').all()
                      else d.n1_core4.ge(1).astype(float)).all()
    e = check.propensity.to_numpy(); folds = check.fold.to_numpy()
    X = (design_fn or c.design)(d); pred = np.zeros_like(y)
    for f in range(5):
        train = (folds != f) & (a == 0); test = folds == f
        model = c.pipeline(); model.fit(X.loc[train], state[train])
        pr = model.predict_proba(X.loc[test]); classes = model[-1].classes_.astype(int)
        pred[np.ix_(test, classes)] = pr
    assert np.allclose(pred.sum(1), 1)
    w = (1-a)*e/(1-e); n1 = a.sum(); w0 = w.sum()
    p1 = y[a == 1].mean(0)
    meanm = pred[a == 1].mean(0)
    aug = (w[:, None] * (y-pred)).sum(0)/w0
    raw0 = meanm + aug; p0 = simplex(raw0)
    if1 = a[:, None]*(y-p1)/n1
    if0 = a[:, None]*(pred-meanm)/n1 + w[:, None]*(y-pred-aug)/w0
    cluster0 = pd.DataFrame(if0).groupby(ids, sort=True).sum().to_numpy()
    cluster1 = pd.DataFrame(if1).groupby(ids, sort=True).sum().to_numpy()
    rng = np.random.default_rng(c.SEEDS[cohort] + 800)
    base = decompose(p0, p1)
    reps = []; prob0=[]; prob1=[]; information_rows=[]
    for b in range(1000):
        z = rng.choice([-1., 1.], len(cluster0))
        # Perturb the augmented estimate before projecting each multiplier draw.
        q0 = simplex(raw0 + z@cluster0); q1 = simplex(p1 + z@cluster1)
        r = decompose(q0, q1)
        reps.append(r['phi']); prob0.append(q0); prob1.append(q1)
        information_rows.append([information(r['p0']), information(r['p1'])])
    reps = np.asarray(reps); prob0 = np.asarray(prob0); prob1 = np.asarray(prob1)
    np.savez_compressed(OUT/f'{output_prefix}_{cohort.lower()}.npz', phi=reps, p0=prob0, p1=prob1)
    rows=[]
    for k in range(16):
        effect = p1[k]-raw0[k]
        se = np.sqrt(np.square(cluster1[:, k]-cluster0[:, k]).sum()*len(cluster0)/(len(cluster0)-1))
        rows.append(dict(cohort=cohort,state=k,pattern=''.join(str(int(v)) for v in BITS[k]),
            count=int(COUNT[k]),reference=raw0[k],reference_projected=p0[k],fracture=p1[k],difference=effect,
            n=len(d),cases=int(n1),
            case_count=int(y[a==1,k].sum()),control_count=int(y[a==0,k].sum())))
    components=[]
    for j, name in enumerate(PLAYERS):
        for k, outcome in enumerate(['any','multiple','three_plus','count']):
            lo, hi = np.quantile(reps[:,j,k], [.025,.975])
            components.append(dict(cohort=cohort,component=name,outcome=outcome,value=base['phi'][j,k],lo=lo,hi=hi))
    values=[]
    for s in range(32):
        for k, outcome in enumerate(['any','multiple','three_plus','count']):
            values.append(dict(cohort=cohort,coalition=s,players=int(s).bit_count(),outcome=outcome,value=base['value'][s,k]))
    summaries=[]
    for label, pp in [('reference',base['p0']),('fracture',base['p1'])]:
        summaries.append(dict(cohort=cohort,arm=label,multi_risk=pp[COUNT>=2].sum(),
            independent_multi=independent(pp)[COUNT>=2].sum(),total_correlation_bits=information(pp),
            mean_count=pp@COUNT,entropy_bits=-np.sum(pp*np.log2(pp)),projection_l1=np.abs(p0-raw0).sum()))
    transitions=[]
    for arm in [0,1]:
        for t0,t1,tag in [('p1_','n1_','Pre-interval to follow-up'),('','n1_','Reporting to follow-up')]:
            g=d.loc[d.a.eq(arm)]
            x=(g[[t0+x for x in c.CORE]].to_numpy(int)*2**np.arange(4)).sum(1)
            z=(g[[t1+x for x in c.CORE]].to_numpy(int)*2**np.arange(4)).sum(1)
            table=np.zeros((16,16),int);np.add.at(table,(x,z),1)
            for i in range(16):
                for j in range(16):transitions.append(dict(cohort=cohort,arm=arm,window=tag,from_state=i,to_state=j,n=int(table[i,j]),row_n=int(table[i].sum()),fraction=table[i,j]/max(1,table[i].sum())))
    meta=dict(cohort=cohort,n=len(d),cases=int(n1),persons=len(cluster0),
        raw_reference_min=float(raw0.min()),simplex_projection_l1=float(np.abs(p0-raw0).sum()),
        sum_check=float(raw0.sum()),bootstrap=1000,ipf_error=float(base['marginal_error']),
        multiplier_center='unprojected augmented reference estimate',
        interval_scope='conditional on fitted nuisance models and empirical case support')
    return rows,components,values,summaries,transitions,meta


def run(output_dir: Path):
    """Estimate joint distributions from cross-fitted clinical landmarks."""
    output_dir = Path(output_dir)
    outdir = output_dir/'stats/mosaic'
    outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(output_dir/'harmonized/clinical_landmarks.csv', dtype={'id': str}, low_memory=False)
    with threadpool_limits(limits=1):
        results = [fit(cohort, frame, output_dir) for cohort in c.SEEDS]
    names = ['joint_states','mosaic_components','mosaic_coalitions','joint_information','joint_transitions']
    for j, name in enumerate(names):
        pd.DataFrame([row for result in results for row in result[j]]).to_csv(outdir/(name+'.csv'), index=False)
    (outdir/'mosaic_fit.json').write_text(json.dumps([result[-1] for result in results], indent=2), encoding='utf-8')
    return outdir
