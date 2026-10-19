"""Simple entry point: clean, evaluate saved models, train, or rebuild the HTML report."""
from pathlib import Path
import argparse, hashlib, html, json, platform, sys, time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from Cleaned_Dataset.clean_data import (ROOT, WAPS, BUILDINGS, FLOORS, SignalEncoder,
                                       read_data, clean_file, clean_all)
from PCA_KNN.model import knn_pipeline, neighbour_labels, predict_all_k
from Random_Forest.model import rf_pipeline

ALGOS = ('PCA_KNN', 'Random_Forest')
ARCHS = ('building_wise', 'combined')
SEEDS = (42, 43, 44, 45, 46)
STAGES = (25, 50, 100, 150, 200, 250)
EXPECTED = {'numpy':'2.5.4', 'pandas':'2.3.3', 'sklearn':'1.9.1', 'joblib':'1.6.0'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def metadata(algo):
    return json.loads((ROOT/algo/'Saved_Models/metadata.json').read_text())


def invalidate_comparison():
    for name in ('Comparison.html', 'Comparison.csv'):
        (ROOT/'Results'/name).unlink(missing_ok=True)


def check_runtime(meta):
    actual = versions()
    for k, v in meta['versions'].items():
        if k != 'python' and actual[k] != v:
            raise ValueError(f'Incompatible {k}: {actual[k]}; install requirements.txt ({v} required).')
    if sys.version_info[:2] != (3, 13):
        raise ValueError('Use Python 3.13; the supplied models were recorded with Python 3.13.14.')


def model_files(arch):
    return ['combined.joblib'] if arch == 'combined' else [f'building_{b}.joblib' for b in BUILDINGS]


def load_models(algo, arch):
    meta = metadata(algo)
    if meta['status'] != 'complete': raise ValueError(f'{algo} training is incomplete.')
    check_runtime(meta)
    models = {}
    for filename in model_files(arch):
        path = ROOT/algo/'Saved_Models'/filename
        if sha256(path) != meta['model_sha256'][filename]:
            raise ValueError(f'Model integrity mismatch: {path.name}')
        key = 'All' if filename == 'combined.joblib' else filename.split('_')[1].split('.')[0]
        # joblib is pickle-based: load only the supplied/trusted project artifacts.
        models[key] = joblib.load(path)
    return models


def predict(models, frame):
    if set(models) == {'All'}: return models['All'].predict(frame[WAPS])
    out = np.full(len(frame), -1, dtype=int)
    for b in BUILDINGS:
        rows = np.flatnonzero(frame.BUILDINGID.to_numpy() == b)
        if len(rows): out[rows] = models[str(b)].predict(frame.iloc[rows][WAPS])
    if (out < 0).any(): raise ValueError('Unrouted building.')
    return out


def metrics(y, pred):
    correct = int((np.asarray(y) == pred).sum()); n = len(y)
    return dict(rows=n, correct=correct, errors=n-correct,
                accuracy=correct/n, error=(n-correct)/n)


def evaluate(algo, frame, source):
    out = ROOT/'Results'/algo; out.mkdir(parents=True, exist_ok=True)
    # Invalidated before prediction so a failure cannot leave a stale report.
    for f in ('Validation_Results.csv','Validation_Predictions.csv','Confusion_Matrices.csv'):
        (out/f).unlink(missing_ok=True)
    info = metadata(algo); records=[]; predictions=[]; matrices=[]
    with threadpool_limits(limits=2):
        for arch in ARCHS:
            pred = predict(load_models(algo, arch), frame)
            for scope, idx in [('overall', np.arange(len(frame)))]+[
                    (f'building_{b}', np.flatnonzero(frame.BUILDINGID.to_numpy()==b)) for b in BUILDINGS]:
                if not len(idx): continue
                records.append(dict(architecture=arch, scope=scope,
                    **metrics(frame.FLOOR.to_numpy()[idx], pred[idx]),
                    validation_sha256=sha256(source), training_sha256=info['training_sha256'],
                    model_sha256=';'.join(info['model_sha256'][f] for f in model_files(arch))))
            predictions.append(pd.DataFrame(dict(architecture=arch, row_id=np.arange(len(frame)),
                BUILDINGID=frame.BUILDINGID.to_numpy(), true_floor=frame.FLOOR.to_numpy(), predicted_floor=pred)))
            matrix=confusion_matrix(frame.FLOOR, pred, labels=FLOORS)
            matrices.extend(dict(architecture=arch, true_floor=i, predicted_floor=j, count=int(matrix[i,j]))
                            for i in FLOORS for j in FLOORS)
            score = metrics(frame.FLOOR, pred)
            print(f'{algo} / {arch}: ERROR {100*score["error"]:.2f}% ({score["errors"]}/{score["rows"]}); '
                  f'accuracy {100*score["accuracy"]:.2f}%', flush=True)
    pd.DataFrame(records).to_csv(out/'Validation_Results.csv',index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'Validation_Predictions.csv',index=False)
    pd.DataFrame(matrices).to_csv(out/'Confusion_Matrices.csv',index=False)


def choose_k(scores):
    means = np.asarray(scores).mean(axis=0); history=[]
    for upper in sorted(set(min(s,len(means)) for s in STAGES)):
        best_score=means[:upper].max()
        k=int(np.flatnonzero(means[:upper]>=best_score-1e-12)[0]+1)
        stop=k <= upper-5 and means[upper-5:upper].max() < best_score-1e-12
        history.append(dict(upper=upper, best_k=k, stopping_rule_met=bool(stop)))
        if stop: break
    return k, upper, history


def make_folds(frame):
    strata = frame.BUILDINGID.astype(str)+':'+frame.FLOOR.astype(str)
    if strata.value_counts().min()<10: raise ValueError('Each building-floor stratum needs >=10 rows.')
    folds=[]
    for seed in SEEDS:
        ids=np.full(len(frame),-1,dtype=int)
        for f,(train,test) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(frame,strata),1):
            for b in BUILDINGS:
                expected=set(frame.loc[frame.BUILDINGID==b,'FLOOR'])
                assert set(frame.iloc[train].query('BUILDINGID == @b').FLOOR)==expected
                assert set(frame.iloc[test].query('BUILDINGID == @b').FLOOR)==expected
            ids[test]=f
        assert (ids>0).all(); folds.append(ids)
    return folds


def train(algo, overwrite=False):
    """Full 5x10-fold protocol and final refit. Never reads validation labels/data."""
    saved=ROOT/algo/'Saved_Models'; saved.mkdir(parents=True,exist_ok=True)
    if any(saved.glob('*.joblib')) and not overwrite:
        raise FileExistsError('Prebuilt models exist. To deliberately retrain use train --overwrite.')
    for k,v in EXPECTED.items():
        if versions()[k]!=v: raise ValueError('Install pinned requirements before training.')
    invalidate_comparison()
    result_dir=ROOT/'Results'/algo; result_dir.mkdir(parents=True,exist_ok=True)
    for file in result_dir.glob('*.csv'): file.unlink()
    meta=dict(status='training', versions=versions(), training_sha256=sha256(ROOT/'Dataset/trainingData.csv'),
              folds=10, seeds=list(SEEDS), grouping=False, pca_components=200,
              k_stages=list(STAGES), partitions={},model_sha256={})
    save_json(saved/'metadata.json',meta)
    frame=clean_file(ROOT/'Dataset/trainingData.csv', ROOT/'Cleaned_Dataset/trainingData_cleaned.csv',training=True)
    folds=make_folds(frame); X=frame[WAPS].to_numpy(); y=frame.FLOOR.to_numpy()
    all_scores=[]; all_oof=[]; search=[]
    with threadpool_limits(limits=2):
        for arch in ARCHS:
            oof=np.full((5,len(frame)),-1,dtype=np.int8); part_info={}
            parts=[('All',np.arange(len(frame)))] if arch=='combined' else [
                (str(b),np.flatnonzero(frame.BUILDINGID.to_numpy()==b)) for b in BUILDINGS]
            for part,idx in parts:
                mintrain=min(int((folds[r][idx]!=f).sum()) for r in range(5) for f in range(1,11))
                if algo=='PCA_KNN' and mintrain<200: raise ValueError('PCA requires >=200 training rows per fold.')
                ceiling=min(250,mintrain); scores=[]; heldout=[]
                for r,fold_ids in enumerate(folds):
                    for f in range(1,11):
                        tr=idx[fold_ids[idx]!=f]; te=idx[fold_ids[idx]==f]
                        if algo=='PCA_KNN':
                            pca=PCA(n_components=200,svd_solver='full',whiten=False)
                            ztr=pca.fit_transform(X[tr]); zte=pca.transform(X[te])
                            pred=predict_all_k(neighbour_labels(ztr,y[tr],zte,ceiling))
                        else:
                            fitted=rf_pipeline().fit(frame.iloc[tr][WAPS],y[tr])
                            pred=fitted.predict(frame.iloc[te][WAPS]).astype(np.int8)[:,None]
                        scores.append((pred==y[te,None]).mean(axis=0)); heldout.append((r,te,pred))
                        if f in (5,10): print(f'{algo} / {arch} / {part}: repeat {r+1}/5 fold {f}/10',flush=True)
                if algo=='PCA_KNN':
                    k,upper,history=choose_k(scores); column=k-1
                    search.extend(dict(architecture=arch,partition=part,k=n,
                        mean_cv_accuracy=float(np.mean(scores,axis=0)[n-1]),
                        mean_cv_error=float(1-np.mean(scores,axis=0)[n-1]),selected=n==k) for n in range(1,upper+1))
                    model=knn_pipeline(k); details=dict(k=k,tested_through=upper,search_history=history)
                else:
                    column=0; model=rf_pipeline()
                    details=dict(trees=200,max_features='sqrt',max_depth=None,min_samples_leaf=1,bootstrap=True,random_state=42)
                for r,te,pred in heldout: oof[r,te]=pred[:,column]
                model.fit(frame.iloc[idx][WAPS],y[idx])
                details.update(training_rows=len(idx),mean_selection_cv_accuracy=float(np.asarray(scores)[:,column].mean()))
                if algo=='PCA_KNN':
                    details.update(pca_components=200,retained_variance=float(model.named_steps['pca'].explained_variance_ratio_.sum()))
                part_info[part]=details
                filename='combined.joblib' if part=='All' else f'building_{part}.joblib'
                joblib.dump(model,saved/filename,compress=3); meta['model_sha256'][filename]=sha256(saved/filename)
            assert (oof>=0).all(); repeats=[]; fold_records=[]
            for r,ids in enumerate(folds):
                z=metrics(y,oof[r]); repeats.append(z['accuracy'])
                all_scores.append(dict(architecture=arch,kind='repeat',repeat=r+1,fold=0,**z))
                for f in range(1,11):
                    mask=ids==f; z=metrics(y[mask],oof[r,mask])
                    rec=dict(architecture=arch,kind='fold',repeat=r+1,fold=f,**z)
                    fold_records.append(rec); all_scores.append(rec)
                all_oof.append(pd.DataFrame(dict(architecture=arch,row_id=np.arange(len(y)),repeat=r+1,
                    fold=ids,BUILDINGID=frame.BUILDINGID.to_numpy(),true_floor=y,predicted_floor=oof[r])))
            mean=float(np.mean([x['accuracy'] for x in fold_records]))
            all_scores.append(dict(architecture=arch,kind='summary',repeat=0,fold=0,rows=len(y),
                accuracy=mean,error=1-mean,mean_repeat_accuracy=float(np.mean(repeats)),repeat_sd=float(np.std(repeats,ddof=1))))
            meta['partitions'][arch]=part_info
            print(f'Finished {algo} / {arch}: mean CV error {100*(1-mean):.3f}%',flush=True)
    pd.DataFrame(all_scores).to_csv(result_dir/'CV_Results.csv',index=False)
    pd.concat(all_oof,ignore_index=True).to_csv(result_dir/'CV_Predictions.csv',index=False)
    if search: pd.DataFrame(search).to_csv(result_dir/'K_Search.csv',index=False)
    meta.update(status='complete',cv_results_sha256=sha256(result_dir/'CV_Results.csv'),
                cv_predictions_sha256=sha256(result_dir/'CV_Predictions.csv'))
    if search: meta['k_search_sha256']=sha256(result_dir/'K_Search.csv')
    save_json(saved/'metadata.json',meta)
    print('Training finished. Run evaluate to generate fresh validation results and Comparison.html.',flush=True)


def table(headers,rows):
    def tr(values,tag): return '<tr>'+''.join(f'<{tag}>'+html.escape(str(v))+f'</{tag}>' for v in values)+'</tr>'
    return '<div class="table-wrap"><table><thead>'+tr(headers,'th')+'</thead><tbody>'+''.join(tr(row,'td') for row in rows)+'</tbody></table></div>'


def label(algo,arch):
    return ('Building-wise ' if arch=='building_wise' else 'Combined ')+('PCA–kNN' if algo=='PCA_KNN' else 'Random Forest')


def report():
    """Error-first, offline HTML. Reject stale/mixed benchmark evaluations."""
    records=[]; details={}; cv_data={}; meta_data={}; bench=set(); content=''
    for algo in ALGOS:
        out=ROOT/'Results'/algo; path=out/'Validation_Results.csv'
        if not path.exists(): continue
        meta=metadata(algo)
        if meta['status']!='complete': continue
        cv_path=out/'CV_Results.csv'
        if sha256(cv_path)!=meta['cv_results_sha256']: raise ValueError(f'{algo} CV file changed.')
        val=pd.read_csv(path); cv=pd.read_csv(cv_path)
        pred=pd.read_csv(out/'Validation_Predictions.csv'); cm=pd.read_csv(out/'Confusion_Matrices.csv')
        for arch in ARCHS:
            v=val.query('architecture == @arch and scope == "overall"').iloc[0]
            binding=';'.join(meta['model_sha256'][f] for f in model_files(arch))
            if v.model_sha256!=binding: raise ValueError(f'{algo} model/results mismatch; evaluate again.')
            for f in model_files(arch):
                if sha256(ROOT/algo/'Saved_Models'/f)!=meta['model_sha256'][f]: raise ValueError('Changed saved model.')
            pv=pred.query('architecture == @arch'); check=metrics(pv.true_floor,pv.predicted_floor.to_numpy())
            for field in ('rows','correct','errors','accuracy','error'):
                if not np.isclose(check[field],v[field],rtol=0,atol=1e-12): raise ValueError('Prediction/score mismatch.')
            actualcm=confusion_matrix(pv.true_floor,pv.predicted_floor,labels=FLOORS)
            storedcm=cm.query('architecture == @arch').pivot(index='true_floor',columns='predicted_floor',values='count').reindex(index=FLOORS,columns=FLOORS).to_numpy()
            if not np.array_equal(actualcm,storedcm): raise ValueError('Confusion matrix mismatch.')
            for vsub in val.query('architecture == @arch and scope != "overall"').itertuples():
                building=int(str(vsub.scope).replace('building_',''))
                sub=pv[pv.BUILDINGID==building]
                checksub=metrics(sub.true_floor,sub.predicted_floor.to_numpy())
                for field in ('rows','correct','errors','accuracy','error'):
                    if not np.isclose(checksub[field],getattr(vsub,field),rtol=0,atol=1e-12):
                        raise ValueError('Building-score/prediction mismatch.')
            c=cv.query('architecture == @arch and kind == "summary"').iloc[0]
            bench.add(v.validation_sha256)
            records.append(dict(algorithm=algo,architecture=arch,configuration=label(algo,arch),
                validation_error=v.error,validation_errors=int(v.errors),validation_rows=int(v.rows),
                validation_accuracy=v.accuracy,correct=int(v.correct),cv_error=c.error,cv_accuracy=c.accuracy,
                repeat_sd_pp=100*c.repeat_sd,validation_sha256=v.validation_sha256))
        details[algo]=(val,pred,cm); cv_data[algo]=cv; meta_data[algo]=meta
    if not records: raise ValueError('No evaluation results. Run python run.py evaluate first.')
    if len(bench)!=1: raise ValueError('Validation files differ across algorithms. Evaluate both on the same file.')
    comparison=pd.DataFrame(records)
    n=records[0]['validation_rows']; best=min(r['validation_error'] for r in records)
    winners=[r['configuration'] for r in records if abs(r['validation_error']-best)<1e-12]
    content='<header><p class="eyebrow">WIFI FLOOR CLASSIFICATION · MODEL EVALUATION</p><h1>Results comparison</h1>'
    content+=f'<p>Classification error first. {len(records)} configurations evaluated on the same validation observations.</p></header>'
    if len(records)!=4: content+='<p class="notice">Partial comparison: only evaluated configurations are shown. Run python run.py evaluate for all four.</p>'
    content+=f'<section><h2>1. Validation classification error</h2><p><strong>Lower is better.</strong> Error = incorrect floor predictions ÷ evaluated observations. Accuracy = 1 − error.</p><div class="cards">'
    for r in records:
        content+=f'<article class="card"><h3>{html.escape(r["configuration"])}</h3><div class="big">{100*r["validation_error"]:.2f}%</div><p>classification error<br/>{r["validation_errors"]:,} incorrect / {r["validation_rows"]:,} scans</p></article>'
    content+='</div><p class="finding"><strong>Lowest observed error:</strong> '+html.escape(' and '.join(winners))+f' ({100*best:.2f}%). '
    content+='A tie in overall error does not mean identical predictions.</p>'
    content+=table(['Configuration','Error ↓','Incorrect','Accuracy ↑','Correct','Validation rows'],[
        [r['configuration'],f'{100*r["validation_error"]:.2f}%',r['validation_errors'],f'{100*r["validation_accuracy"]:.2f}%',r['correct'],r['validation_rows']] for r in records])+'</section>'
    content+='<section><h2>2. Cross-validation versus validation</h2><p>Training only: five repetitions of stratified ten-fold CV, seeds 42–46, matched building–floor splits, no location grouping. CV values are the mean of 50 fold scores.</p>'
    content+=table(['Configuration','CV error ↓','Validation error ↓','Gap (pp)','CV accuracy','Validation accuracy','Repeat SD (pp)'],[
        [r['configuration'],f'{100*r["cv_error"]:.3f}%',f'{100*r["validation_error"]:.2f}%',f'{100*(r["validation_error"]-r["cv_error"]):.2f}',f'{100*r["cv_accuracy"]:.2f}%',f'{100*r["validation_accuracy"]:.2f}%',f'{r["repeat_sd_pp"]:.3f}'] for r in records])
    content+='<p>SD is the sample standard deviation of five pooled repetition accuracies, not a confidence interval. CV k selection is not nested. The benchmark was inspected in earlier development; it is not an untouched final test.</p><h3>Five pooled repetition scores</h3>'
    content+=table(['Configuration','Repeat','Error ↓','Accuracy','Incorrect / scans'],[
        [label(a,str(v.architecture)),int(v.repeat),f'{100*v.error:.3f}%',f'{100*v.accuracy:.3f}%',f'{int(v.errors)} / {int(v.rows)}']
        for a,df in cv_data.items() for v in df.query('kind == "repeat"').itertuples()])+'</section>'
    content+='<section><h2>3. Validation by building</h2><p>Subgroup results show where errors occur. An overall leader need not lead in every building.</p>'
    content+=table(['Configuration','Building','Scans','Error ↓','Incorrect','Accuracy'],[
        [label(a,str(v.architecture)),str(v.scope).replace('building_',''),int(v.rows),f'{100*v.error:.2f}%',int(v.errors),f'{100*v.accuracy:.2f}%']
        for a,(val,_,_) in details.items() for v in val[val.scope!='overall'].itertuples()])+'</section>'
    content+='<section><h2>4. Validation confusion matrices</h2><p>Rows = actual floor; columns = predicted floor. Diagonal cells are correct predictions; all other cells are errors. Floor labels are ordered 0–4. Each matrix contains all evaluated scans.</p><div class="matrices">'
    for a,(_,_,cm) in details.items():
        for arch in ARCHS:
            matrix=cm.query('architecture == @arch').pivot(index='true_floor',columns='predicted_floor',values='count')
            content+='<article><h3>'+html.escape(label(a,arch))+'</h3>'+table(['Actual / predicted',0,1,2,3,4],[[i]+[int(matrix.loc[i,j]) for j in FLOORS] for i in FLOORS])+'</article>'
    content+='</div><p>Floor 4 has comparatively few validation observations; read its errors together with its support.</p></section>'
    content+='<section><h2>5. Model settings and k selection</h2><p>PCA–kNN: 200 fold-local components, full SVD, no whitening or StandardScaler; exact Euclidean neighbours and equal votes. Random Forest: 200 trees, sqrt features per split, unrestricted depth, minimum leaf 1, bootstrap enabled, seed 42; no PCA.</p>'
    if 'PCA_KNN' in meta_data:
        content+=table(['Architecture','Training scope','Rows','Selected k','Search range','Selection CV error ↓','PCA variance retained'],[
            [arch,'Combined' if part=='All' else 'Building '+part,v['training_rows'],v['k'],f'1–{v["tested_through"]}',f'{100*(1-v["mean_selection_cv_accuracy"]):.3f}%',f'{100*v["retained_variance"]:.3f}%']
            for arch,parts in meta_data['PCA_KNN']['partitions'].items() for part,v in parts.items()])
        search_path=ROOT/'Results/PCA_KNN/K_Search.csv'
        if sha256(search_path)!=meta_data['PCA_KNN']['k_search_sha256']:
            raise ValueError('k-search score file changed.')
        search=pd.read_csv(search_path)
        content+='<details><summary>Show every evaluated k and its mean CV error</summary>'+table(['Architecture','Scope','k','CV error ↓','Selected'],[
            [v.architecture,'Combined' if str(v.partition)=='All' else 'Building '+str(v.partition),v.k,f'{100*v.mean_cv_error:.3f}%','Yes' if v.selected else ''] for v in search.itertuples()])+'</details>'
    content+='<p>k starts at 1–25. Extend through 50, 100, 150, 200, 250 unless the best k is at least five below the current limit and the final five candidates are strictly worse. Score ties within 10⁻¹² choose smaller k. This finite search does not prove a global optimum.</p></section>'
    content+='<section><h2>6. Interpretation and reproducibility</h2><p>Repeated locations and duplicate fingerprints may cross random CV folds. High CV accuracy is not evidence of unseen-location performance. Building-wise models require a known building ID; no-signal scans receive a prediction without abstention. PCA and forest settings are fixed design choices.</p><p>Evaluate loads saved pipelines only: no retraining, PCA refitting or validation-based k selection. All 520 WAP signals are used, with 100 mapped to −110 and genuine zeros retained.</p>'
    content+='<details><summary>Dataset and model provenance</summary><p>Validation file SHA-256: <code>'+html.escape(next(iter(bench)))+'</code></p>'+table(['Algorithm','Training SHA-256','Runtime recorded'],[
        [a,m['training_sha256'],', '.join(k+' '+str(v) for k,v in m['versions'].items())] for a,m in meta_data.items()])+'</details></section>'
    css='''*{box-sizing:border-box}body{margin:0;background:#fff;color:#171717;font:15px/1.55 Arial,sans-serif}main{max-width:1180px;margin:auto;padding:40px 24px 64px}header{border-bottom:2px solid #171717;padding-bottom:22px}h1{font-size:34px;line-height:1.15;margin:10px 0}h2{font-size:23px;line-height:1.25;margin:0 0 14px}h3{font-size:16px;line-height:1.3}section{padding:30px 0;border-bottom:1px solid #bbb}.eyebrow{font-size:11px;letter-spacing:1.2px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}.card{border:1px solid #333;padding:18px}.card h3{min-height:42px;margin:0 0 16px}.big{font-size:38px;font-weight:700;line-height:1}.card p{font-size:13px;margin-bottom:0}.finding,.notice{padding:14px;border-left:3px solid #171717;background:#f6f6f6}.table-wrap{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px 9px;border-bottom:1px solid #ccc;vertical-align:top}th{border-bottom:2px solid #333;font-weight:700}th:not(:first-child),td:not(:first-child){font-variant-numeric:tabular-nums}.matrices{display:grid;grid-template-columns:1fr 1fr;gap:20px}.matrices article{min-width:0}details{margin-top:18px}summary{font-weight:700;cursor:pointer;padding:10px 0}code{overflow-wrap:anywhere;font-size:12px}@media(max-width:760px){main{padding:24px 16px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.matrices{grid-template-columns:1fr}h1{font-size:28px}h2{font-size:21px}.card{padding:14px}.big{font-size:32px}}@media(max-width:380px){.cards{grid-template-columns:1fr}.card h3{min-height:0}}@media print{main{padding:0}section,.card{break-inside:avoid}.table-wrap{overflow:visible}body{font-size:11px}table{font-size:10px}details{display:none}}'''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WiFi Floor Classification — Results</title><style>'+css+'</style></head><body><main>'+content+'</main></body></html>'
    comparison.to_csv(ROOT/'Results/Comparison.csv',index=False)
    path=ROOT/'Results/Comparison.html'; path.write_text(page,encoding='utf-8')
    print(f'Report ready: {path}',flush=True); return path


def main(only=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['clean','evaluate','train','report'])
    parser.add_argument('--overwrite',action='store_true',help='Explicitly replace existing trained models.')
    parser.add_argument('--validation',type=Path,default=ROOT/'Dataset/validationData.csv',help='Labelled validation CSV; evaluate only.')
    args=parser.parse_args()
    algorithms=[only] if only else list(ALGOS)
    if args.overwrite and args.command!='train': parser.error('--overwrite is only for train.')
    if args.command=='clean': clean_all()
    elif args.command=='train':
        for algo in algorithms: train(algo,args.overwrite)
    elif args.command=='evaluate':
        invalidate_comparison()
        destination=ROOT/'Cleaned_Dataset/validationData_cleaned.csv'
        if args.validation.resolve()!=(ROOT/'Dataset/validationData.csv').resolve():
            # Do not replace the supplied cleaned validation file for a custom benchmark.
            frame=read_data(args.validation)
            frame[WAPS]=SignalEncoder().fit_transform(frame[WAPS])
        else: frame=clean_file(args.validation,destination)
        for algo in algorithms: evaluate(algo,frame,args.validation)
        report()
    else:
        invalidate_comparison(); report()


if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError,FileExistsError) as e:
        print(f'ERROR: {e}',file=sys.stderr); sys.exit(1)
