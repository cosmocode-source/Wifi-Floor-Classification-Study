"""Shared checks, honest evaluation, artifact handling and offline HTML reporting.

Only WAP001..WAP520 enter a classifier. FLOOR is the label. BUILDINGID routes
rows to the appropriate building model. Combined models use these same WAP predictors; BUILDINGID is not an extra
combined-model input. Other metadata is used ONLY to define
training CV groups, never as model inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import html
import json
import os
import tempfile
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
WAPS = [f'WAP{i:03d}' for i in range(1, 521)]
BUILDINGS = [0, 1, 2]
FLOORS = [0, 1, 2, 3, 4]
N_FOLDS = 3
SEED = 42
MODEL_IDS = ['Building_Specific_KNN', 'Building_Specific_Random_Forest',
             'Combined_Buildings_KNN', 'Combined_Buildings_Random_Forest']
DISPLAY_NAMES = {'Building_Specific_KNN': 'Building-specific PCA-kNN',
                 'Building_Specific_Random_Forest': 'Building-specific Random Forest',
                 'Combined_Buildings_KNN': 'Combined-building PCA-kNN',
                 'Combined_Buildings_Random_Forest': 'Combined-building Random Forest'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents readers seeing a half-written JSON file.
    text = json.dumps(content, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=Path(path).parent, delete=False) as handle:
        handle.write(text)
        temporary_path = handle.name
    os.replace(temporary_path, path)


def check_waps(frame):
    missing = [column for column in WAPS if column not in frame]
    if missing:
        raise ValueError(f'Missing WAP columns: {missing[:5]}')
    # Select in the canonical order even if the input CSV columns were reordered.
    values = frame[WAPS].apply(pd.to_numeric, errors='raise')
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('WAP values must be finite: no missing values or infinities.')
    invalid = (array != 100) & ((array < -110) | (array > 0))
    if invalid.any():
        raise ValueError('WAP values must be -110..0, or the raw sentinel 100.')
    return values


def read_data(path, *, for_training=False, require_floor=True):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV column names are not allowed.')
    if set(column for column in header if column.startswith('WAP')) != set(WAPS):
        raise ValueError('The CSV must contain exactly WAP001..WAP520 as its WAP columns.')
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError('The CSV contains no rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    frame.loc[:, WAPS] = check_waps(frame)
    required = ['BUILDINGID'] + (['FLOOR'] if require_floor else [])
    if for_training:
        required += ['LONGITUDE', 'LATITUDE']
    for column in required:
        if column not in frame:
            raise ValueError(f'Missing required column: {column}')
        values = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'{column} contains missing/nonfinite values.')
        frame[column] = values
    for column, allowed in [('BUILDINGID', BUILDINGS), ('FLOOR', FLOORS)]:
        if column in required:
            if not frame[column].isin(allowed).all():
                raise ValueError(f'{column} must contain values in {allowed}.')
            frame[column] = frame[column].astype(int)
    if for_training and set(frame.BUILDINGID) != set(BUILDINGS):
        raise ValueError('Training must include all three known buildings.')
    return frame


class MissingSignalEncoder(BaseEstimator, TransformerMixin):
    """A FIXED transformation, so no statistics are learned from validation.

    Keep this class in shared.py so serialized pipelines load consistently.
    Saving it inside the pipeline also permits inference on raw sentinel-100 CSVs.
    """
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != WAPS:
            raise ValueError('Model predictors must be exactly WAP001..WAP520 in order.')
        check_waps(X)
        self.n_features_in_ = len(WAPS)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != WAPS:
            raise ValueError('Predictor schema/order does not match the saved pipeline.')
        return check_waps(X).replace(100, -110).to_numpy(dtype=float)


def training_groups(frame):
    """Groups connect exact locations OR identical 520-WAP fingerprints.

    Location = (building, longitude, latitude, floor). Fingerprint duplicates
    are joined EVEN across buildings, because combined models see all buildings.
    Connected groups cannot straddle training and held-out CV folds.
    Coordinates are grouping metadata, never predictors or spatial buffers.
    """
    parent = np.arange(len(frame))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def join(codes):
        first_row = {}
        for index, code in enumerate(codes):
            if code in first_row:
                parent[find(index)] = find(first_row[code])
            else:
                first_row[code] = index

    locations = pd.MultiIndex.from_frame(frame[['BUILDINGID', 'LONGITUDE', 'LATITUDE', 'FLOOR']])
    join(pd.factorize(locations)[0])
    fingerprints = pd.MultiIndex.from_frame(frame[WAPS].replace(100, -110))
    join(pd.factorize(fingerprints)[0])
    return np.array([find(index) for index in range(len(frame))])


def choose_folds(X, stratification_labels, groups):
    """Keep every building/floor stratum represented in all three CV folds."""
    labels = set(stratification_labels)
    for label in labels:
        if len(set(groups[np.asarray(stratification_labels) == label])) < N_FOLDS:
            raise ValueError(f'Stratum {label} has too few independent groups for {N_FOLDS}-fold CV.')
    for seed in range(SEED, SEED + 30):
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, stratification_labels, groups))
        valid = all(set(stratification_labels.iloc[a]) == labels and
                    set(stratification_labels.iloc[b]) == labels for a, b in splits)
        if valid:
            for a, b in splits:
                assert set(groups[a]).isdisjoint(set(groups[b]))
            return splits, seed
    raise ValueError('Cannot construct grouped folds with all strata. Do not silently switch to random-row CV.')


def ensure_shared_folds(frame=None, *, overwrite=False):
    """One training-only fold manifest for ALL four model/architecture combinations."""
    data_dir = ROOT / 'Cleaned_Data'
    path = data_dir / 'Training_CV_Folds.csv'
    info_path = data_dir / 'CV_Folds_Metadata.json'
    source_hash = file_hash(data_dir / 'Training.csv')
    if frame is None:
        frame = read_data(data_dir / 'Training.csv', for_training=True)
    if path.exists() and info_path.exists() and not overwrite:
        info = json.loads(info_path.read_text())
        if info['training_sha256'] != source_hash or info['fold_file_sha256'] != file_hash(path):
            raise RuntimeError('Training data or fold manifest changed. Regenerate folds intentionally.')
        manifest = pd.read_csv(path)
    else:
        groups = training_groups(frame)
        strata = frame.BUILDINGID.astype(str) + ':' + frame.FLOOR.astype(str)
        splits, seed = choose_folds(frame[WAPS], strata, groups)
        assignment = np.full(len(frame), -1, dtype=int)
        for number, (_, held_out) in enumerate(splits):
            assignment[held_out] = number
        assert (assignment >= 0).all()
        manifest = pd.DataFrame({'source_row': np.arange(len(frame)),
                                 'BUILDINGID': frame.BUILDINGID, 'FLOOR': frame.FLOOR,
                                 'group_id': groups, 'cv_fold': assignment})
        manifest.to_csv(path, index=False)
        save_json(info_path, {'training_sha256': source_hash, 'fold_file_sha256': file_hash(path),
                              'cv_folds': N_FOLDS, 'seed': seed, 'groups': len(set(groups)),
                              'definition': 'Connected exact building/location/floor keys and identical WAP vectors, including cross-building duplicates',
                              'stratification': 'BUILDINGID + FLOOR; training labels only'})
    # Row order, labels and group isolation must agree before any model can train.
    assert len(manifest) == len(frame)
    assert np.array_equal(manifest.source_row, np.arange(len(frame)))
    assert np.array_equal(manifest.BUILDINGID, frame.BUILDINGID)
    assert np.array_equal(manifest.FLOOR, frame.FLOOR)
    assert set(manifest.cv_fold) == set(range(N_FOLDS))
    assert manifest.groupby('group_id').cv_fold.nunique().max() == 1
    return manifest


def metrics(y, predicted, labels):
    return {'rows': len(y), 'correct': int(np.sum(np.asarray(y) == predicted)),
            'errors': int(np.sum(np.asarray(y) != predicted)),
            'accuracy': float(accuracy_score(y, predicted)),
            'macro_f1': float(f1_score(y, predicted, labels=labels, average='macro', zero_division=0)),
            'floor_labels': list(labels),
            'confusion_matrix': confusion_matrix(y, predicted, labels=labels).tolist()}


def identity(architecture, model):
    if architecture not in ['Building_Specific', 'Combined_Buildings'] or model not in ['KNN', 'Random_Forest']:
        raise ValueError('Unknown architecture/model.')
    return f'{architecture}_{model}'


def model_directory(architecture, model):
    return ROOT / architecture / model / 'Saved_Models'


def choose_candidate(results):
    """Choose CV accuracy, then macro-F1; exact ties prefer smaller k, then grid order.

    This is an objective selection rule. An elbow is a visual diagnostic, not a
    reliable unique answer for classification. Validation scores are not consulted.
    """
    accuracy = np.asarray(results['mean_test_accuracy'])
    eligible = np.flatnonzero(np.isclose(accuracy, accuracy.max(), rtol=0, atol=1e-12))
    f1 = np.asarray(results['mean_test_macro_f1'])
    eligible = eligible[np.isclose(f1[eligible], f1[eligible].max(), rtol=0, atol=1e-12)]
    return int(min(eligible, key=lambda index: (
        results['params'][index].get('classifier__n_neighbors', 0), index)))


def pca_diagnostics(final_model, X):
    """Training-only explained variance; never fit this curve on validation.

    Full spectrum shows how many PCs reach 90/95/99% variance. The fitted model
    may use a fixed count or a fold-adaptive variance target selected by CV.
    """
    encoded = MissingSignalEncoder().fit_transform(X)
    spectrum = PCA(svd_solver='full', whiten=False).fit(encoded)
    cumulative = np.cumsum(spectrum.explained_variance_ratio_)
    fitted = final_model.named_steps['pca']
    return {'cumulative_explained_variance': cumulative.tolist(),
            'components_for_retention': {str(target): int(np.searchsorted(cumulative, target) + 1)
                                         for target in [0.90, 0.95, 0.99]},
            'selected_component_count': int(fitted.n_components_),
            'selected_explained_variance': float(fitted.explained_variance_ratio_.sum()),
            'curve_population': 'Final fitting training rows only; descriptive, not final-validation selection'}


def train_model(architecture, model, pipeline, parameter_grid, *, overwrite=False):
    key = identity(architecture, model)
    destination = model_directory(architecture, model)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError('Saved models exist. Use --overwrite only to intentionally replace them.')
    frame = read_data(ROOT / 'Cleaned_Data/Training.csv', for_training=True)
    manifest = ensure_shared_folds(frame)
    # Training/selection uses no official validation rows or scores.
    metadata = {'model': model, 'architecture': architecture, 'model_id': key,
                'trained_utc': datetime.now(timezone.utc).isoformat(),
                'training_sha256': file_hash(ROOT / 'Cleaned_Data/Training.csv'),
                'fold_manifest_sha256': file_hash(ROOT / 'Cleaned_Data/Training_CV_Folds.csv'),
                'training_rows': len(frame), 'cv_folds': N_FOLDS,
                'predictors': 'WAP001..WAP520 only; no BUILDINGID in combined-model feature vector',
                'selection_policy': 'Highest mean grouped-CV accuracy, then macro-F1; ties smaller k then grid order',
                'cv_warning': 'Selected settings scored on selection folds, not an unbiased independent test.',
                'libraries': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                              'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
                'partitions': {}}
    partitions = [(str(b), np.flatnonzero(frame.BUILDINGID.to_numpy() == b)) for b in BUILDINGS]
    if architecture == 'Combined_Buildings':
        partitions = [('All_Buildings', np.arange(len(frame)))]
    oof = np.full(len(frame), -999, dtype=int)
    search_rows, models_to_save = [], {}
    started = time.perf_counter()
    import tempfile
    for partition, indices in partitions:
        subset = frame.iloc[indices].reset_index(drop=True)
        X, y = subset[WAPS], subset.FLOOR
        fold_numbers = manifest.cv_fold.to_numpy()[indices]
        splits = [(np.flatnonzero(fold_numbers != fold), np.flatnonzero(fold_numbers == fold))
                  for fold in range(N_FOLDS)]
        if any(set(y.iloc[a]) != set(y) or set(y.iloc[b]) != set(y) for a, b in splits):
            raise ValueError('The shared folds lack a floor in this partition.')
        print(f'{DISPLAY_NAMES[key]} | {partition}: {len(subset)} rows; {len(parameter_grid.get("pca__n_components", [0])) * len(parameter_grid.get("classifier__n_neighbors", [0])) if model=="KNN" else 4} candidates', flush=True)
        with tempfile.TemporaryDirectory(prefix='wifi_grouped_cv_') as cache:
            candidate = clone(pipeline).set_params(memory=joblib.Memory(cache, verbose=0))
            search = GridSearchCV(candidate, parameter_grid, cv=splits,
                                  scoring={'accuracy': 'accuracy', 'macro_f1': 'f1_macro'},
                                  refit=choose_candidate, n_jobs=1, error_score='raise',
                                  return_train_score=False)
            search.fit(X, y)
            selected = search.best_params_
            for train_indices, test_indices in splits:
                fitted = clone(pipeline).set_params(**selected)
                fitted.fit(X.iloc[train_indices], y.iloc[train_indices])
                oof[indices[test_indices]] = fitted.predict(X.iloc[test_indices])
            # Identical final population size across architectures: every training row.
            final_model = clone(pipeline).set_params(**selected)
            final_model.fit(X, y)
            models_to_save[partition] = final_model
            info = {'training_rows': len(subset), 'parameters': selected,
                    'selected_cv': metrics(y, oof[indices], sorted(int(v) for v in y.unique()))}
            if model == 'KNN':
                info['pca'] = pca_diagnostics(final_model, X)
                info['k_curve'] = []
            for candidate_index, params in enumerate(search.cv_results_['params']):
                row = {'partition': partition, 'parameters': json.dumps(params, sort_keys=True),
                       'mean_cv_accuracy': float(search.cv_results_['mean_test_accuracy'][candidate_index]),
                       'std_cv_accuracy': float(search.cv_results_['std_test_accuracy'][candidate_index]),
                       'mean_cv_macro_f1': float(search.cv_results_['mean_test_macro_f1'][candidate_index]),
                       'selected': candidate_index == search.best_index_}
                if model == 'KNN':
                    row.update({'pca_setting': params['pca__n_components'],
                                'k': params['classifier__n_neighbors']})
                    if params['pca__n_components'] == selected['pca__n_components']:
                        info['k_curve'].append({'k': int(params['classifier__n_neighbors']),
                                                'mean_cv_error_percent': 100 * (1 - row['mean_cv_accuracy']),
                                                'std_cv_accuracy_percent': 100 * row['std_cv_accuracy']})
                search_rows.append(row)
            metadata['partitions'][partition] = info
    assert not (oof == -999).any()
    destination.mkdir(parents=True, exist_ok=True)
    for partition, fitted in models_to_save.items():
        suffix = f'Building_{partition}' if partition != 'All_Buildings' else partition
        path = destination / f'{model}_{suffix}.joblib'
        joblib.dump(fitted, path, compress=3)
        metadata['partitions'][partition]['artifact_filename'] = path.name
        metadata['partitions'][partition]['artifact_sha256'] = file_hash(path)
    metadata['selected_cv_pooled'] = metrics(frame.FLOOR, oof, FLOORS)
    metadata['training_seconds_including_search'] = time.perf_counter() - started
    save_json(destination / f'{model}_Metadata.json', metadata)
    pd.DataFrame(search_rows).to_csv(ROOT / f'Results/{key}_CV_Search.csv', index=False)
    pd.DataFrame({'source_row': np.arange(len(frame)), 'BUILDINGID': frame.BUILDINGID,
                  'true_floor': frame.FLOOR, 'predicted_floor': oof,
                  'cv_fold': manifest.cv_fold}).to_csv(ROOT / f'Results/{key}_CV_Predictions.csv', index=False)
    for filename in [f'{key}_Metrics.json', f'{key}_Predictions.csv',
                     'Model_Comparison.csv', 'Validation_Predictions.csv', 'Results_Summary.pdf']:
        (ROOT / 'Results' / filename).unlink(missing_ok=True)
    build_html()
    print(f'Saved {key}. Selected grouped-CV accuracy: {metadata["selected_cv_pooled"]["accuracy"]:.2%}', flush=True)


def load_models(architecture, model):
    destination = model_directory(architecture, model)
    metadata = json.loads((destination / f'{model}_Metadata.json').read_text())
    if metadata['libraries']['sklearn'] != sklearn.__version__:
        raise RuntimeError('Saved sklearn version differs. Install requirements.txt or intentionally retrain.')
    pipelines = {}
    for partition, info in metadata['partitions'].items():
        # Do not let a changed metadata filename point outside this model folder.
        if Path(info['artifact_filename']).name != info['artifact_filename']:
            raise RuntimeError('Invalid artifact filename.')
        path = destination / info['artifact_filename']
        if file_hash(path) != info['artifact_sha256']:
            raise RuntimeError(f'Artifact integrity check failed: {path.name}')
        # Only load locally produced TRUSTED joblib files. A hash detects changes
        # but does not make an untrusted pickle safe to execute.
        pipelines[partition] = joblib.load(path)
    return pipelines, metadata


def predict_frame(architecture, pipelines, frame):
    if architecture == 'Combined_Buildings':
        # One PCA/classifier sees every scan. No building-specific routing occurs.
        return pipelines['All_Buildings'].predict(frame[WAPS]).astype(int)
    predicted = np.full(len(frame), -999, dtype=int)
    for building in BUILDINGS:
        mask = frame.BUILDINGID.to_numpy() == building
        if mask.any():
            predicted[mask] = pipelines[str(building)].predict(frame.loc[mask, WAPS])
    if (predicted == -999).any():
        raise ValueError('Some rows lack a known-building pipeline.')
    return predicted


def evaluate_model(architecture, model):
    key = identity(architecture, model)
    pipelines, metadata = load_models(architecture, model)
    if metadata['training_sha256'] != file_hash(ROOT / 'Cleaned_Data/Training.csv'):
        raise RuntimeError('Training data changed after fitting. Retrain before evaluating.')
    if metadata['fold_manifest_sha256'] != file_hash(ROOT / 'Cleaned_Data/Training_CV_Folds.csv'):
        raise RuntimeError('CV manifest changed after fitting; use matching models for the comparison.')
    frame = read_data(ROOT / 'Cleaned_Data/Validation.csv')
    predicted = predict_frame(architecture, pipelines, frame)
    result = {'model_id': key, 'architecture': architecture, 'model': model,
              'evaluation_role': 'official_validation_benchmark', 'validation_already_inspected': True,
              'evaluated_utc': datetime.now(timezone.utc).isoformat(),
              'training_sha256': metadata['training_sha256'],
              'validation_sha256': file_hash(ROOT / 'Cleaned_Data/Validation.csv'),
              'artifact_hashes': {part: info['artifact_sha256'] for part, info in metadata['partitions'].items()},
              'pooled': metrics(frame.FLOOR, predicted, FLOORS), 'buildings': {}}
    for building in BUILDINGS:
        mask = frame.BUILDINGID.to_numpy() == building
        if mask.any():
            # Include any extra predicted floor in the matrix: never hide an
            # invalid-for-this-building prediction from a combined model.
            labels = sorted(set(int(v) for v in frame.loc[mask, 'FLOOR']) |
                            set(int(v) for v in predicted[mask]))
            result['buildings'][str(building)] = metrics(frame.loc[mask, 'FLOOR'], predicted[mask], labels)
    prediction_path = ROOT / f'Results/{key}_Predictions.csv'
    pd.DataFrame({'source_row': np.arange(len(frame)), 'BUILDINGID': frame.BUILDINGID,
                  'true_floor': frame.FLOOR, 'predicted_floor': predicted}).to_csv(prediction_path, index=False)
    result['predictions_sha256'] = file_hash(prediction_path)
    save_json(ROOT / f'Results/{key}_Metrics.json', result)
    (ROOT / 'Results/Results_Summary.pdf').unlink(missing_ok=True)
    build_html()
    print(f'{DISPLAY_NAMES[key]} benchmark: {result["pooled"]["accuracy"]:.2%}; '
          f'{result["pooled"]["errors"]} errors / {len(frame)} rows', flush=True)


def demo_model(architecture, model, input_csv=None):
    pipelines, _ = load_models(architecture, model)
    if input_csv:
        frame = read_data(input_csv, require_floor=False)
        print('Inference on supplied scans, not a benchmark.')
    else:
        frame = read_data(ROOT / 'Cleaned_Data/Training.csv').groupby('BUILDINGID', group_keys=False).head(1)
        print('TRAINING REHEARSAL: one training scan per building, not a validation score.')
    predicted = predict_frame(architecture, pipelines, frame)
    view = pd.DataFrame({'building': frame.BUILDINGID.to_numpy(), 'predicted_floor': predicted})
    view['detected_WAPs'] = ((frame[WAPS] != 100) & (frame[WAPS] != -110)).sum(axis=1).to_numpy()
    if 'FLOOR' in frame:
        view['provided_floor'] = frame.FLOOR.to_numpy()
    if (view.detected_WAPs == 0).any():
        print('Warning: a scan has no detected access points; its prediction is not reliable. No abstention is implemented.')
    print(view.to_string(index=False))


def make_table(headers, rows):
    head = ''.join(f'<th>{html.escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def line_svg(xs, ys, *, x_label, y_label, title, selected_x=None, percentage=True):
    """Small offline SVG diagnostic chart, with labelled axes and real data points."""
    width, height = 540, 270
    left, right, top, bottom = 58, 20, 36, 48
    xmin, xmax = min(xs), max(xs)
    ymax = 100 if percentage else max(ys) * 1.15
    ymax = max(ymax, 1.0)
    sx = lambda x: left + (x - xmin) / max(xmax - xmin, 1) * (width-left-right)
    sy = lambda y: height-bottom - y / ymax * (height-top-bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<title>{html.escape(title)}</title>',
             f'<text x="{left}" y="20" font-size="13" font-weight="bold">{html.escape(title)}</text>']
    for fraction in [0, .25, .5, .75, 1]:
        value = ymax * fraction; y = sy(value)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dbe4ec"/>',
                  f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{value:.1f}</text>']
    ticks = sorted(set([xmin, xmax] + ([50, 200] if xmax >= 200 else [5, 9, 15, 21])))
    for value in ticks:
        if xmin <= value <= xmax:
            parts.append(f'<text x="{sx(value):.1f}" y="{height-bottom+20}" text-anchor="middle" font-size="11">{value}</text>')
    points = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(xs,ys))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#075985" stroke-width="2.5"/>')
    if len(xs) <= 30:
        parts += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#075985"><title>{x}: {y:.3f}</title></circle>' for x,y in zip(xs,ys)]
    if selected_x is not None:
        closest = min(range(len(xs)), key=lambda index: abs(xs[index]-selected_x))
        x,y=sx(xs[closest]),sy(ys[closest])
        parts += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#b45309" stroke-dasharray="4 3"/>',
                  f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#b45309"/>']
    parts += [f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12">{html.escape(x_label)}</text>',
              f'<text x="14" y="{height/2}" transform="rotate(-90 14 {height/2})" text-anchor="middle" font-size="12">{html.escape(y_label)}</text>', '</svg>']
    return ''.join(parts)


def build_html():
    """Four-way standalone comparison and training-only PCA/k diagnostic plots."""
    result_dir = ROOT / 'Results'
    result_dir.mkdir(exist_ok=True)
    available, metadata = {}, {}
    for key in MODEL_IDS:
        architecture = 'Building_Specific' if key.startswith('Building_Specific') else 'Combined_Buildings'
        model = 'Random_Forest' if key.endswith('Random_Forest') else 'KNN'
        meta_path = model_directory(architecture, model) / f'{model}_Metadata.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta['training_sha256'] == file_hash(ROOT / 'Cleaned_Data/Training.csv') and
                meta['fold_manifest_sha256'] == file_hash(ROOT / 'Cleaned_Data/Training_CV_Folds.csv')):
                metadata[key] = meta
        metric_path = result_dir / f'{key}_Metrics.json'
        prediction_path = result_dir / f'{key}_Predictions.csv'
        if key in metadata and metric_path.exists() and prediction_path.exists():
            result = json.loads(metric_path.read_text())
            hashes = {part: info['artifact_sha256'] for part,info in metadata[key]['partitions'].items()}
            if (result['artifact_hashes'] == hashes and
                result['training_sha256'] == metadata[key]['training_sha256'] and
                result['validation_sha256'] == file_hash(ROOT / 'Cleaned_Data/Validation.csv') and
                result['predictions_sha256'] == file_hash(prediction_path)):
                available[key] = result
    comparison, combined, cards = [], None, ''
    for key in MODEL_IDS:
        label = DISPLAY_NAMES[key]
        if key in available:
            m = available[key]['pooled']
            comparison.append({'Model': label, 'Rows': m['rows'], 'Correct': m['correct'], 'Errors': m['errors'],
                               'Accuracy_percent': round(100*m['accuracy'],4), 'Macro_F1_percent': round(100*m['macro_f1'],4)})
            cards += f'<article class="card"><div class="eyebrow">{label}</div><div class="score">{m["accuracy"]:.2%}</div><p>{m["errors"]} errors / {m["rows"]} scans</p><b>{m["macro_f1"]:.2%} macro-F1</b></article>'
            p = pd.read_csv(result_dir/f'{key}_Predictions.csv')
            keys = ['source_row','BUILDINGID','true_floor']
            if combined is None:
                combined=p[keys].copy()
            else:
                pd.testing.assert_frame_equal(combined[keys],p[keys])
            combined[key+'_floor']=p.predicted_floor
        else:
            cards += f'<article class="card"><div class="eyebrow">{label}</div><h3>Awaiting evaluation</h3></article>'
    if comparison:
        pd.DataFrame(comparison).to_csv(result_dir/'Model_Comparison.csv',index=False)
        combined.to_csv(result_dir/'Validation_Predictions.csv',index=False)
    else:
        (result_dir/'Model_Comparison.csv').unlink(missing_ok=True)
        (result_dir/'Validation_Predictions.csv').unlink(missing_ok=True)
    status = f'{len(available)} of 4 configurations evaluated'
    architecture_rows=[]
    for model in ['KNN','Random_Forest']:
        separate='Building_Specific_'+model;pooled='Combined_Buildings_'+model
        if separate in available and pooled in available:
            a,b=available[separate]['pooled'],available[pooled]['pooled']
            architecture_rows.append([model.replace('_',' '),f'{a["accuracy"]:.2%}',f'{b["accuracy"]:.2%}',
                                      f'{100*(b["accuracy"]-a["accuracy"]):+.3f} percentage points',b['errors']-a['errors']])
    architecture_table=make_table(['Model','Building-specific','Combined','Combined − separate accuracy','Combined − separate errors'],architecture_rows)
    building_rows=[[DISPLAY_NAMES[key],b,m['rows'],m['errors'],f'{m["accuracy"]:.2%}',f'{m["macro_f1"]:.2%}']
                   for key,result in available.items() for b,m in result['buildings'].items()]
    settings_rows,cv_rows,diagnostics,matrices=[],[], '', ''
    for key,meta in metadata.items():
        cv=meta['selected_cv_pooled']
        cv_rows.append([DISPLAY_NAMES[key],f'{cv["accuracy"]:.2%}',f'{cv["macro_f1"]:.2%}',meta['training_rows']])
        charts=''
        for partition,info in meta['partitions'].items():
            params=info['parameters'];parts=[f'{k.split("__")[-1]}={v}' for k,v in params.items()]
            if meta['model']=='KNN':
                partition_label = f'Building {partition}' if partition != 'All_Buildings' else 'All buildings'
                pca=info['pca'];parts.append(f'actual PCs={pca["selected_component_count"]}')
                count=pca['selected_component_count'];selected_k=params['classifier__n_neighbors']
                spectrum=[100*v for v in pca['cumulative_explained_variance']]
                curve=sorted(info['k_curve'],key=lambda row:row['k'])
                a=line_svg(list(range(1,len(spectrum)+1)),spectrum,x_label='Number of principal components',y_label='Cumulative variance (%)',title=f'{partition_label}: PCA explained variance',selected_x=count)
                b=line_svg([r['k'] for r in curve],[r['mean_cv_error_percent'] for r in curve],x_label='Number of neighbours (k)',y_label='Mean grouped-CV error (%)',title=f'{partition_label}: CV error versus k',selected_x=selected_k,percentage=False)
                thresholds=', '.join(f'{float(target):.0%}: {number} PCs' for target,number in pca['components_for_retention'].items())
                charts+=f'<h3>{partition_label}</h3><p>Selected k={selected_k}; {count} PCs; retained variance={pca["selected_explained_variance"]:.2%}. Training-only thresholds: {thresholds}.</p><div class="grid two">{a}{b}</div>'
            settings_rows.append([DISPLAY_NAMES[key],partition,info['training_rows'],', '.join(parts)])
        if charts:
            diagnostics+=f'<details class="panel"><summary>{DISPLAY_NAMES[key]} — PCA and k selection plots</summary><p class="small">Orange markers identify the selected final settings. PCA curves use fitting training rows only. k curves use held-out training folds at the selected PCA setting; their minima, not a subjective elbow, drive selection. Changing k changes votes, not PCA components.</p>{charts}</details>'
        if key in available:
            m=available[key]['pooled']
            matrix=make_table(['True / predicted']+m['floor_labels'],[[label]+row for label,row in zip(m['floor_labels'],m['confusion_matrix'])])
            matrices+=f'<article><h3>{DISPLAY_NAMES[key]}</h3>{matrix}</article>'
    settings=make_table(['Configuration','Training partition','Final training rows','Selected settings'],settings_rows)
    cvtable=make_table(['Configuration','Selected grouped-CV accuracy','Grouped-CV macro-F1','Final training rows'],cv_rows)
    buildings=make_table(['Configuration','Building','Scans','Errors','Accuracy','Macro-F1'],building_rows)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WiFi — Separate versus Combined Buildings</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#102a43;font:16px/1.55 Arial,sans-serif}.wrap{max-width:1250px;margin:auto;padding:40px 24px}h1{font-size:42px;line-height:1.15}h2{font-size:24px}h3{font-size:18px}.eyebrow{font-size:12px;font-weight:bold;color:#075985;text-transform:uppercase;letter-spacing:.08em}p,.small{color:#526777}.small{font-size:13px}.grid{display:grid;gap:18px}.four{grid-template-columns:repeat(4,1fr)}.two{grid-template-columns:repeat(2,1fr)}section,.card,.panel{background:white;border:1px solid #dbe4ec;border-radius:14px;padding:22px;margin:18px 0}.card{margin:0}.card .eyebrow{min-height:38px}.score{font-size:38px;font-weight:bold;margin-top:15px}.notice{border-left:4px solid #d69e2e;background:#fff8e6;padding:16px;margin:20px 0}.badge{background:#e0f2fe;border-radius:20px;padding:6px 12px;font-size:13px}button{border:0;background:#075985;color:white;padding:10px 16px;border-radius:8px;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #dbe4ec;vertical-align:top}th{background:#edf3f8}tbody tr:nth-child(even){background:#f8fafc}.table-wrap{overflow-x:auto}summary{cursor:pointer;font-weight:bold;font-size:19px}svg{width:100%;height:auto;background:#f8fafc;border-radius:10px;font-family:Arial,sans-serif}.footer{font-size:12px;color:#526777;overflow-wrap:anywhere}@media(max-width:1000px){.four{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.four,.two{grid-template-columns:1fr}.wrap{padding:22px 14px}h1{font-size:32px}}@media print{body{background:white;font-size:11px;line-height:1.35}.wrap{padding:0}button{display:none}section,.card,.panel{padding:12px;margin:10px 0;break-inside:avoid}h1{font-size:28px}h2{font-size:18px}h3{font-size:14px}th,td{padding:6px;font-size:10px}.four{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:repeat(2,1fr)}.score{font-size:30px}}
    </style></head><body><main class="wrap"><div class="eyebrow">Known-building floor classification · training-only selection</div><h1>Separate buildings or one combined model?</h1><p>Two model families. Two architectures. The same data and matched CV folds.</p><span class="badge">{{STATUS}}</span> <button onclick="window.print()">Print / save as PDF</button>
    <div class="notice"><b>Benchmark, not an untouched test:</b> official validation has already been inspected. No validation rows or scores enter PCA fitting, k selection, forest tuning or final training. All final configurations use the 19,937 training rows.</div>
    <div class="grid four">{{CARDS}}</div><section><h2>Does building-specific training matter?</h2>{{ARCHITECTURES}}<p class="small">Positive error difference means the combined model made more mistakes. This is a descriptive comparison, not a significance claim. Separate models use BUILDINGID to route to a building pipeline. Combined models use one pipeline for all buildings, with the same 520 WAP predictors; BUILDINGID is not an additional feature. Thus the comparison tests sharing a model versus known-building routing, not every possible combined architecture.</p></section>
    <section><h2>Official validation by building</h2><p class="small">Swipe wide tables sideways on small screens. Per-building macro-F1 uses the observed true/predicted floor union; pooled macro-F1 uses floors 0–4.</p>{{BUILDINGS}}</section>
    <section><h2>How PCA and k are chosen</h2><p><b>PCA:</b> test fixed counts 5, 50 and 200, plus targets retaining 90%, 95% and 99% of variance. A variance target selects a component count separately on each fold’s training rows. PCA centres inputs; no standard scaling or whitening is used.</p><p><b>k:</b> test odd values 1–25. Choose the highest mean grouped-CV accuracy, then macro-F1. Exact ties prefer smaller k, then candidate order. The error curve can reveal a plateau or elbow, but classification does not always have a unique elbow. Variance retention alone does not guarantee the best floor classification.</p>{{SETTINGS}}<p class="small">Random Forest uses cleaned raw WAPs: 200 trees, max_features=sqrt; CV checks depth None/20 and minimum leaf size 1/5.</p></section>
    {{DIAGNOSTICS}}
    <section><h2>Matched training-only CV</h2>{{CV}}<p class="small">All four configurations use the same held-out training rows. Groups connect exact building/location/floor keys and identical WAP vectors, including cross-building duplicates. Fold stratification uses building/floor strata, never final validation. These scores selected settings, so they are not an independent unbiased test. Exact coordinate grouping is not a spatial buffer.</p></section>
    <section><h2>Confusion matrices</h2><p class="small">Rows = true floor; columns = predicted floor. Diagonal entries are correct. Matrices pool buildings.</p><div class="grid two">{{MATRICES}}</div></section>
    <section><h2>Scope and limitations</h2><ul><li>Floor prediction, not room or metre-level positioning.</li><li>Missing signal 100 → −110 is an explicit assumption; genuine zeros remain unchanged.</li><li>The paper’s k=1 and PCA=5/50/200 settings are included, but its full sampling/preprocessing/repeated-CV protocol is not reproduced.</li><li>A combined model might behave differently if BUILDINGID were added as a feature; that is a separate experiment.</li><li>Do not chase scores on this already inspected benchmark and call it an untouched final test.</li><li>No-signal abstention is not implemented.</li></ul></section><footer class="footer">Generated UTC: {{TIME}}<br>Data/model/fold provenance is recorded in the cleaning summary, CV metadata and model/metric JSON files.</footer></main></body></html>'''
    values={'STATUS':html.escape(status),'CARDS':cards,'ARCHITECTURES':architecture_table,'BUILDINGS':buildings,
            'SETTINGS':settings,'DIAGNOSTICS':diagnostics,'CV':cvtable,'MATRICES':matrices,
            'TIME':datetime.now(timezone.utc).isoformat()}
    for key,value in values.items():
        page=page.replace('{{'+key+'}}',value)
    (result_dir/'Model_Comparison.html').write_text(page,encoding='utf-8')


def run_cli(architecture, model, pipeline, parameter_grid):
    key=identity(architecture,model)
    parser=argparse.ArgumentParser(description=DISPLAY_NAMES[key]+': grouped CV, full-training refit, evaluation and demo')
    parser.add_argument('action',choices=['train','evaluate','demo'])
    parser.add_argument('--overwrite',action='store_true',help='Intentionally replace saved models during training.')
    parser.add_argument('--input-csv',type=Path,help='Demo CSV with 520 WAPs and BUILDINGID.')
    args=parser.parse_args()
    try:
        if args.action=='train':train_model(architecture,model,pipeline,parameter_grid,overwrite=args.overwrite)
        elif args.action=='evaluate':evaluate_model(architecture,model)
        else:demo_model(architecture,model,args.input_csv)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        parser.exit(1,f'Error: {error}\n')
