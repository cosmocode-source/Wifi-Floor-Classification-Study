"""Clean TRAINING and VALIDATION separately. No learned preprocessing happens here.

Example:
    python prepare_data.py --train-csv path/to/trainingData.csv \
        --validation-csv path/to/validationData.csv

The package already includes prepared data, so you only need this command when
replacing the source dataset. Existing files are protected unless --overwrite is used.
"""
from pathlib import Path
import argparse
import json
from shared import ROOT, WAPS, read_data, file_hash, save_json, build_html, ensure_shared_folds


def prepare(train_path, validation_path, *, overwrite=False):
    destination = ROOT / 'Cleaned_Data'
    destination.mkdir(exist_ok=True)
    outputs = [destination / 'Training.csv', destination / 'Validation.csv']
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError('Cleaned data already exists. Use --overwrite intentionally.')

    # Validate both inputs before writing either file. Do not drop rows or duplicates.
    frames = [read_data(Path(train_path), for_training=True),
              read_data(Path(validation_path), for_training=False)]
    summary = {'transformation': 'WAP values of 100 -> -110; genuine zeros preserved',
               'learned_preprocessing': 'None. PCA is fitted only during KNN training/CV.',
               'rows_dropped': 0, 'files': {}}
    for source, frame, output in zip([Path(train_path), Path(validation_path)], frames, outputs):
        changed = int((frame[WAPS] == 100).to_numpy().sum())
        frame.loc[:, WAPS] = frame[WAPS].replace(100, -110)
        frame.to_csv(output, index=False)
        summary['files'][output.name] = {
            'source_name': source.name, 'source_sha256': file_hash(source),
            'prepared_sha256': file_hash(output), 'rows': len(frame),
            'columns': len(frame.columns), 'sentinel_cells_replaced': changed}
    save_json(destination / 'Cleaning_Summary.json', summary)
    ensure_shared_folds(frames[0], overwrite=True)
    (ROOT / 'Results/Results_Summary.pdf').unlink(missing_ok=True)
    build_html()  # The report hides any scores made stale by replacing data.
    print('Prepared Training.csv and Validation.csv separately. No rows were removed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, default=ROOT / 'trainingData.csv')
    parser.add_argument('--validation-csv', type=Path, default=ROOT / 'validationData.csv')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    prepare(args.train_csv, args.validation_csv, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
