"""Random Forest on 520 cleaned WiFi signals; fixed baseline settings."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from Cleaned_Dataset.clean_data import SignalEncoder

def rf_pipeline():
    return Pipeline([
        ('encode', SignalEncoder()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_features='sqrt',
             max_depth=None, min_samples_leaf=1, bootstrap=True, random_state=42, n_jobs=2))])


