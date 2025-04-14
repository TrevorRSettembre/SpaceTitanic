import time
import multiprocessing
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

# Simulate a medium workload
def train_fold(X, y, train_idx, valid_idx):
    model = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=1)
    model.fit(X[train_idx], y[train_idx])
    return model.score(X[valid_idx], y[valid_idx])

def run_cv_simulation(n_jobs_inner=1, n_jobs_outer=1):
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    def run_one_trial():
        scores = Parallel(n_jobs=n_jobs_inner)(
            delayed(train_fold)(X, y, train_idx, valid_idx)
            for train_idx, valid_idx in splits
        )
        return sum(scores) / len(scores)

    print(f"\nRunning benchmark: Outer trials = {n_jobs_outer}, Inner CV folds = {n_jobs_inner}")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs_outer)(
        delayed(run_one_trial)() for _ in range(4)  # simulate 4 Optuna trials
    )

    duration = time.time() - start_time
    print(f"✅ Finished in {duration:.2f}s | Avg accuracy: {sum(results)/len(results):.4f}")
    return duration

# List of (outer, inner) combinations to try
cpu_count = multiprocessing.cpu_count()
print(f"Detected CPU cores: {cpu_count}")

test_configs = [
    (1, 1),
    (2, 2),
    (4, 2),
    (4, 4),
    (2, 6),
    (1, 8),
]

# Run benchmarks
for outer, inner in test_configs:
    run_cv_simulation(n_jobs_inner=inner, n_jobs_outer=outer)
