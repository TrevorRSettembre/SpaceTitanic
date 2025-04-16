import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import ydf
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


def process_df(df0: pd.DataFrame) -> pd.DataFrame:
    df0 = df0.copy()

    # Boolean columns — fill NA with False and cast to bool
    for col in ['CryoSleep', 'VIP']:
        df0[col] = df0[col].astype(str).str.lower() == 'true'
        df0[col] = df0[col].fillna(False)

    # Categorical columns — fill with 'Unknown'
    cat_default = {'HomePlanet': 'Unknown', 'Destination': 'Unknown', 'Name': 'Unknown'}
    df0.fillna(cat_default, inplace=True)

      # Age — fill with median and bucket
    df0['Age'] = df0['Age'].fillna(df0['Age'].median())
    df0['AgeBucket'] = pd.cut(df0['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    
    # Encode 'AgeBucket' as numerical labels
    age_bucket_map = {
        'Child': 0,
        'Teen': 1,
        'YoungAdult': 2,
        'Adult': 3,
        'Senior': 4
    }
    df0['AgeBucket'] = df0['AgeBucket'].map(age_bucket_map)

    # Spending — fill NA with 0
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spending_cols:
        df0[col] = df0[col].fillna(0)

    df0['TotalSpending'] = df0[spending_cols].sum(axis=1)
    df0['NumServicesUsed'] = (df0[spending_cols] > 0).sum(axis=1)
    df0['HasLuxurySpending'] = (df0['Spa'] + df0['VRDeck']) > 0
    df0['IsSleepingWithExpenses'] = df0['CryoSleep'] & (df0['TotalSpending'] > 0)

    # Cabin processing
    df0['Cabin'] = df0['Cabin'].fillna("Unknown/0/Unknown")
    df0[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df0['Cabin'].str.split('/', expand=True)
    df0['Cabin_Num'] = pd.to_numeric(df0['Cabin_Num'], errors='coerce')
    df0.drop(columns=['Cabin'], inplace=True)


    df0 = pd.get_dummies(df0, columns=['Cabin_Deck', 'Cabin_Side'], dummy_na=True)

    # Group size
    df0['GroupID'] = df0['PassengerId'].apply(lambda x: x.split('_')[0])
    group_sizes = df0.groupby('GroupID')['PassengerId'].count().reset_index()
    group_sizes.rename(columns={'PassengerId': 'GroupSize'}, inplace=True)
    df0 = df0.merge(group_sizes, on='GroupID', how='left')

    # Drop columns
    df0.drop(columns=['PassengerId', 'Name'], inplace=True, errors='ignore')

    return df0



df = process_df(df)
test = process_df(test)


train_df, final_valid_df = train_test_split(
    df, test_size=0.1, stratify=df["Transported"], random_state=42
)

def train_and_evaluate_fold(train_idx, valid_idx, df, params):
    train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]
    model = ydf.GradientBoostedTreesLearner(**params).train(train_df)
    evaluation = model.evaluate(valid_df)
    return evaluation.accuracy

def objective(trial):   
    params = {
        "label": "Transported",
        "random_seed": 42,
        "num_trees": trial.suggest_int("num_trees", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 32),
        "shrinkage": trial.suggest_float("shrinkage", 0.001, 0.3, log=True),
        "min_examples": trial.suggest_int("min_examples", 2, 20),
        "split_axis": "AXIS_ALIGNED",
        "categorical_algorithm":trial.suggest_categorical("categorical_algorithm", ["CART", "RANDOM"]),
        "growing_strategy": trial.suggest_categorical("growing_strategy", ["LOCAL", "BEST_FIRST_GLOBAL"]),
        "sampling_method": "SELGB",
        "selective_gradient_boosting_ratio": trial.suggest_float("selective_gradient_boosting_ratio", 0.01, 0.5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "use_hessian_gain": True,  
          
    }
    
    sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = [
        train_and_evaluate_fold(train_idx, valid_idx, df, params)
        for train_idx, valid_idx in sgkf.split(df, df['Transported'])
    ] 

    return sum(fold_accuracies) / len(fold_accuracies)


# Create a study with pruning enabled
study = optuna.create_study(
    storage="sqlite:///study.db",
    direction="maximize",
    pruner=HyperbandPruner(),
    sampler=TPESampler(),
    load_if_exists=True
)

#study = optuna.create_study(storage="sqlite:///study.db", direction="maximize", load_if_exists=True)
study.optimize(objective, n_trials=2, n_jobs=12)

best_params = study.best_params
print("\n\nBest hyperparameters:", best_params)

# Evaluate final best model on hold-out validation
# best_model = ydf.GradientBoostedTreesLearner(label="Transported",**best_params).train(train_df)
templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
print(templates)
best_model = ydf.GradientBoostedTreesLearner(label="Transported",**templates["benchmark_rank1@1"]).train(train_df)
final_eval = best_model.evaluate(final_valid_df)
print(f"Final hold-out validation accuracy: {final_eval.accuracy}")

valid_probs = best_model.predict(final_valid_df)
true_labels = final_valid_df["Transported"]

thresholds = np.linspace(0.0, 1.0, 101)
best_thresh = 0.5
best_acc = 0

for t in thresholds:
    preds = valid_probs > t
    acc = accuracy_score(true_labels, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"✅ Best threshold: {best_thresh:.3f} with accuracy: {best_acc:.4f}")

# Load the sample submission template
sample_submission_df = pd.read_csv('../data/sample_submission.csv')

test_probs = best_model.predict(test)
sample_submission_df['Transported'] = test_probs > best_thresh

sample_submission_df.to_csv('../output/submission.csv', index=False)

