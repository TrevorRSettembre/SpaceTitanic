import ydf
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold


df = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


def process_df(df0: pd.DataFrame) -> pd.DataFrame:
    df0 = df0.copy()

    # Boolean columns — fill NA with False and cast to bool
    for col in ['CryoSleep', 'VIP']:
        df0[col] = df0[col].apply(lambda x: str(x).lower() == 'true' if pd.notnull(x) else False).astype(bool)

    # Categorical columns — fill with 'Unknown'
    cat_default = {'HomePlanet': 'Unknown', 'Destination': 'Unknown', 'Name': 'Unknown'}
    df0.fillna(cat_default, inplace=True)

      # Age — fill with median and bucket
    df0['Age'] = df0['Age'].fillna(df['Age'].median())
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
        "num_trees": trial.suggest_int("num_trees", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "shrinkage": trial.suggest_float("shrinkage", 0.01, 0.3, log=True),
        "min_examples": trial.suggest_int("min_examples", 2, 20),
        "categorical_algorithm": trial.suggest_categorical("categorical_algorithm", ["CART", "RANDOM", "ONE_HOT"]),
        "split_axis": trial.suggest_categorical("split_axis", ["AXIS_ALIGNED", "SPARSE_OBLIQUE"]),
        "growing_strategy": "BEST_FIRST_GLOBAL",
        "sampling_method": trial.suggest_categorical("sampling_method", ["NONE", "RANDOM"]),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
       # "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
       # "loss": trial.suggest_categorical("loss", ["DEFAULT", "BINOMIAL_LOG_LIKELIHOOD", "BINARY_FOCAL_LOSS"]),
    }
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = Parallel(n_jobs=4)(
        delayed(train_and_evaluate_fold)(train_idx, valid_idx, df, params)
        for train_idx, valid_idx in sgkf.split(df, df['Transported'], groups=df['GroupID'])
    )

    return sum(fold_accuracies) / len(fold_accuracies)

study = optuna.create_study(storage="sqlite:///study.db", direction="maximize", load_if_exists=True)
study.optimize(objective, n_trials=25, n_jobs=4)

best_params = study.best_params
print("\n\nBest hyperparameters:", best_params)

# Evaluate final best model on hold-out validation
best_model = ydf.RandomForestLearner(label="Transported",**best_params).train(train_df)
final_eval = best_model.evaluate(final_valid_df)
print(f"Final hold-out validation accuracy: {final_eval.accuracy}")

# (Optional) Retrain on full dataset for final submission
final_model = ydf.RandomForestLearner(label="Transported",**best_params).train(df)
final_model.describe()

# Predict on actual test data (Kaggle submission)
predictions = final_model.predict(test)

# Optional: Evaluate on full training set (for curiosity only)
train_eval = final_model.evaluate(df)
print(f"Training set accuracy (for reference): {train_eval.accuracy}")

# Convert YDF predictions to a list of booleans
n_predictions = [pred >= 0.5 for pred in predictions]

# Load the sample submission template
sample_submission_df = pd.read_csv('../data/sample_submission.csv')

# Assign your predictions
sample_submission_df['Transported'] = n_predictions

# Save to CSV
sample_submission_df.to_csv('../ouput/submission.csv', index=False)

# Display a preview
print(sample_submission_df.head())
