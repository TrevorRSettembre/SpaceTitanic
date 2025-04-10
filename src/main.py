import ydf
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Load dataset
df = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

def process_df(df0: pd.DataFrame) -> pd.DataFrame:
    for col in ['CryoSleep', 'VIP']:
        df0[col] = df0[col].apply(lambda x: False if pd.isnull(x) else (str(x).lower() == 'true'))
    
    cat_default = {'HomePlanet': 'Unknown', 'Destination': 'Unknown', 'Name': 'Unknown'}
    df0.fillna(cat_default, inplace=True)

    df0['Age'] = df0['Age'].fillna(df['Age'].median())
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df0[col] = df0[col].fillna(0)
    
    df0['Cabin'] = df0['Cabin'].fillna("Unknown/0/Unknown")
    df0[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df0['Cabin'].str.split('/', expand=True)
    df0['Cabin_Num'] = pd.to_numeric(df0['Cabin_Num'], errors='coerce')
    df0.drop(columns=['Cabin'], inplace=True)

    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df0['TotalSpending'] = df0[spending_cols].sum(axis=1)

    df0['GroupID'] = df0['PassengerId'].apply(lambda x: x.split('_')[0])
    group_sizes = df0.groupby('GroupID')['PassengerId'].count().reset_index()
    group_sizes.rename(columns={'PassengerId': 'GroupSize'}, inplace=True)
    df0 = df0.merge(group_sizes, on='GroupID', how='left')

    drop_cols = [
        'PassengerId',  # ID column
        'Name',         # Too unique
        'Cabin_Num',    # Redundant after processing? Optional
        'FoodCourt',    # Weak correlation
        'ShoppingMall', # Weak correlation
    ]
    df0.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return df0 


df = process_df(df)
test = process_df(test)

# Split dataset into train and validation
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

def train_and_evaluate_fold(train_idx, valid_idx, df, params):
    # Split data into train and validation sets
    train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]

    # Train the model with the current fold's training set
    model = ydf.RandomForestLearner(**params).train(train_df)

    # Evaluate the model with the current fold's validation set
    evaluation = model.evaluate(valid_df)
    return evaluation.accuracy

def objective(trial):   
    params = {
        "label": "Transported",
        "num_trees": trial.suggest_int("num_trees", 500, 1500),
        "max_depth": trial.suggest_int("max_depth", 5, 64),
        "min_examples": trial.suggest_int("min_examples", 2, 50),
        "categorical_algorithm": trial.suggest_categorical("categorical_algorithm", ["CART", "ONE_HOT", "RANDOM"]),
        "growing_strategy": trial.suggest_categorical("growing_strategy", ["BEST_FIRST_GLOBAL","LOCAL"]),
        "winner_take_all": trial.suggest_categorical("winner_take_all", [True]),
        "split_axis": trial.suggest_categorical("split_axis", ["SPARSE_OBLIQUE"]),
        "sparse_oblique_weights": trial.suggest_categorical("sparse_oblique_weights", ["BINARY","CONTINUOUS"]),
        "sparse_oblique_normalization": trial.suggest_categorical("sparse_oblique_normalization", ["NONE", "STANDARD_DEVIATION", "MIN_MAX"]),
        "sparse_oblique_num_projections_exponent": trial.suggest_float("sparse_oblique_num_projections_exponent", 0.25,2),
        "num_candidate_attributes_ratio": trial.suggest_float("num_candidate_attributes_ratio", 0.01, 1.0),   
        "compute_oob_performances": trial.suggest_categorical("compute_oob_performances", [True]),
    }

    # Initialize StratifiedKFold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = Parallel(n_jobs=12)(
    delayed(train_and_evaluate_fold)(train_idx, valid_idx, df, params)
    for train_idx, valid_idx in skf.split(df, df['Transported'])
    )

    # Return the average accuracy across all K folds
    return sum(fold_accuracies) / len(fold_accuracies)

sampler = TPESampler(
    n_startup_trials=15,  # More random trials to explore the space initially
    seed=42               # Ensuring reproducibility
)

# Add pruner to stop underperforming trials early
pruner = MedianPruner(
    n_startup_trials=10,   # Number of trials to run before pruning starts
    n_warmup_steps=5       # Give the model a few trials to stabilize before pruning
)

# Create the study with pruner integrated
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,  # This is where the pruner gets added
    storage="sqlite:///study.db",  # Database for storing study results
    load_if_exists=True
)
study.optimize(objective, n_trials=25, n_jobs=12)

# Best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model with best hyperparameters
best_model = ydf.RandomForestLearner(label="Transported", **best_params).train(df)
best_model.describe()
best_model.predict(test)

evaluation = best_model.evaluate(df)
print(f"Final test accuracy: {evaluation.accuracy}")

# # Predict on test set
# predictions = best_model.predict(test)

# # Convert predictions to True/False
# predicted_labels = predictions.to_numpy().astype(bool)

# # Load the sample submission to get the correct order and format
# sample_submission = pd.read_csv('/mnt/data/sample_submission(1).csv')

# # Ensure the IDs match and build the final submission DataFrame
# submission = pd.DataFrame({
#     "PassengerId": test["PassengerId"],
#     "Transported": predicted_labels
# })

# # Sort to match the sample submission order if needed
# submission = submission.set_index("PassengerId").loc[sample_submission["PassengerId"]].reset_index()

# # Save to CSV
# submission.to_csv("my_submission.csv", index=False)

# print("Submission file 'my_submission.csv' created!")


