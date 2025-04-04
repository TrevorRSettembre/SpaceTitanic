import ydf
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split

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
    
    return df0

df = process_df(df)
test = process_df(test)

# Split dataset into train and validation
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

def objective(trial):
    num_trees = trial.suggest_int("num_trees", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_examples = trial.suggest_int("min_examples", 2, 10)
    categorical_algorithm = trial.suggest_categorical("categorical_algorithm", ["CART", "ONE_HOT", "RANDOM"])
    growing_strategy = trial.suggest_categorical("growing_strategy", ["BEST_FIRST_GLOBAL","LOCAL"])
    
    model = ydf.RandomForestLearner(
        label="Transported",
        num_trees=num_trees,
        max_depth=max_depth,
        min_examples=min_examples,
        categorical_algorithm=categorical_algorithm,
        growing_strategy=growing_strategy
    ).train(train_df)
    
    evaluation = model.evaluate(valid_df)
    return evaluation.accuracy  # Optuna will maximize this

# Run Optuna hyperparameter tuning
study = optuna.create_study(storage="sqlite:///study.db", direction="maximize", load_if_exists=True)
study.optimize(objective, n_trials=50, n_jobs=6)

# Best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model with best hyperparameters
best_model = ydf.RandomForestLearner(label="Transported", **best_params).train(df)
best_model.describe()
best_model.predict(test)

evaluation = best_model.evaluate(df)
print(f"Final test accuracy: {evaluation.accuracy}")
