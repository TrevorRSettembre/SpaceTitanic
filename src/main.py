import ydf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

def process_df(df0: pd.DataFrame) -> pd.DataFrame:
    for col in ['CryoSleep', 'VIP']:
        df0[col] = df0[col].apply(lambda x: False if pd.isnull(x) else (str(x).lower()== 'true'))
    
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


print("Full train dataset shape is {}".format(df.shape))
templates = ydf.RandomForestLearner.hyperparameter_templates()

hyperparams = {
    "baseline": {
        "num_trees": 1000,
        "max_depth": 25,
        "min_examples": 3,
        "categorical_algorithm": "CART",
        "growing_strategy": "BEST_FIRST_GLOBAL"
    },
    "template": templates["benchmark_rank1v1"]
}

modelT = ydf.RandomForestLearner(label="Transported", **hyperparams["baseline"]).train(df)
modelT.describe()
modelT.predict(test)


evaluationT = modelT.evaluate(df)

# Query individual evaluation metrics
print(f"test accuracy: {evaluationT.accuracy}")

model = ydf.RandomForestLearner(label="Transported", **hyperparams["template"]).train(df)
model.describe()
model.predict(test)

evaluation = model.evaluate(df)

# Query individual evaluation metrics
print(f"test accuracy: {evaluation.accuracy}")

 