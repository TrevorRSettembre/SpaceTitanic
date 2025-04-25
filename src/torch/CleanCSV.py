import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
import os

TRAIN_FEATURES_PATH = "saved_features/feature_columns.npy"
os.makedirs("saved_features", exist_ok=True)

def process_df(df0: pd.DataFrame, is_train=True, scaler=None, target_column=None):
    df = df0.copy()

    # Boolean conversions
    for col in ['CryoSleep', 'VIP']:
        df[col] = (df[col].astype(str).str.lower() == 'true').fillna(False).astype(int)

    # Fill categorical with default
    cat_defaults = {'HomePlanet': 'Unknown', 'Destination': 'Unknown', 'Name': 'Unknown'}
    for col, val in cat_defaults.items():
        df[col] = df[col].fillna(val)

    # One-hot encode HomePlanet & Destination
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'], dummy_na=True)

    # Cabin splits
    df['Cabin'] = df['Cabin'].fillna("Unknown/0/Unknown")
    df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')
    df.drop(columns=['Cabin'], inplace=True)
    df = pd.get_dummies(df, columns=['Cabin_Deck', 'Cabin_Side'], dummy_na=True)

    # Spending features
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spend_cols:
        df[col] = df[col].fillna(0)

    df['TotalSpending'] = df[spend_cols].sum(axis=1)
    df['NumServicesUsed'] = (df[spend_cols] > 0).sum(axis=1)
    df['SpendingPerService'] = df['TotalSpending'] / df['NumServicesUsed'].replace(0, 1)
    df['HasLuxurySpending'] = ((df['Spa'] + df['VRDeck']) > 0).astype(int)
    df['IsSleepingWithExpenses'] = ((df['CryoSleep'] == 1) & (df['TotalSpending'] > 0)).astype(int)

    # PCA on spending columns
    if is_train:
        pca = PCA(n_components=2)
        spend_pca = pca.fit_transform(df[spend_cols])
        np.save("saved_features/pca_components.npy", pca.components_)
    else:
        pca = PCA(n_components=2)
        pca.components_ = np.load("saved_features/pca_components.npy")
        spend_pca = pca.transform(df[spend_cols])

    df['SpendingPC1'] = spend_pca[:, 0]
    df['SpendingPC2'] = spend_pca[:, 1]

    # Age processing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    df['AgeBucket'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    df['AgeBucket'] = df['AgeBucket'].map({k: v for v, k in enumerate(age_labels)})

    # Group features
    df['GroupID'] = df['PassengerId'].str.split('_').str[0]
    df['PassengerOrder'] = df['PassengerId'].str.split('_').str[1].astype(int)
    df['PassengerOrderNorm'] = df['PassengerOrder'] / df['PassengerOrder'].max()
    group_sizes = df.groupby('GroupID')['PassengerId'].transform('count')
    df['GroupSize'] = group_sizes
    df['IsAlone'] = (group_sizes == 1).astype(int)
    df['LargeGroup'] = (group_sizes > 3).astype(int)

    # Name features
    df['Name'] = df['Name'].fillna('Unknown Unknown')
    df['NameLength'] = df['Name'].apply(len)
    df['FirstName'] = df['Name'].apply(lambda x: x.split()[0])
    df['LastName'] = df['Name'].apply(lambda x: x.split()[-1])
    df['NameLengthBucket'] = pd.qcut(df['NameLength'], q=4, labels=False)
    df['NameHash'] = df['Name'].apply(lambda x: hash(x) % 1000)
    last_name_freq = df['LastName'].value_counts()
    df['LastNameFreq'] = df['LastName'].map(last_name_freq)
    df['RareLastName'] = (df['LastNameFreq'] == 1).astype(int)

    # Interaction terms
    df['CryoSleepWithSpending'] = df['CryoSleep'] * df['TotalSpending']
    df['VIPWithAge'] = df['VIP'] * df['Age']
    df['AgeTimesSpending'] = df['Age'] * df['TotalSpending']
    df['ServicesPerGroup'] = df['NumServicesUsed'] / df['GroupSize'].replace(0, 1)
    df['LuxuryRatio'] = (df['Spa'] + df['VRDeck']) / df['TotalSpending'].replace(0, 1)
    df['Spending_Age'] = df['Age'] * df['TotalSpending']
    df['AgeSquared'] = df['Age'] ** 2

    # Advanced features
    df['Cabin_Num_Normalized'] = df['Cabin_Num'] / df['Cabin_Num'].max()
    df['Cabin_Num_Bucket'] = pd.qcut(df['Cabin_Num'].fillna(-1), q=5, duplicates='drop', labels=False)
    df['Cabin_Num_IsNaN'] = df['Cabin_Num'].isna().astype(int)
    df['SpendingEntropy'] = df[spend_cols].apply(lambda row: entropy(row + 1e-6), axis=1)
    df['Cabin_Group'] = df['Cabin_Deck_nan'] * 100 + df['Cabin_Side_nan']
    df['CabinGroupSize'] = df.groupby('Cabin_Group')['Cabin_Num'].transform('count')
    df['LikelyFamily'] = ((df['LastNameFreq'] > 1) & (df['GroupSize'] > 1)).astype(int)
    df['VIP_Cryo_Conflict'] = ((df['VIP'] == 1) & (df['CryoSleep'] == 1)).astype(int)
    df['IsSpenderOutlier'] = (df['TotalSpending'] > df['TotalSpending'].quantile(0.99)).astype(int)

    for col in spend_cols + ['Age', 'TotalSpending']:
        df[f'{col}_ZScore'] = (df[col] - df[col].mean()) / df[col].std()
        df[f'{col}_Outlier'] = (np.abs(df[f'{col}_ZScore']) > 2).astype(int)

    df.drop(columns=['PassengerId', 'Name', 'FirstName', 'LastName', 'GroupID'], inplace=True)

    if target_column and target_column in df.columns:
        y = df[target_column].astype(int).values
        df.drop(columns=[target_column], inplace=True)
    else:
        y = None

    if is_train:
        feature_columns = df.columns.tolist()
        np.save(TRAIN_FEATURES_PATH, feature_columns)
    else:
        feature_columns = np.load(TRAIN_FEATURES_PATH, allow_pickle=True)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    if is_train:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
    else:
        df[df.columns] = scaler.transform(df[df.columns])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, y, scaler


class CleanCSV(Dataset):
    def __init__(self, csv_path, target_column=None, output_path="cleaned_data.csv", is_train=True, scaler=None):
        raw_df = pd.read_csv(csv_path)
        self.passenger_ids = raw_df['PassengerId'].values if 'PassengerId' in raw_df.columns else None
        processed_df, y, self.scaler = process_df(raw_df, is_train=is_train, scaler=scaler, target_column=target_column)
        self.X = torch.tensor(processed_df.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.input_size = self.X.shape[1]
        processed_df.to_csv(output_path, index=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            return x, self.y[idx]
        elif self.passenger_ids is not None:
            return x, self.passenger_ids[idx]
        else:
            return x