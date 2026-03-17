import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Optional: silence future warning
pd.set_option('future.no_silent_downcasting', True)

# Read data
train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')


# Preprocess
def preprocess(df):
    df = df.copy()

    # Drop Name
    df = df.drop(columns=['Name'], errors='ignore')

    # Group features
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    df = df.drop(columns=['Group'])

    # Cabin split
    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin[0]
    df['CabinNum'] = pd.to_numeric(cabin[1], errors='coerce')
    df['Side'] = cabin[2]
    df = df.drop(columns=['Cabin'])

    # Fill CabinNum
    df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())

    # CabinNum bin
    df['CabinNumBin'] = pd.cut(df['CabinNum'], bins=6)

    # VIP / CryoSleep
    for col in ['VIP', 'CryoSleep']:
        df[col] = df[col].astype('object')
        df[col] = df[col].fillna(False)
        df[col] = df[col].astype(bool).astype(int)

    # Numeric columns
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # TotalSpend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)

    # Strong features
    df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)
    df['Cryo_NoSpend'] = df['CryoSleep'] * df['NoSpend']

    # Age bin
    df['AgeBin'] = pd.cut(df['Age'],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])


    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeBin', 'CabinNumBin']

    for col in cat_cols:
        df[col] = df[col].astype('object')
        df[col] = df[col].fillna("Unknown")


    for col in df.columns:
        if col not in cat_cols + ['PassengerId']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 再补一次数值缺失
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


train = preprocess(train)
test = preprocess(test)

# Label
y = train['Transported']
X = train.drop(columns=['Transported'])

test_ids = test['PassengerId']
test_X = test.drop(columns=['PassengerId'])


cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeBin', 'CabinNumBin']
num_cols = [col for col in X.columns if col not in cat_cols]

# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Align test columns
for col in X.columns:
    if col not in test_X.columns:
        test_X[col] = "Unknown" if col in cat_cols else 0

test_X = test_X[X.columns]

# Predict
test_pred = model.predict(test_X)

# Save
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': test_pred
})

output_path = '../outputs/submissions_RF_.csv'
submission.to_csv(output_path, index=False)

print("Saved to:", output_path)