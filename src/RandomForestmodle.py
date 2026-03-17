import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Read data
train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')


#data preprocessing
def preprocess(df):
    df = df.copy()

    # Drop useless column
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    # Split Cabin
    if 'Cabin' in df.columns:
        cabin_split = df['Cabin'].str.split('/', expand=True)
        df['Deck'] = cabin_split[0]
        df['CabinNum'] = cabin_split[1]
        df['Side'] = cabin_split[2]
        df = df.drop(columns=['Cabin'])
    # Fill age
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    # Construct new features
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # TotalSpend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing = [c for c in spend_cols if c in df.columns]
    df['TotalSpend'] = df[existing].sum(axis=1)

    return df


train = preprocess(train)
test = preprocess(test)

#lable
y = train['Transported']
X = train.drop(columns=['Transported'])

test_ids = test['PassengerId']
test_X = test.drop(columns=['PassengerId'])

#4. Classification of Feature Types
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#modle train
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", acc)


for col in X.columns:
    if col not in test_X.columns:
        test_X[col] = None

test_X = test_X[X.columns]


test_pred = model.predict(test_X)

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': test_pred
})


import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

output_dir = os.path.join(BASE_DIR, 'outputs', 'submissions')


os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'submission.csv')

submission.to_csv(output_path, index=False)

print(f" submission Saved to: {output_path}")