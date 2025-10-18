import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os
data = pd.read_csv("data/titanic.csv")
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
X = data.drop(columns=["Survived"])
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)
script_dir = os.getcwd()
model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "titanic_survival_predictor_pipeline.pkl")
joblib.dump(model, model_path)
print(" ✅ Model trained and saved successfully at:", model_path)
print(f" Model Accuracy on Test Set: {model.score(X_test, y_test) * 100:.2f}%")