# ==========================================
# PHASE 2 - ADVANCED MODELING
# Small Dataset Optimized Version
# ==========================================

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# ------------------------------------------
# 1. Load Data
# ------------------------------------------

df = pd.read_csv("data/cleaned_covid19.csv")

df["outcome"] = df["outcome"].astype(str).str.lower()

df = df[df["outcome"].isin(["discharged", "recovered", "deceased", "death"])].copy()

df["outcome"] = df["outcome"].replace({
    "discharged": 0,
    "recovered": 0,
    "deceased": 1,
    "death": 1
})

print("Class distribution:")
print(df["outcome"].value_counts())

# ------------------------------------------
# 2. Features
# ------------------------------------------

features = [
    "age",
    "Fever",
    "Cough",
    "Fatigue",
    "Sore throat",
    "Runny nose",
    "Shortness of breath",
    "Vomiting"
]

X = df[features].copy()
y = df["outcome"]

X["age"] = X["age"].fillna(X["age"].median())

# ------------------------------------------
# 3. Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------
# 4. Cross Validation Strategy
# ------------------------------------------

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ==========================================
# 5. MLflow Tracking
# ==========================================

mlflow.set_experiment("COVID_Small_Dataset_Experiment")

def evaluate_model(name, model):
    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print(f"\n===== {name} =====")
        print("CV F1 Score:", scores.mean())
        print(classification_report(y_test, y_pred))

        mlflow.log_param("model", name)
        mlflow.log_metric("cv_f1_score", scores.mean())

        mlflow.sklearn.log_model(pipeline, name)

# ==========================================
# 6. Models
# ==========================================

# Logistic Regression (baseline)
evaluate_model(
    "Logistic_Regression",
    LogisticRegression(class_weight="balanced", max_iter=1000)
)

# Random Forest
evaluate_model(
    "Random_Forest",
    RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
)

# XGBoost
evaluate_model(
    "XGBoost",
    XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=5,  # important for imbalance
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
)

print("\nALL MODELS TRAINED SUCCESSFULLY!")
