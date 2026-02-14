import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "obesity_data.csv")

print("Loading dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run the download step.")

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────
print("Preprocessing...")

# Separate features and target
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Identify Categorical and Numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

# Encode Target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
target_names = target_encoder.classes_.tolist()
print(f"Target Classes: {target_names}")

# Split Data FIRST (Vital for correct evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Define Preprocessor (OneHot for Cats, Standard for Nums)
# handle_unknown='ignore' creates all zeros for unknown categories in test set
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    verbose_feature_names_out=False
)

# Fit on TRAIN, Transform TRAIN and TEST
print("fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out().tolist()

# Save artifacts
joblib.dump(preprocessor, os.path.join(BASE_DIR, "preprocessor.pkl"))
joblib.dump(target_encoder, os.path.join(BASE_DIR, "target_encoder.pkl"))
joblib.dump(target_names, os.path.join(BASE_DIR, "target_names.pkl"))
joblib.dump(feature_names, os.path.join(BASE_DIR, "feature_names.pkl")) # This is OHE names
joblib.dump(X.columns.tolist(), os.path.join(BASE_DIR, "input_features.pkl")) # Original input features for app validation
joblib.dump(cat_cols, os.path.join(BASE_DIR, "cat_cols.pkl"))
joblib.dump(num_cols, os.path.join(BASE_DIR, "num_cols.pkl"))

# Save test set for App Demo (Raw data, not processed, so app can demo the pipeline)
test_df = X_test.copy()
test_df["NObesity"] = y_test
test_df["NObesity_Label"] = target_encoder.inverse_transform(y_test)
test_df.to_csv(os.path.join(BASE_DIR, "..", "data", "test_data.csv"), index=False)


# ──────────────────────────────────────────────
# 3. SMOTE (Training Data Only)
# ──────────────────────────────────────────────
print("Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print(f"Original train shape: {X_train_processed.shape}")
print(f"Resampled train shape: {X_train_resampled.shape}")


# ──────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ──────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs", multi_class="multinomial", penalty="l2"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=15, min_samples_leaf=5
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7, weights="distance"
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, max_depth=20, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=len(target_names),
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1
    ),
}

# ──────────────────────────────────────────────
# 5. TRAIN & EVALUATE
# ──────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on processed Test set
    y_pred = model.predict(X_test_processed)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_processed)
    else:
        y_prob = None 

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    
    # AUC Note: Optimized for One-vs-Rest, may be optimistic due to SMOTE on train, but standard calculation on Test.
    if y_prob is not None:
         try:
            metrics["AUC"] = round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"), 4)
         except ValueError:
            metrics["AUC"] = 0.0
    else:
        metrics["AUC"] = 0.0

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    results[name] = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, os.path.join(BASE_DIR, f"{safe_name}.pkl"))

    print(f"{name}: Acc={metrics['Accuracy']}, F1={metrics['F1']}")

# ──────────────────────────────────────────────
# 6. SAVE RESULTS
# ──────────────────────────────────────────────
results_path = os.path.join(BASE_DIR, "metrics_results.json")

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def deep_convert(d):
    if isinstance(d, dict):
        return {k: deep_convert(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [deep_convert(i) for i in d]
    else:
        return convert_numpy(d)

results = deep_convert(results)

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("COMPARISON TABLE")
print(f"{'='*70}")
header = f"{'Model':<22} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}"
print(header)
print("-" * len(header))
for name, data in results.items():
    m = data["metrics"]
    print(
        f"{name:<22} {m['Accuracy']:>10.4f} {m['AUC']:>10.4f} "
        f"{m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f} {m['MCC']:>10.4f}"
    )
print(f"{'='*70}")
print(f"\nAll models saved to: {BASE_DIR}/")
