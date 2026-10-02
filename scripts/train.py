from pathlib import Path
import json
import joblib
import sklearn
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, roc_auc_score
)

# 1. Load dữ liệu
bundle = load_breast_cancer(as_frame=True)
X, y = bundle.data, bundle.target
print("Shape:", X.shape)
print("Label counts:\n", y.value_counts())
print("Target names:", bundle.target_names)
print("Missing:", X.isna().sum().sum())

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3. Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )),
])

param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)
model = search.best_estimator_
print("Best params:", search.best_params_)

# 4. Đánh giá
pred = model.predict(X_test)
proba_malignant = model.predict_proba(X_test)[:, 0]

print("Confusion matrix [0=malignant, 1=benign]:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print(classification_report(
    y_test, pred, labels=[0, 1],
    target_names=["malignant", "benign"]
))
print("Sensitivity (malignant):",
      recall_score(y_test, pred, pos_label=0))
print("Precision (malignant):",
      precision_score(y_test, pred, pos_label=0))
print("ROC-AUC (malignant):",
      roc_auc_score((y_test == 0).astype(int), proba_malignant))

# 5. Lưu artifact + metadata
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

joblib.dump(model, ARTIFACT_DIR / "breast_cancer_svm.joblib")

metadata = {
    "model_name": "breast-cancer-svm-rbf",
    "model_version": "1.0.0",
    "feature_names": list(X.columns),
    "class_mapping": {"0": "malignant", "1": "benign"},
    "best_params": search.best_params_,
    "sklearn_version": sklearn.__version__,
    "warning": "Educational use only; not a medical diagnosis.",
}
(ARTIFACT_DIR / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Đã lưu artifact và metadata.")
