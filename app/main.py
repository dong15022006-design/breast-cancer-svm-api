from pathlib import Path
import json
import joblib
import numpy as np
import secrets
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts/breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts/metadata.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

FEATURE_NAMES = metadata["feature_names"]

# ⚠️ ĐỔI USERNAME / PASSWORD Ở ĐÂY
USERNAME = "dong2006"
PASSWORD = "dong123"

security = HTTPBasic()


def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
):
    """Kiểm tra username/password — dùng compare_digest chống timing attack."""
    ok_user = secrets.compare_digest(credentials.username, USERNAME)
    ok_pass = secrets.compare_digest(credentials.password, PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai username hoặc password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only",
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Exactly 30 named numeric features"
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str


def build_vector(payload: PredictionRequest) -> np.ndarray:
    feats = payload.features
    missing = [f for f in FEATURE_NAMES if f not in feats]
    extra = [k for k in feats if k not in FEATURE_NAMES]
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing": missing, "extra": extra},
        )
    values = []
    for f in FEATURE_NAMES:
        v = feats[f]
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid value for feature '{f}': {v}",
            )
        values.append(float(v))
    return np.array(values, dtype=float).reshape(1, -1)


# ⚠️ Tất cả @app.get/post PHẢI đứng TRƯỚC app.mount

@app.get("/api")
def api_root(user: str = Depends(verify_credentials)):
    return {
        "service": "Breast Cancer SVM API",
        "user": user,
        "docs": "/docs",
        "warning": metadata["warning"],
    }


@app.get("/health")
def health():
    # Health check KHÔNG cần auth — để Render ping
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"],
    }


@app.get("/metadata")
def get_metadata(user: str = Depends(verify_credentials)):
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    user: str = Depends(verify_credentials),
):
    x = build_vector(payload)
    predicted_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    classes = list(model.named_steps["svc"].classes_)
    p = {int(c): float(v) for c, v in zip(classes, probabilities)}

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][str(predicted_class)],
        probability_malignant=p[0],
        probability_benign=p[1],
        model_version=metadata["model_version"],
        warning=metadata["warning"],
    )


# ⚠️ MOUNT PHẢI ĐỨNG CUỐI CÙNG
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
