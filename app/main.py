from pathlib import Path
import json
import joblib
import numpy as np
import hashlib
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Response, Cookie
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts/breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts/metadata.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
FEATURE_NAMES = metadata["feature_names"]

# ⚠️ ĐỔI USERNAME / PASSWORD / SECRET_KEY
USERNAME = "dong2006"
PASSWORD = "dong123"
SECRET_KEY = "doi-secret-key-nay-di-abc123xyz-2026"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only",
)


# ============ AUTH HELPERS ============

def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def require_auth(access_token: str = Cookie(None)):
    """Dependency cho API — yêu cầu cookie token hợp lệ."""
    if not access_token:
        raise HTTPException(401, "Chưa đăng nhập")
    user = verify_token(access_token)
    if not user:
        raise HTTPException(401, "Token hết hạn hoặc không hợp lệ")
    return user


# ============ MODELS ============

class LoginRequest(BaseModel):
    username: str
    password: str


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Exactly 30 features")


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
        raise HTTPException(422, detail={"missing": missing, "extra": extra})
    values = []
    for f in FEATURE_NAMES:
        v = feats[f]
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            raise HTTPException(422, detail=f"Invalid value for '{f}': {v}")
        values.append(float(v))
    return np.array(values, dtype=float).reshape(1, -1)


# ============ AUTH ENDPOINTS ============

@app.post("/login")
def login(data: LoginRequest, response: Response):
    """Đăng nhập, set cookie HttpOnly."""
    if data.username != USERNAME or data.password != PASSWORD:
        raise HTTPException(401, "Sai username hoặc password")
    token = create_token(USERNAME)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=TOKEN_EXPIRE_HOURS * 3600,
        samesite="lax",
        secure=True,   # BẮT BUỘC HTTPS
    )
    return {"ok": True, "user": USERNAME}


@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"ok": True}


@app.get("/me")
def me(user: str = Depends(require_auth)):
    return {"user": user}


# ============ API ENDPOINTS ============

@app.get("/api")
def api_root(user: str = Depends(require_auth)):
    return {
        "service": "Breast Cancer SVM API",
        "user": user,
        "docs": "/docs",
        "warning": metadata["warning"],
    }


@app.get("/health")
def health():
    # Không cần auth — cho Render ping
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"],
    }


@app.get("/metadata")
def get_metadata(user: str = Depends(require_auth)):
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, user: str = Depends(require_auth)):
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


# ============ PAGE ENDPOINTS (phải đứng TRƯỚC mount) ============

@app.get("/login")
def login_page():
    """Trả file login.html."""
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/")
def index_page(access_token: str = Cookie(None)):
    """Kiểm tra cookie, chưa login → redirect /login."""
    user = verify_token(access_token) if access_token else None
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(STATIC_DIR / "index.html")


# ============ STATIC FILES (phải ĐỨNG CUỐI) ============

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")
