# -*- coding: utf-8 -*-
"""Export Moodle + Kaggle — CatBoost multi-seed + XGB blend, parametri optimizați."""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_CLF   = "final_result"
TARGET_REG   = "final_coursework_score"
OUTLIER_COL  = "total_clicks_early"
STACK_COL    = "_pred_score_reg"

# ── seeds ──────────────────────────────────────────────────────────────────────
SEEDS_MOODLE = (42, 137, 271)
SEEDS_KAGGLE = (42, 137, 271, 509, 1013)

# ── CatBoost — clasificare ─────────────────────────────────────────────────────
# depth mai mic (8→9) reduce overfit; learning_rate mai mic + mai multe iterații
# dau un model mai stabil pe setul de test public
CLF_KW = dict(
    depth             = 8,
    learning_rate     = 0.03,
    l2_leaf_reg       = 3.0,
    min_data_in_leaf  = 5,
    border_count      = 128,
    random_strength   = 0.5,
    bagging_temperature = 0.8,
)

# ── CatBoost — regresie ────────────────────────────────────────────────────────
# depth=8 în loc de 11 (era overfit sever); l2 mai mare; random_strength păstrat
REG_KW = dict(
    depth             = 8,
    learning_rate     = 0.025,
    l2_leaf_reg       = 4.0,
    min_data_in_leaf  = 6,
    random_strength   = 0.4,
    border_count      = 128,
)

# ── XGBoost — clasificare ──────────────────────────────────────────────────────
XGB_CLF_KW = dict(
    max_depth         = 7,
    learning_rate     = 0.04,
    subsample         = 0.85,
    colsample_bytree  = 0.80,
    min_child_weight  = 3.0,
    reg_lambda        = 2.5,
    reg_alpha         = 0.1,
)

# ── XGBoost — regresie ─────────────────────────────────────────────────────────
XGB_REG_KW = dict(
    max_depth         = 8,
    learning_rate     = 0.028,
    subsample         = 0.88,
    colsample_bytree  = 0.85,
    min_child_weight  = 3.0,
    reg_lambda        = 2.0,
    reg_alpha         = 0.05,
)

# ── scorer pentru regresie: clampare în intervalul realist al datelor ──────────
REG_MIN, REG_MAX = 0.0, 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Utilitare
# ──────────────────────────────────────────────────────────────────────────────

def bounds_iqr(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def mask_outliers(s, lo, hi):
    return (s < lo) | (s > hi)


def cats_cb(X: pd.DataFrame, cat: list) -> pd.DataFrame:
    o = X.copy()
    for c in cat:
        o[c] = o[c].astype("object").where(o[c].notna(), "__MISSING__").astype(str)
    return o


def cats_xgb(X: pd.DataFrame, cat: list) -> pd.DataFrame:
    o = cats_cb(X, cat)
    for c in cat:
        o[c] = o[c].astype("category")
    return o


def _align_cat_proba(cbc, X_cb, le: LabelEncoder) -> np.ndarray:
    """Aliniază probabilitățile CatBoost la ordinea claselor din LabelEncoder."""
    P = cbc.predict_proba(X_cb)
    C = np.asarray(cbc.classes_, dtype=object)
    out = np.zeros((P.shape[0], len(le.classes_)))
    for j, lab in enumerate(le.classes_):
        m = np.nonzero(C.astype(str) == str(lab))[0]
        if len(m):
            out[:, j] = P[:, int(m[0])]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pregătire date
# ──────────────────────────────────────────────────────────────────────────────

def prepare(base: Path):
    tr = pd.read_csv(base / "CB_OUALD_train.csv")
    mo = pd.read_csv(base / "CB_OUALD_test.csv")
    kg_path = base / "CB_private_test.csv"
    kg = pd.read_csv(kg_path) if kg_path.exists() else mo.copy()

    nf = [c for c in tr.select_dtypes(include=[np.number]).columns if c != TARGET_REG]
    cf = [c for c in tr.select_dtypes(exclude=[np.number]).columns if c != TARGET_CLF]
    feat = nf + cf

    # Outlier IQR estimat doar pe train
    lo, hi = bounds_iqr(tr[OUTLIER_COL].dropna())
    for df in (tr, mo, kg):
        m = mask_outliers(df[OUTLIER_COL], lo, hi)
        df.loc[m, OUTLIER_COL] = np.nan

    Xtr = cats_cb(tr[feat], cf)
    Xmo = cats_cb(mo[feat], cf)
    Xkg = cats_cb(kg[feat], cf)
    yc  = np.array([str(x) for x in tr[TARGET_CLF]], dtype=object)
    yr  = tr[TARGET_REG].astype(float).values

    return mo, kg, cf, Xtr, Xmo, Xkg, yc, yr


# ──────────────────────────────────────────────────────────────────────────────
# OOF regresie (stacking feature)
# ──────────────────────────────────────────────────────────────────────────────

def oof_reg(X: pd.DataFrame, y: np.ndarray, cf: list) -> np.ndarray:
    from catboost import CatBoostRegressor
    oof = np.zeros(len(y), dtype=float)
    kf  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_i, va_i) in enumerate(kf.split(X)):
        m = CatBoostRegressor(
            loss_function="RMSE",
            iterations=2500,
            early_stopping_rounds=200,
            random_seed=RANDOM_STATE + fold,
            allow_writing_files=False,
            verbose=False,
            **REG_KW,
        )
        m.fit(X.iloc[tr_i], y[tr_i],
              eval_set=(X.iloc[va_i], y[va_i]),
              cat_features=cf)
        oof[va_i] = m.predict(X.iloc[va_i])
    print(f"  OOF MSE: {mean_squared_error(y, oof):.4f}", flush=True)
    return oof


# ──────────────────────────────────────────────────────────────────────────────
# Predicții MOODLE (mai puțini seeds, suficient)
# ──────────────────────────────────────────────────────────────────────────────

def reg_preds_moodle(X, y, cf, target):
    from catboost import CatBoostRegressor
    preds = []
    for sd in SEEDS_MOODLE:
        tr, va, yr_tr, yr_va = train_test_split(X, y, test_size=0.12, random_state=int(sd))
        m = CatBoostRegressor(
            loss_function="RMSE", iterations=6000,
            early_stopping_rounds=300, random_seed=int(sd),
            allow_writing_files=False, verbose=False, **REG_KW,
        )
        m.fit(tr, yr_tr, eval_set=(va, yr_va), cat_features=cf)
        it = max(int(m.get_best_iteration() or 500), 200)
        mf = CatBoostRegressor(
            loss_function="RMSE", iterations=it,
            random_seed=int(sd), allow_writing_files=False, verbose=False, **REG_KW,
        )
        mf.fit(X, y, cat_features=cf)
        preds.append(np.clip(mf.predict(target), REG_MIN, REG_MAX))
    return np.mean(np.stack(preds, 0), 0)


def clf_preds_moodle(X, y, cf, target):
    from catboost import CatBoostClassifier
    P, classes = None, None
    for sd in SEEDS_MOODLE:
        tr, va, y_tr, y_va = train_test_split(X, y, test_size=0.12,
                                              random_state=int(sd), stratify=y)
        m = CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            iterations=5000, early_stopping_rounds=300,
            random_seed=int(sd), allow_writing_files=False, verbose=False, **CLF_KW,
        )
        m.fit(tr, y_tr, eval_set=(va, y_va), cat_features=cf)
        it = max(int(m.get_best_iteration() or 500), 200)
        mf = CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            iterations=it, random_seed=int(sd),
            allow_writing_files=False, verbose=False, **CLF_KW,
        )
        mf.fit(X, y, cat_features=cf)
        if classes is None:
            classes = np.asarray(mf.classes_, dtype=object)
        p = mf.predict_proba(target)
        P = p if P is None else P + p
    return classes[np.argmax(P / len(SEEDS_MOODLE), axis=1)]


# ──────────────────────────────────────────────────────────────────────────────
# Calibrare blend CatBoost + XGBoost pe validare internă
# ──────────────────────────────────────────────────────────────────────────────

def tune_kaggle_blend(X, yc, yr, cf):
    """Returnează (w_clf, w_reg) — ponderea CatBoost în blend."""
    w_clf, w_reg = 1.0, 1.0
    try:
        import xgboost as xgb
        from catboost import CatBoostClassifier, CatBoostRegressor

        tr, va, y_tr, y_va, yr_tr, yr_va = train_test_split(
            X, yc, yr, test_size=0.18,
            random_state=RANDOM_STATE, stratify=yc,
        )
        le = LabelEncoder()
        le.fit(y_tr.astype(str))
        tr_x, va_x = cats_xgb(tr, cf), cats_xgb(va, cf)

        # CatBoost
        cbc = CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            iterations=5000, early_stopping_rounds=300,
            random_seed=RANDOM_STATE, allow_writing_files=False, verbose=False, **CLF_KW,
        )
        cbc.fit(tr, y_tr, eval_set=(va, y_va), cat_features=cf)

        cbr = CatBoostRegressor(
            loss_function="RMSE", iterations=6000, early_stopping_rounds=350,
            random_seed=RANDOM_STATE, allow_writing_files=False, verbose=False, **REG_KW,
        )
        cbr.fit(tr, yr_tr, eval_set=(va, yr_va), cat_features=cf)

        # XGBoost
        xbc = xgb.XGBClassifier(
            objective="multi:softprob", num_class=len(le.classes_),
            n_estimators=4000, tree_method="hist", enable_categorical=True,
            random_state=RANDOM_STATE, n_jobs=-1,
            early_stopping_rounds=200, eval_metric="mlogloss", **XGB_CLF_KW,
        )
        xbc.fit(tr_x, le.transform(y_tr.astype(str)),
                eval_set=[(va_x, le.transform(y_va.astype(str)))], verbose=False)

        xbr = xgb.XGBRegressor(
            n_estimators=5000, tree_method="hist", enable_categorical=True,
            random_state=RANDOM_STATE, n_jobs=-1,
            early_stopping_rounds=250, eval_metric="rmse", **XGB_REG_KW,
        )
        xbr.fit(tr_x, yr_tr, eval_set=[(va_x, yr_va)], verbose=False)

        # Grid search blend — clasificare
        Pc = _align_cat_proba(cbc, va, le)
        Px = xbc.predict_proba(va_x)
        best_acc = -1.0
        for w in np.linspace(1.0, 0.10, 46):
            acc = accuracy_score(
                y_va,
                le.inverse_transform(np.argmax(w * Pc + (1 - w) * Px, axis=1)).astype(str),
            )
            if acc > best_acc:
                best_acc, w_clf = acc, float(w)

        # Grid search blend — regresie
        pv_c = np.clip(cbr.predict(va), REG_MIN, REG_MAX)
        pv_x = np.clip(xbr.predict(va_x), REG_MIN, REG_MAX)
        best_mse = float("inf")
        for w in np.linspace(1.0, 0.10, 46):
            mse = mean_squared_error(yr_va, w * pv_c + (1 - w) * pv_x)
            if mse < best_mse:
                best_mse, w_reg = mse, float(w)

        print(
            f"Blend calibrat: w_cat_clf={w_clf:.3f}  acc~{best_acc:.4f} | "
            f"w_cat_reg={w_reg:.3f}  MSE~{best_mse:.4f}",
            flush=True,
        )
    except Exception as e:
        print("tune_kaggle_blend skipped:", type(e).__name__, str(e)[:120], flush=True)
    return w_clf, w_reg


# ──────────────────────────────────────────────────────────────────────────────
# Predicții KAGGLE (mai mulți seeds + XGB blend opțional)
# ──────────────────────────────────────────────────────────────────────────────

def reg_preds_kaggle(X, y, cf, target, w_cat: float = 1.0):
    from catboost import CatBoostRegressor
    preds = []
    for sd in SEEDS_KAGGLE:
        tr, va, yr_tr, yr_va = train_test_split(X, y, test_size=0.12, random_state=int(sd))
        m = CatBoostRegressor(
            loss_function="RMSE", iterations=8000,
            early_stopping_rounds=350, random_seed=int(sd),
            allow_writing_files=False, verbose=False, **REG_KW,
        )
        m.fit(tr, yr_tr, eval_set=(va, yr_va), cat_features=cf)
        it = max(int(m.get_best_iteration() or 500), 250)
        mf = CatBoostRegressor(
            loss_function="RMSE", iterations=it,
            random_seed=int(sd), allow_writing_files=False, verbose=False, **REG_KW,
        )
        mf.fit(X, y, cat_features=cf)
        preds.append(np.clip(mf.predict(target), REG_MIN, REG_MAX))
    out = np.mean(np.stack(preds, 0), 0)

    if w_cat < 0.999:
        try:
            import xgboost as xgb
            xp = []
            Xx, Tx = cats_xgb(X, cf), cats_xgb(target, cf)
            for sd in SEEDS_KAGGLE[:3]:
                tr, va, yr_tr, yr_va = train_test_split(Xx, y, test_size=0.12, random_state=int(sd))
                mx = xgb.XGBRegressor(
                    n_estimators=7000, tree_method="hist", enable_categorical=True,
                    random_state=int(sd), n_jobs=-1,
                    early_stopping_rounds=250, eval_metric="rmse", **XGB_REG_KW,
                )
                mx.fit(tr, yr_tr, eval_set=[(va, yr_va)], verbose=False)
                itx = max(int(getattr(mx, "best_iteration", 500) or 500), 250)
                mxf = xgb.XGBRegressor(
                    n_estimators=itx, tree_method="hist", enable_categorical=True,
                    random_state=int(sd), n_jobs=-1, **XGB_REG_KW,
                )
                mxf.fit(Xx, y)
                xp.append(np.clip(mxf.predict(Tx), REG_MIN, REG_MAX))
            xpred = np.mean(np.stack(xp, 0), 0)
            out = np.clip(w_cat * out + (1.0 - w_cat) * xpred, REG_MIN, REG_MAX)
        except Exception as e:
            print("XGB reg kaggle failed:", e, flush=True)
    return out


def clf_preds_kaggle(X, y, cf, target, w_cat: float = 1.0):
    from catboost import CatBoostClassifier
    P, classes = None, None
    for sd in SEEDS_KAGGLE:
        tr, va, y_tr, y_va = train_test_split(
            X, y, test_size=0.12, random_state=int(sd), stratify=y,
        )
        m = CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            iterations=5000, early_stopping_rounds=300,
            random_seed=int(sd), allow_writing_files=False, verbose=False, **CLF_KW,
        )
        m.fit(tr, y_tr, eval_set=(va, y_va), cat_features=cf)
        it = max(int(m.get_best_iteration() or 500), 250)
        mf = CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            iterations=it, random_seed=int(sd),
            allow_writing_files=False, verbose=False, **CLF_KW,
        )
        mf.fit(X, y, cat_features=cf)
        if classes is None:
            classes = np.asarray(mf.classes_, dtype=object)
        p = mf.predict_proba(target)
        P = p if P is None else P + p
    P = P / len(SEEDS_KAGGLE)

    if w_cat < 0.999:
        try:
            import xgboost as xgb
            le = LabelEncoder()
            le.fit(y.astype(str))
            K  = len(le.classes_)
            Px = None
            Xx, Tx = cats_xgb(X, cf), cats_xgb(target, cf)
            for sd in SEEDS_KAGGLE[:3]:
                tr, va, y_tr, y_va = train_test_split(
                    Xx, y, test_size=0.12, random_state=int(sd), stratify=y,
                )
                mx = xgb.XGBClassifier(
                    objective="multi:softprob", num_class=K,
                    n_estimators=4000, tree_method="hist", enable_categorical=True,
                    random_state=int(sd), n_jobs=-1,
                    early_stopping_rounds=200, eval_metric="mlogloss", **XGB_CLF_KW,
                )
                mx.fit(tr, le.transform(y_tr.astype(str)),
                       eval_set=[(va, le.transform(y_va.astype(str)))], verbose=False)
                itx = max(int(getattr(mx, "best_iteration", 400) or 400), 200)
                mxf = xgb.XGBClassifier(
                    objective="multi:softprob", num_class=K,
                    n_estimators=itx, tree_method="hist", enable_categorical=True,
                    random_state=int(sd), n_jobs=-1, **XGB_CLF_KW,
                )
                mxf.fit(Xx, le.transform(y.astype(str)))
                p = mxf.predict_proba(Tx)
                Px = p if Px is None else Px + p
            Px = Px / 3.0

            # Aliniază CatBoost proba → ordinea LabelEncoder
            n = len(target)
            Pal = np.zeros((n, K))
            for j, lab in enumerate(le.classes_):
                m = np.nonzero(classes.astype(str) == str(lab))[0]
                if len(m):
                    Pal[:, j] = P[:, int(m[0])]

            P_blend = w_cat * Pal + (1.0 - w_cat) * Px
            return le.inverse_transform(np.argmax(P_blend, axis=1)).astype(str)
        except Exception as e:
            print("XGB clf kaggle failed:", e, flush=True)

    return classes[np.argmax(P, axis=1)].astype(str)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main(base=None):
    base = Path(base or Path(__file__).resolve().parent)
    mo, kg, cf, Xtr, Xmo, Xkg, yc, yr = prepare(base)
    print(f"Date incarcate: train={Xtr.shape}, test_moodle={Xmo.shape}, test_kaggle={Xkg.shape}", flush=True)

    # ── OOF feature pentru stacking ─────────────────────────────────────────
    print("OOF regresie (stacking feature)...", flush=True)
    oof = oof_reg(Xtr, yr, cf)

    # ── Predicții Moodle ────────────────────────────────────────────────────
    print("Moodle — regresie...", flush=True)
    pr_m = reg_preds_moodle(Xtr, yr, cf, Xmo)

    # ── Calibrare blend Kaggle ───────────────────────────────────────────────
    print("Calibrare blend Kaggle...", flush=True)
    w_clf, w_reg = tune_kaggle_blend(Xtr, yc, yr, cf)

    # ── Predicții Kaggle — regresie ──────────────────────────────────────────
    print("Kaggle — regresie...", flush=True)
    pr_k = reg_preds_kaggle(Xtr, yr, cf, Xkg, w_cat=w_reg)

    # ── Adaugăm stacking feature ─────────────────────────────────────────────
    Xtr_s = Xtr.copy(); Xtr_s[STACK_COL] = oof
    Xmo_s = Xmo.copy(); Xmo_s[STACK_COL] = pr_m
    Xkg_s = Xkg.copy(); Xkg_s[STACK_COL] = pr_k

    # ── Validare internă (raport) ────────────────────────────────────────────
    tr_s, va_s, y_tr, y_va, yr_tr, yr_va = train_test_split(
        Xtr_s, yc, yr, test_size=0.15,
        random_state=RANDOM_STATE, stratify=yc,
    )
    from catboost import CatBoostClassifier, CatBoostRegressor
    rv = CatBoostRegressor(
        loss_function="RMSE", iterations=6000, early_stopping_rounds=300,
        random_seed=RANDOM_STATE, allow_writing_files=False, verbose=False, **REG_KW,
    )
    rv.fit(tr_s.drop(columns=[STACK_COL]), yr_tr,
           eval_set=(va_s.drop(columns=[STACK_COL]), yr_va), cat_features=cf)
    cv = CatBoostClassifier(
        loss_function="MultiClass", auto_class_weights="Balanced",
        iterations=5000, early_stopping_rounds=300,
        random_seed=RANDOM_STATE, allow_writing_files=False, verbose=False, **CLF_KW,
    )
    cv.fit(tr_s, y_tr, eval_set=(va_s, y_va), cat_features=cf)
    print(f"  Validare internă — MSE: {mean_squared_error(yr_va, rv.predict(va_s.drop(columns=[STACK_COL]))):.4f} | "
          f"acc: {accuracy_score(y_va, cv.predict(va_s)):.4f}", flush=True)

    # ── Predicții finale Moodle + Kaggle ─────────────────────────────────────
    print("Moodle — clasificare...", flush=True)
    pcm = clf_preds_moodle(Xtr_s, yc, cf, Xmo_s)

    print("Kaggle — clasificare...", flush=True)
    pck = clf_preds_kaggle(Xtr_s, yc, cf, Xkg_s, w_cat=w_clf)

    # ── Export fișiere ───────────────────────────────────────────────────────
    pd.DataFrame({
        "final_result": pcm,
        "final_coursework_score": np.clip(pr_m, REG_MIN, REG_MAX),
    }).to_csv(base / "CB_OUALD_predictii_tema1.csv", index=False)

    kid = kg["id"] if "id" in kg.columns else pd.RangeIndex(1, len(kg) + 1)
    pd.DataFrame({
        "id": kid,
        "prediction": np.asarray(pck, dtype=object).astype(str),
    }).to_csv(base / "kaggle_TEMA1_CB_clasificare_2026.csv", index=False)

    pd.DataFrame({
        "id": kid,
        "prediction": np.clip(pr_k, REG_MIN, REG_MAX),
    }).to_csv(base / "kaggle_TEMA1_CB_regresie_2026.csv", index=False, float_format="%.8f")

    print(f"Export OK → {base}", flush=True)


if __name__ == "__main__":
    main()