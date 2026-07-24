"""
Prevalence-sweep experiment: when do class-imbalance corrections succeed or fail?

Design:
- Construct datasets at controlled prevalence levels by stratified subsampling
  from the full BRFSS 2020 heart disease dataset (319,795 records).
- N = 20,000 per constructed dataset; 10 random seeds per prevalence.
- 60/20/20 stratified train/validation/test split.
- 3 model families x 6 strategies, evaluated on the untouched test split.
- Decision thresholds are ALWAYS selected on the validation split (never test).

Strategies:
  default          : model as-is, threshold 0.5
  weight           : class_weight='balanced', threshold 0.5
  undersample      : random undersample majority to 1:1, threshold 0.5
  smote            : SMOTE minority to 1:1, threshold 0.5
  threshold        : default model, Youden-J threshold chosen on validation
  weight+threshold : balanced model, Youden-J threshold chosen on validation
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

DATA = "data/heart_2020_cleaned.csv"
OUT = "results/results.csv"
PREVALENCES = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
N_TOTAL = 20000
SEEDS = list(range(10))

os.makedirs("results", exist_ok=True)

CAT = ["Race", "AgeCategory", "GenHealth", "Sex", "Smoking", "AlcoholDrinking",
       "Stroke", "DiffWalking", "Diabetic", "PhysicalActivity", "Asthma",
       "KidneyDisease", "SkinCancer"]

def load_full():
    df = pd.read_csv(DATA).dropna()
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
    return df

def make_dataset(df, prevalence, seed):
    """Stratified subsample at a controlled prevalence, fixed total N."""
    n_pos = int(round(N_TOTAL * prevalence))
    n_neg = N_TOTAL - n_pos
    rng = np.random.RandomState(seed)
    pos = df[df.HeartDisease == 1]
    neg = df[df.HeartDisease == 0]
    idx = np.concatenate([
        rng.choice(pos.index.values, size=n_pos, replace=False),
        rng.choice(neg.index.values, size=n_neg, replace=False),
    ])
    return df.loc[idx].reset_index(drop=True)

def encode(ds):
    X = ds.drop(columns=["HeartDisease"])
    y = ds["HeartDisease"].values
    Xn = X.drop(columns=CAT)
    Xc = pd.get_dummies(X[CAT], drop_first=True)
    return pd.concat([Xn, Xc], axis=1), y

def metrics_row(y_true, y_pred, proba, thr):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return dict(TN=tn, FP=fp, FN=fn, TP=tp,
                accuracy=(tp + tn) / len(y_true),
                sensitivity=sens, specificity=spec,
                balanced_acc=(sens + spec) / 2,
                precision=prec, f1_pos=f1,
                roc_auc=roc_auc_score(y_true, proba),
                pr_auc=average_precision_score(y_true, proba),
                threshold=thr)

def youden_threshold(y_val, proba_val):
    fpr, tpr, thr = roc_curve(y_val, proba_val)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

def get_model(name, weighted, seed):
    if name == "logreg":
        return LogisticRegression(max_iter=2000,
                                  class_weight="balanced" if weighted else None,
                                  random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                      class_weight="balanced" if weighted else None,
                                      random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            class_weight="balanced" if weighted else None, random_state=seed)
    raise ValueError(name)

def run_cell(df, prevalence, seed, writer):
    ds = make_dataset(df, prevalence, seed)
    X, y = encode(ds)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = sc.transform(X_tr), sc.transform(X_val), sc.transform(X_te)

    us = RandomUnderSampler(random_state=seed)
    X_us, y_us = us.fit_resample(X_tr, y_tr)
    sm = SMOTE(random_state=seed)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)

    for model_name in ["logreg", "rf", "hgb"]:
        t0 = time.time()
        m_def = get_model(model_name, False, seed).fit(X_tr, y_tr)
        m_wt = get_model(model_name, True, seed).fit(X_tr, y_tr)
        m_us = get_model(model_name, False, seed).fit(X_us, y_us)
        m_sm = get_model(model_name, False, seed).fit(X_sm, y_sm)

        p_def_te = m_def.predict_proba(X_te)[:, 1]
        p_wt_te = m_wt.predict_proba(X_te)[:, 1]
        p_us_te = m_us.predict_proba(X_te)[:, 1]
        p_sm_te = m_sm.predict_proba(X_te)[:, 1]
        p_def_val = m_def.predict_proba(X_val)[:, 1]
        p_wt_val = m_wt.predict_proba(X_val)[:, 1]

        thr_def = youden_threshold(y_val, p_def_val)
        thr_wt = youden_threshold(y_val, p_wt_val)

        rows = [
            ("default", (p_def_te >= 0.5).astype(int), p_def_te, 0.5),
            ("weight", (p_wt_te >= 0.5).astype(int), p_wt_te, 0.5),
            ("undersample", (p_us_te >= 0.5).astype(int), p_us_te, 0.5),
            ("smote", (p_sm_te >= 0.5).astype(int), p_sm_te, 0.5),
            ("threshold", (p_def_te >= thr_def).astype(int), p_def_te, thr_def),
            ("weight+threshold", (p_wt_te >= thr_wt).astype(int), p_wt_te, thr_wt),
        ]
        for strat, y_pred, proba, thr in rows:
            r = metrics_row(y_te, y_pred, proba, thr)
            r.update(prevalence=prevalence, seed=seed, model=model_name,
                     strategy=strat)
            writer.append(r)
        print(f"  {model_name} done in {time.time()-t0:.1f}s", flush=True)

def main(prevalences=None):
    df = load_full()
    print(f"Full dataset: {df.shape}, prevalence {df.HeartDisease.mean():.4f}", flush=True)
    prevalences = prevalences or PREVALENCES
    seeds = ([int(s) for s in os.environ["SEEDS_ARG"].split(",")]
             if "SEEDS_ARG" in os.environ else SEEDS)
    for prev in prevalences:
        rows = []
        for seed in seeds:
            print(f"prevalence={prev} seed={seed}", flush=True)
            run_cell(df, prev, seed, rows)
        chunk = pd.DataFrame(rows)
        header = not os.path.exists(OUT)
        chunk.to_csv(OUT, mode="a", header=header, index=False)
        print(f"prevalence {prev} written ({len(chunk)} rows)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    main(args if args else None)
