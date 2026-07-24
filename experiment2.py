"""
Extension experiments:
  A) Replication on two additional datasets (cardio: continuous-dominant;
     diabetes BRFSS2015: binary-dominant) at prevalences {2%, 10%, 35%},
     same 3 models x 6 strategies x 10 seeds protocol.
  B) Focal loss on the primary heart dataset, all 6 prevalences:
     - focal Logistic Regression (gamma=2, alpha-balanced), exact scipy fit
     - LightGBM: default / weighted / focal (custom objective) / tuned threshold
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import experiment as E  # reuse metrics_row, youden_threshold, get_model

OUT_EXT = "results/results_ext.csv"
OUT_FOCAL = "results/results_focal.csv"
N_TOTAL = 20000
SEEDS = list(range(10))

# ---------------- dataset loaders ----------------
def load_cardio():
    d = pd.read_csv("data/cardio_train.csv", sep=";").drop(columns=["id"])
    d = d[(d.ap_hi.between(60, 250)) & (d.ap_lo.between(30, 200)) & (d.ap_hi > d.ap_lo)]
    d["gender"] = (d["gender"] == 2).astype(int)
    y = d.pop("cardio")
    d["target"] = y
    return d.reset_index(drop=True)

def load_diabetes():
    d = pd.read_csv("data/diabetes_binary_BRFSS2015.csv")
    y = d.pop("Diabetes_binary").astype(int)
    d["target"] = y
    return d.reset_index(drop=True)

def make_dataset(df, prevalence, seed):
    n_pos = int(round(N_TOTAL * prevalence)); n_neg = N_TOTAL - n_pos
    rng = np.random.RandomState(seed)
    pos, neg = df[df.target == 1], df[df.target == 0]
    idx = np.concatenate([rng.choice(pos.index.values, n_pos, replace=False),
                          rng.choice(neg.index.values, n_neg, replace=False)])
    ds = df.loc[idx].reset_index(drop=True)
    return ds.drop(columns=["target"]), ds["target"].values

def splits(X, y, seed):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40,
                                                random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                random_state=seed, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_val), sc.transform(X_te), y_tr, y_val, y_te

# ---------------- A) replication datasets ----------------
def run_repl_cell(df, dataset, prevalence, seed, out):
    X, y = make_dataset(df, prevalence, seed)
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits(X, y, seed)
    X_us, y_us = RandomUnderSampler(random_state=seed).fit_resample(X_tr, y_tr)
    X_sm, y_sm = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
    for model_name in ["logreg", "rf", "hgb"]:
        m_def = E.get_model(model_name, False, seed).fit(X_tr, y_tr)
        m_wt  = E.get_model(model_name, True,  seed).fit(X_tr, y_tr)
        m_us  = E.get_model(model_name, False, seed).fit(X_us, y_us)
        m_sm  = E.get_model(model_name, False, seed).fit(X_sm, y_sm)
        p = {k: m.predict_proba(X_te)[:, 1] for k, m in
             dict(default=m_def, weight=m_wt, undersample=m_us, smote=m_sm).items()}
        thr_def = E.youden_threshold(y_val, m_def.predict_proba(X_val)[:, 1])
        thr_wt  = E.youden_threshold(y_val, m_wt.predict_proba(X_val)[:, 1])
        arms = [("default", p["default"], 0.5), ("weight", p["weight"], 0.5),
                ("undersample", p["undersample"], 0.5), ("smote", p["smote"], 0.5),
                ("threshold", p["default"], thr_def),
                ("weight+threshold", p["weight"], thr_wt)]
        for strat, proba, thr in arms:
            r = E.metrics_row(y_te, (proba >= thr).astype(int), proba, thr)
            r.update(dataset=dataset, prevalence=prevalence, seed=seed,
                     model=model_name, strategy=strat)
            out.append(r)

# ---------------- B) focal loss ----------------
GAMMA = 2.0

def focal_grad_z(y, p, alpha):
    """dLoss/dz per sample for focal loss with logits z, gamma=GAMMA."""
    g = np.empty_like(p)
    pos = y == 1
    pp = p[pos]; pn = p[~pos]
    logp = np.log(np.clip(pp, 1e-12, 1))
    log1p = np.log(np.clip(1 - pn, 1e-12, 1))
    g[pos] = alpha * (GAMMA * pp * (1 - pp) ** GAMMA * logp - (1 - pp) ** (GAMMA + 1))
    g[~pos] = (1 - alpha) * (-GAMMA * pn ** GAMMA * (1 - pn) * log1p + pn ** (GAMMA + 1))
    return g

def focal_loss_val(y, p, alpha):
    pt = np.where(y == 1, p, 1 - p)
    at = np.where(y == 1, alpha, 1 - alpha)
    return float(np.mean(-at * (1 - pt) ** GAMMA * np.log(np.clip(pt, 1e-12, 1))))

class FocalLR:
    def __init__(self, lam=1e-4, seed=0):
        self.lam = lam; self.seed = seed
    def fit(self, X, y):
        n, d = X.shape
        alpha = float((y == 0).mean())  # up-weight the rare positive class
        self.alpha = alpha
        def obj(wb):
            w, b = wb[:d], wb[d]
            p = expit(X @ w + b)
            loss = focal_loss_val(y, p, alpha) + self.lam * np.sum(w * w)
            gz = focal_grad_z(y, p, alpha) / n
            gw = X.T @ gz + 2 * self.lam * w
            gb = np.sum(gz)
            return loss, np.concatenate([gw, [gb]])
        res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=500))
        self.w, self.b = res.x[:d], res.x[d]
        return self
    def predict_proba(self, X):
        p = expit(X @ self.w + self.b)
        return np.column_stack([1 - p, p])

def gradcheck():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5); w = rng.randn(5) * 0.3; b = 0.1
    y = (rng.rand(200) < expit(X @ w)).astype(int)
    alpha = float((y == 0).mean())
    z = X @ w + b; p = expit(z)
    g_analytic = focal_grad_z(y, p, alpha)
    eps = 1e-6; g_num = np.empty_like(z)
    for i in range(len(z)):
        z1, z2 = z.copy(), z.copy(); z1[i] += eps; z2[i] -= eps
        l1 = focal_loss_val(y, expit(z1), alpha) * len(z)
        l2 = focal_loss_val(y, expit(z2), alpha) * len(z)
        g_num[i] = (l1 - l2) / (2 * eps)
    err = np.max(np.abs(g_analytic - g_num))
    assert err < 1e-5, f"gradcheck failed: {err}"
    return err

def lgb_focal_objective(alpha):
    def obj(y_true, y_pred):
        p = expit(y_pred)
        grad = focal_grad_z(y_true, p, alpha)
        eps = 1e-4
        gp = focal_grad_z(y_true, expit(y_pred + eps), alpha)
        gm = focal_grad_z(y_true, expit(y_pred - eps), alpha)
        hess = np.clip((gp - gm) / (2 * eps), 1e-6, None)
        return grad, hess
    return obj

def run_focal_cell(df, prevalence, seed, out):
    import lightgbm as lgb
    ds = E.make_dataset(df, prevalence, seed)
    X, y = E.encode(ds)
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits(X, y, seed)
    alpha = float((y_tr == 0).mean())

    # focal Logistic Regression
    flr = FocalLR(seed=seed).fit(X_tr, y_tr)
    p_te = flr.predict_proba(X_te)[:, 1]
    r = E.metrics_row(y_te, (p_te >= 0.5).astype(int), p_te, 0.5)
    r.update(prevalence=prevalence, seed=seed, model="logreg", strategy="focal")
    out.append(r)

    # LightGBM arms
    common = dict(n_estimators=200, learning_rate=0.1, num_leaves=31,
                  random_state=seed, verbose=-1, n_jobs=-1)
    m_def = lgb.LGBMClassifier(**common).fit(X_tr, y_tr)
    m_wt = lgb.LGBMClassifier(scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                              **common).fit(X_tr, y_tr)
    m_foc = lgb.LGBMRegressor(objective=lgb_focal_objective(alpha), **common).fit(X_tr, y_tr)

    p_def_te = m_def.predict_proba(X_te)[:, 1]
    p_wt_te = m_wt.predict_proba(X_te)[:, 1]
    p_foc_te = expit(m_foc.predict(X_te))
    thr_def = E.youden_threshold(y_val, m_def.predict_proba(X_val)[:, 1])

    arms = [("lgb_default", p_def_te, 0.5), ("lgb_weight", p_wt_te, 0.5),
            ("lgb_focal", p_foc_te, 0.5), ("lgb_threshold", p_def_te, thr_def)]
    for strat, proba, thr in arms:
        r = E.metrics_row(y_te, (proba >= thr).astype(int), proba, thr)
        r.update(prevalence=prevalence, seed=seed, model="lgb", strategy=strat)
        out.append(r)

# ---------------- CLI ----------------
def main():
    task = sys.argv[1]
    prevs = [float(a) for a in sys.argv[2:]]
    seeds = ([int(s) for s in os.environ["SEEDS_ARG"].split(",")]
             if "SEEDS_ARG" in os.environ else SEEDS)
    if task == "gradcheck":
        print("gradcheck max err:", gradcheck()); return
    if task in ("cardio", "diabetes"):
        df = load_cardio() if task == "cardio" else load_diabetes()
        print(f"{task}: {df.shape}, positives {int(df.target.sum())}", flush=True)
        for prev in prevs:
            rows = []
            for seed in seeds:
                t0 = time.time()
                run_repl_cell(df, task, prev, seed, rows)
                print(f"{task} p={prev} seed={seed} {time.time()-t0:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT_EXT, mode="a",
                                      header=not os.path.exists(OUT_EXT), index=False)
    elif task == "focal":
        df = E.load_full()
        for prev in prevs:
            rows = []
            for seed in seeds:
                t0 = time.time()
                run_focal_cell(df, prev, seed, rows)
                print(f"focal p={prev} seed={seed} {time.time()-t0:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT_FOCAL, mode="a",
                                      header=not os.path.exists(OUT_FOCAL), index=False)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
