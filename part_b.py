"""Part B: RFM, K-means segments, PCA, lapse model. Run: python part_b.py (after part_a.py)"""
import json, os
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")
comp = orders[orders.status == "completed"].copy()

# ============================ Task 1: RFM ============================
SNAP = pd.Timestamp("2025-10-01 00:00:00")
rfm = comp.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum")).reset_index()
rfm["recency"] = (SNAP - rfm["last"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print("=== RFM:", len(rfm), "customers | recency median/min/max", rfm.recency.median(), rfm.recency.min(), rfm.recency.max(),
      "| freq mean/max", round(rfm.frequency.mean(), 2), rfm.frequency.max(),
      "| monetary mean/max", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2))

# ============================ Task 2: K-means ============================
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows); print("\n=== K-means table"); print(ktab.to_string(index=False))
K = 4   # chosen: inertia drops sharply up to K=4 (elbow), silhouette peaks at K=4, four segments are actionable
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ============================ Task 3: profile, name, PCA ============================
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\n=== segment profile"); print(prof)
premium_id = prof["monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["frequency"].idxmax()
rest2 = rest.drop(index=loyal_id)
lapsing_id = rest2["recency"].idxmax()
names = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars", lapsing_id: "Lapsing / At-Risk"}
for sid in prof.index:
    names.setdefault(sid, "Occasional Shoppers")
actions = {"Premium Big Spenders": "VIP treatment: early access to Electronics offers and priority support.",
           "Loyal Regulars": "Reward habit: loyalty points and subscription-style bundles to protect frequency.",
           "Lapsing / At-Risk": "Win-back: time-limited discount and reminder before they churn.",
           "Occasional Shoppers": "Nudge: first-order-of-the-week coupons to raise frequency."}
for sid in prof.index:
    print(f"  segment {sid}: {names[sid]} -> {actions[names[sid]]}")
print("  (cluster numbers are arbitrary labels with no ordering)")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)
pca = PCA(n_components=2).fit(X)
print("PCA explained variance ratio:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))

# ============================ Task 4: lapse problem ============================
CUT = pd.Timestamp("2025-09-01")
pre = orders[orders.order_ts < CUT]
pre_c = pre[pre.status == "completed"]
cohort_ids = pre_c.customer_id.unique()
sept = comp[(comp.order_ts >= CUT) & (comp.order_ts < pd.Timestamp("2025-10-01"))]
active = set(sept.customer_id)
f = pre_c.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                     monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                     avg_delivery=("delivery_minutes", "mean"))
f["recency"] = (CUT - f["last"]).dt.days
fin = pre[pre.status.isin(["completed", "cancelled"])]
f["cancel_rate"] = (fin.assign(c=fin.status == "cancelled").groupby("customer_id")["c"].mean()).reindex(f.index).fillna(0)
cat_count = pre_c.groupby(["customer_id", "category"]).size().reset_index(name="n").sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
f["fav_category"] = cat_count.drop_duplicates("customer_id").set_index("customer_id")["category"]
f = f.drop(columns="last").reset_index().merge(cust, on="customer_id", how="left")
f["tenure_days"] = (CUT - f["signup_date"]).dt.days
f["lapsed"] = (~f["customer_id"].isin(active)).astype(int)
num = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
nom = ["city", "gender", "fav_category"]
y = f["lapsed"]; Xf = f[["customer_id"] + num + nom]
print(f"\n=== lapse cohort: {len(f)} customers, {y.sum()} lapsed ({y.mean()*100:.2f}%)")
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=42)
ids_te = Xte["customer_id"].values
Xtr, Xte = Xtr.drop(columns="customer_id"), Xte.drop(columns="customer_id")
print("train/test:", len(Xtr), len(Xte))
base_pred = np.zeros(len(yte), dtype=int)
print(f"baseline accuracy {accuracy_score(yte, base_pred)*100:.1f}%  lapsed F1 {f1_score(yte, base_pred, zero_division=0)}")

# ============================ Task 5: preprocessing ============================
# avg_delivery is missing when every pre-cutoff delivery time is missing; impute inside the pipeline (train-fitted)
from sklearn.impute import SimpleImputer
def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nom)])
def pipe(model): return Pipeline([("prep", prep()), ("model", model)])
tmp = pipe(LogisticRegression(max_iter=1000)); tmp.fit(Xtr, ytr)
learned = tmp.named_steps["prep"].named_transformers_["num"].named_steps["sc"].mean_[num.index("frequency")]
print(f"scaler mean(frequency) learned on train {learned:.4f} vs full-cohort mean {f.frequency.mean():.4f} "
      "-> they differ because the scaler only saw the 924 training rows, never the test rows.")
print("missing in raw features (imputed in-pipeline):", int(Xf[num + nom].isna().sum().sum()))
Xt_check = tmp.named_steps["prep"].transform(Xtr); Xe_check = tmp.named_steps["prep"].transform(Xte)
assert not np.isnan(Xt_check).any() and not np.isnan(Xe_check).any(); print("no missing value in the transformed feature matrices: OK")

# ============================ Task 6: train, tune, evaluate ============================
def lapsed_metrics(m, X, yy):
    p = m.predict(X)
    return dict(cm=confusion_matrix(yy, p), prec=precision_score(yy, p, zero_division=0),
                rec=recall_score(yy, p, zero_division=0), f1=f1_score(yy, p, zero_division=0))
gs_lr = GridSearchCV(pipe(LogisticRegression(max_iter=1000)), {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
gs_rf = GridSearchCV(pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                     {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\nLR best", gs_lr.best_params_, "cv F1", round(gs_lr.best_score_, 3))
print("RF best", gs_rf.best_params_, "cv F1", round(gs_rf.best_score_, 3))
m_lr, m_rf = lapsed_metrics(gs_lr.best_estimator_, Xte, yte), lapsed_metrics(gs_rf.best_estimator_, Xte, yte)
for n, m in [("LogReg", m_lr), ("RandForest", m_rf)]:
    print(f"{n}: confusion {m['cm'].tolist()}  precision {m['prec']:.3f} recall {m['rec']:.3f} F1 {m['f1']:.3f}")
best_name, best_gs, best_m = ("RandomForest", gs_rf, m_rf) if m_rf["f1"] >= m_lr["f1"] else ("LogisticRegression", gs_lr, m_lr)
best = best_gs.best_estimator_
print("selected:", best_name, "test F1", round(best_m["f1"], 3), "(>= 0.55:", best_m["f1"] >= 0.55, ")")

# ============================ Task 7: class-balance experiment ============================
rng = np.random.RandomState(42)
n_neg, n_pos = int((ytr == 0).sum()), int((ytr == 1).sum())
extra = rng.choice(np.where(ytr.values == 1)[0], size=n_neg - n_pos, replace=True)
Xbal = pd.concat([Xtr, Xtr.iloc[extra]]); ybal = pd.concat([ytr, ytr.iloc[extra]])
print(f"\nbalanced train: {int((ybal==0).sum())} / {int((ybal==1).sum())}")
lr_bal = pipe(LogisticRegression(max_iter=1000, C=gs_lr.best_params_["model__C"])).fit(Xbal, ybal)
m_bal = lapsed_metrics(lr_bal, Xte, yte)
print(f"un-balanced LR: P {m_lr['prec']:.3f} R {m_lr['rec']:.3f} F1 {m_lr['f1']:.3f}")
print(f"balanced    LR: P {m_bal['prec']:.3f} R {m_bal['rec']:.3f} F1 {m_bal['f1']:.3f}")

# ============================ Task 8: interpret, persist, use ============================
names_out = best.named_steps["prep"].get_feature_names_out()
if best_name == "RandomForest":
    imp = pd.Series(best.named_steps["model"].feature_importances_, index=names_out).sort_values(ascending=False)
    print("\ntop-5 RF importances:"); print(imp.head(5).round(3))
rf_imp = pd.Series(gs_rf.best_estimator_.named_steps["model"].feature_importances_,
                   index=gs_rf.best_estimator_.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False)
print("\ntop-5 RF importances (best random forest):"); print(rf_imp.head(5).round(3))
coef = pd.Series(gs_lr.best_estimator_.named_steps["model"].coef_[0],
                 index=gs_lr.best_estimator_.named_steps["prep"].get_feature_names_out())
print("LR coefficients:"); print(coef[["num__recency", "num__frequency", "num__monetary"]].round(3))
joblib.dump(best, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
assert (reloaded.predict(Xte) == best.predict(Xte)).all(); print("reloaded model reproduces test predictions exactly: OK")
proba = best.predict_proba(Xte)[:, 1]
res = pd.DataFrame({"customer_id": ids_te, "lapse_probability": proba, "actually_lapsed": yte.values})
prem = set(rfm.loc[rfm.is_premium, "customer_id"])
ret = res[res.customer_id.isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
ret.round({"lapse_probability": 4}).to_csv("outputs/retention_list.csv", index=False)
print(f"\nretention list: {len(ret)} premium customers, {int(ret.actually_lapsed.sum())} truly lapsed "
      "(illustrative only: the Premium flag uses whole-window RFM including September, so this is not a leakage-free evaluation)")
json.dump({"cohort": len(f), "lapsed": int(y.sum()), "best_model": best_name}, open("outputs/part_b_stats.json", "w"))
