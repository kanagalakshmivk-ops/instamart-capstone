"""Part A: audit, clean, store, SQL and pandas analysis. Run: python part_a.py"""
import json, os, re, sqlite3
import numpy as np, pandas as pd

RAW, CLEAN = "data/raw", "data/clean"
os.makedirs(CLEAN, exist_ok=True); os.makedirs("outputs", exist_ok=True)
stats = {}   # numbers quoted in the README

# ---------------------------------------------------------------- audit
orders_raw = pd.read_csv(f"{RAW}/orders.csv")
customers_raw = pd.read_csv(f"{RAW}/customers.csv")
products_raw = pd.read_csv(f"{RAW}/products.csv")
tickets_raw = pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False)
for name, df in [("orders", orders_raw), ("customers", customers_raw), ("products", products_raw), ("tickets", tickets_raw)]:
    print(f"\n=== {name}: shape {df.shape}"); print(pd.DataFrame({
        "dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2)}))

# string-level view so that "N/A" is not silently turned into NaN
o_str = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)
c_str = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)
amt_num = pd.to_numeric(o_str["order_amount"], errors="coerce")
dm_num = pd.to_numeric(o_str["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(c_str["age"], errors="coerce")
gen = c_str["gender"].str.strip()
audit = {
    "raw orders rows": len(o_str),
    "exact duplicate rows": int(o_str.duplicated().sum()),
    "pending rows": int((o_str["status"] == "pending").sum()),
    "distinct order_id": int(o_str["order_id"].nunique()),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((dm_num < 0).sum()),
    "blank delivery_minutes": int((o_str["delivery_minutes"].str.strip() == "").sum()),
    "blank age": int(age_num.isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": int(c_str["city"].nunique()),
    "distinct non-blank raw gender spellings": int(gen[gen != ""].nunique()),
    "blank gender": int((gen == "").sum()),
}
REF_AUDIT = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]
audit_df = pd.DataFrame({"figure": audit.keys(), "value": audit.values(), "reference": REF_AUDIT})
print("\n=== audit figures"); print(audit_df.to_string(index=False))
stats["audit"] = audit

# ---------------------------------------------------------------- clean customers
cust = customers_raw.copy()
cust["city"] = cust["city"].str.strip().str.title()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
cust["gender"] = cust["gender"].fillna("").astype(str).str.strip().str.lower().map(gmap).fillna("unknown")
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
bad_age = ((cust["age"] < 10) | (cust["age"] > 100))
stats["age_set_missing"] = int(bad_age.sum())
cust.loc[bad_age, "age"] = np.nan
stats["age_missing_before_fill"] = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
print("\n=== clean customers"); print(cust["city"].value_counts().to_string()); print(cust["gender"].value_counts().to_string())
print("missing ages:", cust["age"].isna().sum(), "| age range:", cust["age"].min(), cust["age"].max())

# ---------------------------------------------------------------- clean orders
o = orders_raw.copy()
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])                                   # rule 1
o = o.drop_duplicates();                         stats["removed_exact_dups"] = n0 - len(o)       # rule 2
n1 = len(o)
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")          # rule 3
stats["removed_older_records"] = n1 - len(o)
o = o.sort_values("order_id").reset_index(drop=True)
prod = products_raw.set_index("product_id")
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")           # rule 4
miss_amt = o["order_amount"].isna()
stats["amounts_filled"] = int(miss_amt.sum())
calc = (o["product_id"].map(prod["price_inr"]) * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o.loc[miss_amt, "order_amount"] = calc[miss_amt]
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # rule 5
neg = o["delivery_minutes"] < 0
stats["negative_set_missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
o["_cat"] = o["product_id"].map(prod["category"])
comp = o["status"] == "completed"
med = o[comp].groupby("_cat")["delivery_minutes"].transform("median")
fill_mask = comp & o["delivery_minutes"].isna()
stats["delivery_filled_completed"] = int(fill_mask.sum())
o.loc[fill_mask, "delivery_minutes"] = med[fill_mask]
o = o.drop(columns="_cat")
comp = o["status"] == "completed"
checks = {
    "rows": len(o), "completed": int(comp.sum()), "cancelled": int((o.status == "cancelled").sum()),
    "pending": int((o.status == "pending").sum()), "missing_amount": int(o.order_amount.isna().sum()),
    "missing_delivery_completed": int(o.loc[comp, "delivery_minutes"].isna().sum()),
    "missing_delivery_cancelled": int(o.loc[~comp, "delivery_minutes"].isna().sum()),
    "completed_revenue": round(float(o.loc[comp, "order_amount"].sum()), 2)}
print("\n=== clean orders"); print(checks); stats["orders_checks"] = checks; print(stats)

# ---------------------------------------------------------------- store
o_out, c_out = o.copy(), cust.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
products_raw.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists("data/instamart.db"): os.remove("data/instamart.db")
with sqlite3.connect("data/instamart.db") as con:
    c_out.to_sql("customers", con, index=False); products_raw.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)

# ---------------------------------------------------------------- SQL analysis
sql_text = open("sql/queries.sql").read()
queries = [q.strip() for q in re.split(r"(?m)^-- Q\d.*$", sql_text) if q.strip()]
assert len(queries) == 7
res = {}
with sqlite3.connect("data/instamart.db") as con:
    for i, q in enumerate(queries, 1):
        res[i] = pd.read_sql_query(q, con)
print("\n=== SQL results")
print("Q1 customers:", len(res[1]))
print("Q2 rows:", len(res[2]), "total:", round(res[2].revenue.sum(), 2)); print(res[2].head(1).to_string(index=False))
print("Q3 products:", len(res[3]))
print("Q4 customers without completed order:", len(res[4]))
print("Q5 customers above avg:", len(res[5]), "avg:", res[5].avg_total.iloc[0])
q6 = res[6]; print("Q6 top per city:"); print(q6[q6.row_num == 1].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
print("Q7 days:", len(res[7]), "last cumulative:", res[7].cumulative_revenue.iloc[-1])

# ---------------------------------------------------------------- pandas cross-check
co = o.merge(cust[["customer_id", "city"]], on="customer_id").merge(products_raw[["product_id", "category"]], on="product_id")
cc = co[co.status == "completed"]
pt = pd.pivot_table(cc, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\n=== pivot grand total:", round(pt.loc["All", "All"], 2), "| SQL total:", round(res[2].revenue.sum(), 2))
city_stats = cc.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), n=("order_amount", "count")).round(2)
print(city_stats)
x = cc["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} median {x.median():.2f} skew {x.skew():.2f} Q1 {q1:.2f} Q3 {q3:.2f} "
      f"fences [{lo:.2f}, {hi:.2f}] below {int((x < lo).sum())} above {int((x > hi).sum())}")
print(cc[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3))
json.dump(stats, open("outputs/part_a_stats.json", "w"), indent=2)
